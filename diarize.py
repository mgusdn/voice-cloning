"""
화자 분리(diarization) 스크립트.
test.mp3 에서 "누가 언제 말했는지"를 분석하고,
각 화자별 발화 구간을 출력 + 화자별 오디오 클립으로 저장한다.

사용법:
    export HF_TOKEN=hf_xxxxx
    python diarize.py [입력파일]   # 기본값 test.mp3
"""

import os
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from pyannote.audio import Pipeline

INPUT = Path(sys.argv[1] if len(sys.argv) > 1 else "test.mp3")
OUTDIR = Path("segments")
WAV = Path("_input_16k_mono.wav")  # pyannote 권장 포맷

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    sys.exit("환경변수 HF_TOKEN이 없습니다.  export HF_TOKEN=hf_xxxxx 후 다시 실행하세요.")


def to_wav(src: Path, dst: Path):
    """mp3 -> 16kHz mono wav (pyannote가 가장 잘 처리하는 포맷)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
        capture_output=True,
    )


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def main():
    if not INPUT.exists():
        sys.exit(f"입력 파일을 찾을 수 없습니다: {INPUT}")

    print(f"[1/4] {INPUT} -> {WAV} 변환 중...")
    to_wav(INPUT, WAV)

    print("[2/4] pyannote 모델 로딩 중... (최초 1회 다운로드, 몇 분 걸릴 수 있음)")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=HF_TOKEN
    )
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))  # Apple Silicon 가속
        print("      -> Apple Silicon(MPS) 가속 사용")

    print("[3/4] 화자 분리 분석 중...")
    output = pipeline(str(WAV))

    # pyannote 4.0: output 객체에서 Annotation 꺼내기
    # - speaker_diarization: 전체(겹침 포함) -> 발화시간 집계용
    # - exclusive_speaker_diarization: 겹침 제거 -> 깨끗한 클립 추출용
    full = getattr(output, "speaker_diarization", output)
    clean = getattr(output, "exclusive_speaker_diarization", full)

    # 화자별 총 발화시간(전체 기준) 집계
    talk_time = defaultdict(float)
    for turn, _, speaker in full.itertracks(yield_label=True):
        talk_time[speaker] += turn.end - turn.start

    # 클립 추출은 "겹침 제거" 버전 사용 (다른 사람 목소리 섞인 구간 자동 배제)
    segments = []  # (start, end, speaker)
    for turn, _, speaker in clean.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    print("\n===== 화자별 총 발화 시간 =====")
    for spk, t in sorted(talk_time.items(), key=lambda x: -x[1]):
        print(f"  {spk}: {t:6.1f}초")

    print("\n===== 발화 구간 (시간순) =====")
    for start, end, spk in segments:
        print(f"  {fmt(start)} ~ {fmt(end)}  [{spk}]  ({end - start:.1f}s)")

    # 화자별로 오디오 클립 저장
    print(f"\n[4/4] 화자별 오디오 클립을 {OUTDIR}/ 에 저장 중...")
    OUTDIR.mkdir(exist_ok=True)
    for i, (start, end, spk) in enumerate(segments):
        out = OUTDIR / f"{spk}_{i:03d}_{fmt(start).replace(':','-')}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(INPUT), "-ss", str(start),
             "-to", str(end), "-ac", "1", "-ar", "16000", str(out)],
            check=True, capture_output=True,
        )

    # 화자별 전체를 이어붙인 합본도 생성 (TTS용 샘플로 쓰기 좋음)
    print("      화자별 합본(concat) 파일 생성 중...")
    for spk in talk_time:
        spk_files = sorted(OUTDIR.glob(f"{spk}_*.wav"))
        if not spk_files:
            continue
        listfile = OUTDIR / f"_{spk}_list.txt"
        listfile.write_text("".join(f"file '{f.name}'\n" for f in spk_files))
        merged = OUTDIR / f"MERGED_{spk}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c", "copy", str(merged)],
            check=True, capture_output=True,
        )
        print(f"      -> {merged}")

    print("\n완료! segments/ 폴더에서 화자별 클립과 MERGED_*.wav 를 확인하세요.")
    print("각 MERGED_화자.wav 를 들어보고, 원하는 사람이 누구인지 알려주시면 됩니다.")


if __name__ == "__main__":
    main()
