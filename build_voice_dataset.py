"""
GPT-SoVITS 학습용 화자 데이터셋 빌더.

diarize.py 로 뽑은 화자 합본(MERGED_*.wav) 하나를 받아:
  1) demucs 로 보컬만 분리(BGM/잡음 제거)
  2) faster-whisper(VAD+large-v3)로 문장 단위 슬라이스 + 한국어 전사
  3) dataset/<화자>/wavs/ + <화자>.list(GPT-SoVITS 학습 포맷) + <화자>.colab.list
     + transcript.jsonl(검수용) 로 저장

사용법:
    python build_voice_dataset.py <입력.wav> <화자이름>
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

if len(sys.argv) < 3:
    sys.exit(f"사용법: python {sys.argv[0]} <입력.wav> <화자이름>")

INPUT = Path(sys.argv[1])
SPEAKER = sys.argv[2]

OUT_DIR = Path("dataset") / SPEAKER
WAVS_DIR = OUT_DIR / "wavs"
WORK = Path("_ds_work") / SPEAKER

SR = 24000
MIN_SEC = 3.0
MAX_SEC = 15.0
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("DEMUCS_DEVICE", "cpu")


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def separate_vocals(src: Path, workdir: Path) -> Path:
    """demucs 로 보컬만 분리 -> vocals.wav 경로 반환."""
    run(["demucs", "--two-stems=vocals", "-d", DEVICE, "-o", str(workdir), str(src)])
    matches = list(workdir.rglob("vocals.wav"))
    if not matches:
        sys.exit("demucs 보컬 분리 결과를 찾지 못했습니다.")
    return matches[0]


def to_wav_24k(src: Path, dst: Path):
    run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)])


def fmt(t: float) -> str:
    return f"{t:.1f}s"


def main():
    if not INPUT.exists():
        sys.exit(f"입력 파일을 찾을 수 없습니다: {INPUT}")

    WORK.mkdir(parents=True, exist_ok=True)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] demucs 로 보컬 분리 중... ({INPUT}, device={DEVICE})")
    vocals = separate_vocals(INPUT, WORK)

    print("[2/4] 24kHz mono 로 변환 중...")
    clean = WORK / "vocals_24k.wav"
    to_wav_24k(vocals, clean)

    print(f"[3/4] faster-whisper({WHISPER_MODEL}) 로 VAD+전사 중... (시간 걸릴 수 있음)")
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = whisper.transcribe(str(clean), language="ko", vad_filter=True)
    asr = [(seg.start, seg.end, seg.text.strip()) for seg in segments if seg.text.strip()]
    print(f"      -> whisper 세그먼트 {len(asr)}개")

    audio, sr = sf.read(str(clean))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print(f"[4/4] {MIN_SEC}~{MAX_SEC}초 구간만 클립 저장 + .list 작성 중...")
    list_lines, colab_lines, transcript = [], [], []
    abs_wavs = WAVS_DIR.resolve()
    i = 0
    for start, end, text in asr:
        dur = end - start
        if dur < MIN_SEC or dur > MAX_SEC:
            continue
        clip = audio[int(start * sr):int(end * sr)]
        name = f"{SPEAKER}_{i:03d}_{fmt(dur)}.wav"
        out = WAVS_DIR / name
        sf.write(str(out), clip, sr)
        list_lines.append(f"{abs_wavs / name}|{SPEAKER}|ko|{text}")
        colab_lines.append(f"/content/{SPEAKER}/wavs/{name}|{SPEAKER}|ko|{text}")
        transcript.append({"file": name, "start": start, "end": end, "text": text})
        i += 1

    (OUT_DIR / f"{SPEAKER}.list").write_text("\n".join(list_lines) + "\n", encoding="utf-8")
    (OUT_DIR / f"{SPEAKER}.colab.list").write_text("\n".join(colab_lines) + "\n", encoding="utf-8")
    with open(OUT_DIR / "transcript.jsonl", "w", encoding="utf-8") as fh:
        for rec in transcript:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_dur = sum(r["end"] - r["start"] for r in transcript)
    print(f"\n완료! {OUT_DIR}/")
    print(f"  클립 {i}개, 총 {total_dur/60:.1f}분")
    print(f"  -> {OUT_DIR}/wavs/, {SPEAKER}.list, {SPEAKER}.colab.list, transcript.jsonl")


if __name__ == "__main__":
    main()
