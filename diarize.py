#!/usr/bin/env python3
"""화자 분리: 원본 오디오를 분석하고 화자별 클립과 합본을 저장합니다."""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import subprocess
import tempfile


def to_wav(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        check=True, capture_output=True,
    )


def fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def export_speaker_audio(source, outdir, segments):
    """Concatenate only clips extracted by this call; ignore earlier output files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    current = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="clips-", dir=outdir) as work:
        for i, (start, end, speaker) in enumerate(segments):
            clip = Path(work) / f"{speaker}_{i:03d}_{fmt(start).replace(':', '-')}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(source), "-ss", str(start),
                 "-t", str(end - start), "-ac", "1", "-ar", "16000", str(clip)],
                check=True, capture_output=True,
            )
            current[speaker].append(clip)
        for speaker, clips in current.items():
            manifest = Path(work) / f"_{speaker}_list.txt"
            manifest.write_text("".join(f"file '{clip.name}'\n" for clip in clips), encoding="utf-8")
            merged = outdir / f"MERGED_{speaker}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
                 "-c", "copy", str(merged)], check=True, capture_output=True,
            )
            for clip in clips:
                clip.replace(outdir / clip.name)
            print(f"      -> {merged}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="입력 오디오 파일")
    parser.add_argument("--output-dir", type=Path, default=Path("segments"))
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"입력 파일을 찾을 수 없습니다: {args.input}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("환경변수 HF_TOKEN이 없습니다. pyannote 모델 접근 토큰을 설정하세요.")

    import torch
    from pyannote.audio import Pipeline

    with tempfile.TemporaryDirectory(prefix="diarize-") as work:
        wav = Path(work) / "input_16k_mono.wav"
        print(f"[1/4] {args.input}: 16 kHz mono 변환 중...")
        to_wav(args.input, wav)
        print("[2/4] pyannote 모델 로딩 중...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
        print("[3/4] 화자 분리 분석 중...")
        output = pipeline(str(wav))
        full = getattr(output, "speaker_diarization", output)
        # Exclusive labels assign one speaker per instant; overlapping voices may remain in the audio.
        clean = getattr(output, "exclusive_speaker_diarization", full)
        talk_time = defaultdict(float)
        for turn, _, speaker in full.itertracks(yield_label=True):
            talk_time[speaker] += turn.end - turn.start
        segments = [(turn.start, turn.end, speaker)
                    for turn, _, speaker in clean.itertracks(yield_label=True)]

    print("\n===== 화자별 총 발화 시간 =====")
    for speaker, duration in sorted(talk_time.items(), key=lambda item: -item[1]):
        print(f"  {speaker}: {duration:6.1f}초")
    for start, end, speaker in segments:
        print(f"  {fmt(start)} ~ {fmt(end)}  [{speaker}]  ({end - start:.1f}s)")
    print(f"[4/4] 화자별 클립 및 합본 저장: {args.output_dir}")
    export_speaker_audio(args.input, args.output_dir, segments)
    print("완료! 이번에 출력된 MERGED_*.wav를 듣고 타겟 화자를 선택하세요.")


if __name__ == "__main__":
    main()
