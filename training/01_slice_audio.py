#!/usr/bin/env python3
"""Slice speech into <=10s WAVs, preserving isolated clips shorter than 3s.

Long utterances are divided evenly so that a small final remainder is not lost.
Use a fresh output directory for each source; existing segments are never replaced.
"""

import argparse
from pathlib import Path

MIN_LEN_SEC = 3.0
MAX_LEN_SEC = 10.0
TOP_DB = 35
SR_TARGET = 32000


def segment_ranges(intervals, sample_rate):
    """Return ordered sample ranges, merging neighbors only within the 10s cap.

An isolated speech interval shorter than 3s is retained if merging across the
next silence would exceed 10s. These clips require review before training.
"""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    maximum = int(MAX_LEN_SEC * sample_rate)
    if maximum < 1:
        raise ValueError("sample_rate is too small")
    pieces = []
    previous_end = 0
    for start, end in intervals:
        start, end = int(start), int(end)
        if start < previous_end or end < start:
            raise ValueError("Speech intervals must be ordered and non-overlapping")
        previous_end = end
        length = end - start
        if length == 0:
            continue
        count = (length + maximum - 1) // maximum
        # Integer boundaries partition the original samples exactly once.
        for index in range(count):
            pieces.append((start + length * index // count,
                           start + length * (index + 1) // count))

    merged = []
    for start, end in pieces:
        if merged and end - merged[-1][0] <= maximum:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def slice_audio(input_path, output_dir):
    output = Path(output_dir).expanduser().resolve()
    if any(output.glob("seg_*.wav")):
        raise FileExistsError(f"Existing segments in {output}; choose a fresh output directory")

    import librosa
    import soundfile as sf

    print(f"오디오 로드 중: {input_path}")
    audio, sample_rate = librosa.load(input_path, sr=SR_TARGET, mono=True)
    # librosa's relative threshold can classify an all-zero signal as one region.
    if not len(audio) or not any(audio):
        print("음성이 없습니다. 저장한 세그먼트: 0개")
        return []
    ranges = segment_ranges(librosa.effects.split(audio, top_db=TOP_DB), sample_rate)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (start, end) in enumerate(ranges):
        path = output / f"seg_{index:04d}.wav"
        sf.write(path, audio[start:end], sample_rate)
        paths.append(str(path))
        duration = (end - start) / sample_rate
        if duration < MIN_LEN_SEC:
            print(f"검토 필요: {path.name} ({duration:.2f}초) — 짧은 발화를 보존했습니다")
    print(f"완료: {len(paths)}개, {output}")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_audio")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    try:
        paths = slice_audio(args.input_audio, args.output_dir)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"슬라이싱 실패: {exc}\n")
    if not paths:
        parser.exit(1, "학습 가능한 음성이 없습니다. 입력 오디오를 확인하세요.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
