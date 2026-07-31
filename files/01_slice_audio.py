#!/usr/bin/env python3
"""
긴 음성 파일 하나를 GPT-SoVITS 학습에 적합한 짧은 세그먼트(3~10초)로 슬라이싱.

무음 구간을 기준으로 자르되, 너무 짧은 조각은 이어붙이고
너무 긴 조각은 다시 잘라서 목표 길이 범위에 맞춥니다.

사용법:
    python 01_slice_audio.py 원본.wav ./sliced_output

사전 설치:
    pip install librosa soundfile numpy --break-system-packages
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf

MIN_LEN_SEC = 3.0     # 최소 세그먼트 길이
MAX_LEN_SEC = 10.0    # 최대 세그먼트 길이
TOP_DB = 35           # 무음 판정 임계값 (낮을수록 민감하게 자름)
SR_TARGET = 32000      # GPT-SoVITS 권장 샘플레이트


def slice_audio(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/3] 오디오 로드 중: {input_path}")
    y, sr = librosa.load(input_path, sr=SR_TARGET, mono=True)

    print("[2/3] 무음 구간 기준으로 1차 분할 중...")
    intervals = librosa.effects.split(y, top_db=TOP_DB)

    # 짧은 조각들을 목표 길이 범위(MIN~MAX)로 병합/재분할
    segments = []
    buf_start = None
    buf_end = None

    for start, end in intervals:
        if buf_start is None:
            buf_start, buf_end = start, end
            continue

        merged_len = (end - buf_start) / sr
        if merged_len <= MAX_LEN_SEC:
            # 이어붙이기
            buf_end = end
        else:
            segments.append((buf_start, buf_end))
            buf_start, buf_end = start, end

    if buf_start is not None:
        segments.append((buf_start, buf_end))

    # 너무 짧은 세그먼트(MIN_LEN_SEC 미만)는 버리지 않고 다음 것과 합치기 시도
    final_segments = []
    i = 0
    while i < len(segments):
        s, e = segments[i]
        length = (e - s) / sr
        if length < MIN_LEN_SEC and i + 1 < len(segments):
            ns, ne = segments[i + 1]
            s, e = s, ne
            i += 2
        else:
            i += 1
        final_segments.append((s, e))

    print(f"[3/3] 총 {len(final_segments)}개 세그먼트로 분할, 저장 중...")

    list_lines = []
    for idx, (s, e) in enumerate(final_segments):
        clip = y[s:e]
        length_sec = len(clip) / sr
        if length_sec < 1.0:
            continue  # 너무 짧은 조각(1초 미만)은 노이즈일 가능성 높아 스킵

        fname = f"seg_{idx:04d}.wav"
        fpath = os.path.join(output_dir, fname)
        sf.write(fpath, clip, sr)
        list_lines.append(fpath)

    print(f"완료. 저장 위치: {output_dir}")
    print(f"세그먼트 개수: {len(list_lines)}개")
    return list_lines


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"사용법: python {sys.argv[0]} 원본오디오.wav 출력폴더")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    slice_audio(input_path, output_dir)
