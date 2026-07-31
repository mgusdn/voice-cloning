#!/usr/bin/env python3
"""
슬라이싱된 오디오 세그먼트들을 faster-whisper로 자동 전사(ASR)하여
GPT-SoVITS 학습용 .list 파일을 생성.

GPT-SoVITS가 요구하는 포맷:
    오디오경로|화자이름|언어코드|텍스트

사용법:
    python 02_transcribe.py ./sliced_output 화자이름 ./output.list

사전 설치:
    pip install faster-whisper --break-system-packages
"""

import sys
import os
import glob
from faster_whisper import WhisperModel

# GPT-SoVITS 언어 코드: ko (한국어), en, ja, zh 등
LANGUAGE_CODE = "ko"
MODEL_SIZE = "large-v3"  # 정확도 중요하면 large-v3, 속도 중요하면 medium


def transcribe_segments(input_dir: str, speaker_name: str, output_list_path: str):
    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not wav_files:
        print(f"오류: {input_dir}에 wav 파일이 없습니다.")
        sys.exit(1)

    print(f"[1/2] faster-whisper 모델 로드 중 ({MODEL_SIZE})...")
    # Apple Silicon은 CPU 또는 int8 권장 (MPS는 faster-whisper 미지원)
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    print(f"[2/2] {len(wav_files)}개 세그먼트 전사 중...")
    entries = []
    skipped = 0

    for i, wav_path in enumerate(wav_files):
        segments, info = model.transcribe(wav_path, language=LANGUAGE_CODE, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()

        if not text:
            skipped += 1
            continue

        entries.append(f"{wav_path}|{speaker_name}|{LANGUAGE_CODE}|{text}")

        if (i + 1) % 10 == 0:
            print(f"  진행: {i + 1}/{len(wav_files)}")

    with open(output_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))

    print("")
    print(f"완료! 리스트 파일: {output_list_path}")
    print(f"성공: {len(entries)}개, 스킵(빈 텍스트): {skipped}개")
    print("이 .list 파일을 GPT-SoVITS webui의 '학습 데이터 처리' 단계에 넣으면 됩니다.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"사용법: python {sys.argv[0]} 세그먼트폴더 화자이름 출력.list")
        sys.exit(1)

    input_dir = sys.argv[1]
    speaker_name = sys.argv[2]
    output_list_path = sys.argv[3]
    transcribe_segments(input_dir, speaker_name, output_list_path)
