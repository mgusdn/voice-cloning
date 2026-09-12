#!/usr/bin/env python3
"""
유튜브 오디오 mp3 추출 + 특정 구간 제외 스크립트

사용법:
    python youtube.py "유튜브URL"

사전 설치 필요:
    python -m pip install -r requirements-data.txt
    (ffmpeg는 별도 설치 필요: brew install ffmpeg / apt install ffmpeg)
"""

import sys
import subprocess
import shutil

# ===== 여기서 제외할 구간을 초 단위로 설정 (영상마다 다시 설정 필요) =====
EXCLUDE_RANGES = [
]
# ==============================================

TMP_FILE = "full_audio.mp3"
OUT_FILE = "output.mp3"


def check_dependencies():
    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            print(f"오류: '{tool}'가 설치되어 있지 않습니다.")
            sys.exit(1)


def download_audio(url: str, out_path: str):
    print("[1/3] 유튜브에서 오디오 추출 중...")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        url,
        "-o", out_path,
    ]
    subprocess.run(cmd, check=True)


def build_filter(ranges):
    print("[2/3] 제외 구간 필터 생성 중...")
    parts = [f"between(t,{start},{end})" for start, end in ranges]
    filter_expr = "+".join(parts)
    print(f"생성된 필터: not({filter_expr})")
    return filter_expr


def apply_filter(input_path: str, output_path: str, filter_expr: str):
    print("[3/3] 구간 제외 후 최종 mp3 생성 중...")
    af = f"aselect='not({filter_expr})', asetpts=N/SR/TB"
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", af,
        "-vn", output_path,
    ]
    subprocess.run(cmd, check=True)


def main():
    if len(sys.argv) < 2:
        print(f"사용법: python {sys.argv[0]} \"유튜브URL\"")
        sys.exit(1)

    url = sys.argv[1]

    check_dependencies()
    download_audio(url, TMP_FILE)
    if EXCLUDE_RANGES:
        filter_expr = build_filter(EXCLUDE_RANGES)
        apply_filter(TMP_FILE, OUT_FILE, filter_expr)
    else:
        print("[2/3] 제외 구간 없음 - 원본 그대로 사용")
        shutil.copyfile(TMP_FILE, OUT_FILE)

    print()
    print(f"완료! 결과 파일: {OUT_FILE}")
    print(f"원본 파일({TMP_FILE})은 확인 후 삭제하셔도 됩니다.")


if __name__ == "__main__":
    main()
