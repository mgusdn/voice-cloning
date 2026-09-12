#!/usr/bin/env python3
"""Extract one reference clip as a 24 kHz mono WAV using ffmpeg."""
import argparse
import math
from pathlib import Path
import subprocess
import tempfile
import wave


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="원본 오디오")
    parser.add_argument("output", type=Path, help="참조 음성 .wav 출력 경로")
    parser.add_argument("--start", required=True, type=float, help="시작 시각 (초)")
    parser.add_argument("--end", required=True, type=float, help="종료 시각 (초)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.start) or not math.isfinite(args.end) or args.start < 0 or args.end <= args.start:
        parser.error("--start는 0 이상이어야 하고 --end는 --start보다 커야 합니다.")
    if not args.input.is_file():
        parser.error(f"입력 오디오를 찾을 수 없습니다: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("입력과 출력은 서로 다른 파일이어야 합니다.")
    if args.output.suffix.lower() != ".wav":
        parser.error("출력 파일 확장자는 .wav여야 합니다.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".reference-", suffix=".wav",
                                         dir=args.output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.input),
             "-ss", str(args.start), "-t", str(args.end - args.start), "-ac", "1",
             "-ar", "24000", "-c:a", "pcm_s16le", str(temporary)],
            check=True,
        )
        with wave.open(str(temporary), "rb") as audio:
            if audio.getnframes() <= 0 or not audio.readframes(1):
                raise ValueError("추출된 음성이 비어 있습니다. --start와 --end가 원본 길이 안에 있는지 확인하세요.")
            duration = audio.getnframes() / audio.getframerate()
        temporary.replace(args.output)
    except FileNotFoundError:
        parser.error("ffmpeg 또는 입력/출력 경로를 찾을 수 없습니다. ffmpeg는 brew install ffmpeg로 설치할 수 있습니다.")
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"참조 음성 추출 실패 (ffmpeg 종료 코드 {exc.returncode}).\n")
    except (OSError, wave.Error, EOFError, ValueError) as exc:
        parser.exit(1, f"참조 음성 추출 실패: {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"참조 음성 생성: {args.output} ({duration:.2f}초, 24 kHz mono)")


if __name__ == "__main__":
    main()
