"""Save a cloned voice using the same API client as the chatbot."""
import argparse
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GPT-SoVITS API로 텍스트를 WAV로 합성합니다.")
    parser.add_argument("text", help="합성할 텍스트")
    parser.add_argument("--output", type=Path, default=Path("runtime/output.wav"))
    parser.add_argument("--play", action="store_true", help="생성과 동시에 ffplay로 재생")
    args = parser.parse_args(argv)
    from .tts_client import TTSClient, TTSError

    try:
        output = TTSClient().synthesize(args.text, args.output, play=args.play)
    except TTSError as exc:
        parser.exit(1, f"오류: {exc}\n")
    print(f"저장 완료: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
