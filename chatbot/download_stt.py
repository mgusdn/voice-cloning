"""Downloads a ggml model for whisper.cpp (via pywhispercpp).

Run:
    python -m chatbot.download_stt          # downloads small (default)
    python -m chatbot.download_stt medium   # or any name in pywhispercpp.constants.AVAILABLE_MODELS
"""
import sys

from pywhispercpp.utils import download_model

DEFAULT_MODEL = "small"


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    path = download_model(model_name)
    print(f"[download_model] '{model_name}' ready at: {path}")


if __name__ == "__main__":
    main()
