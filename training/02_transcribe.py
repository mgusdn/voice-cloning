#!/usr/bin/env python3
"""Transcribe WAV clips into absolute-path GPT-SoVITS training records."""

import argparse
from pathlib import Path

LANGUAGE_CODE = "ko"
MODEL_SIZE = "large-v3"


def transcribe_segments(input_dir, speaker_name, output_list_path, *,
                        language=LANGUAGE_CODE, model_size=MODEL_SIZE,
                        device="cpu", compute_type="int8"):
    if not speaker_name.strip() or any(char in speaker_name for char in "|\r\n"):
        raise ValueError("Speaker name must be nonempty and cannot contain | or newlines")
    if not language.strip() or any(char in language for char in "|\r\n"):
        raise ValueError("Language must be a nonempty GPT-SoVITS language code")
    output = Path(output_list_path).expanduser()
    if output.suffix.lower() != ".list":
        raise ValueError("Output filename must end in .list")
    output = output.resolve()
    speaker_name = speaker_name.strip()
    directory = Path(input_dir).expanduser().resolve()
    wav_files = sorted(path for path in directory.glob("*")
                       if path.is_file() and path.suffix.lower() == ".wav")
    if not wav_files:
        raise ValueError(f"No WAV files in {directory}")
    if any(any(char in str(path) for char in "|\r\n") for path in wav_files):
        raise ValueError("WAV paths cannot contain | or newlines in a training list")
    if output in {path.resolve() for path in wav_files}:
        raise ValueError("Output list cannot overwrite an input WAV file")

    from faster_whisper import WhisperModel

    print(f"faster-whisper 모델 로드 중: {model_size}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    entries = []
    for index, path in enumerate(wav_files):
        segments, _ = model.transcribe(str(path), language=language, beam_size=5)
        text = " ".join("".join(segment.text for segment in segments).replace("|", " ").split())
        if text:
            entries.append(f"{path}|{speaker_name}|{language}|{text}")
        if (index + 1) % 10 == 0:
            print(f"진행: {index + 1}/{len(wav_files)}")
    if not entries:
        raise ValueError("No nonempty transcriptions; the existing output list was not changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"완료: {len(entries)}개, 빈 텍스트 스킵: {len(wav_files) - len(entries)}개, {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("speaker")
    parser.add_argument("output_list")
    parser.add_argument("--language", default=LANGUAGE_CODE)
    parser.add_argument("--model", default=MODEL_SIZE)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args(argv)
    try:
        transcribe_segments(args.input_dir, args.speaker, args.output_list,
                            language=args.language, model_size=args.model,
                            device=args.device, compute_type=args.compute_type)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"전사 실패: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
