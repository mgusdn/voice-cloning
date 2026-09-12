#!/usr/bin/env python3
"""Rebuild GPT-SoVITS labels from either retained transcript JSONL schema."""
import argparse
import json
from pathlib import Path, PurePosixPath


def rebuild_list(transcript, *, speaker=None, audio_dir=None, output=None, colab=None):
    transcript = Path(transcript).resolve()
    speaker = speaker or transcript.parent.name
    if not speaker or any(char in speaker for char in "|\r\n/\\"):
        raise ValueError("화자 이름에는 경로 구분자, 줄바꿈 또는 |를 사용할 수 없습니다.")
    audio_dir = Path(audio_dir).resolve() if audio_dir else transcript.parent / "wavs"
    if colab is not None:
        target = PurePosixPath(colab)
        if not target.is_absolute():
            raise ValueError("--colab에는 /content/<화자>/wavs 같은 절대 대상 경로를 지정하세요.")
    else:
        target = audio_dir
    suffix = ".colab.list" if colab is not None else ".list"
    output = Path(output) if output else transcript.parent / f"{speaker}{suffix}"
    if output.suffix.lower() != ".list":
        raise ValueError("--output 파일 확장자는 .list여야 합니다.")
    output = output.resolve()
    if output == transcript:
        raise ValueError("--output은 원본 transcript.jsonl과 다른 파일이어야 합니다.")
    entries = []
    for number, line in enumerate(transcript.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{transcript}:{number}: JSON 오류: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{transcript}:{number}: JSON 객체가 필요합니다.")
        source = record.get("file") or record.get("wav")
        text = record.get("text")
        if not isinstance(source, str) or not source or not isinstance(text, str) or not text.strip():
            raise ValueError(f"{transcript}:{number}: file 또는 wav, 비어 있지 않은 text가 필요합니다.")
        name = Path(source).name
        if name in ("", ".", "..") or any(char in name + text for char in "|\r\n"):
            raise ValueError(f"{transcript}:{number}: 파일 이름과 text에 줄바꿈 또는 |를 사용할 수 없습니다.")
        if output == (audio_dir / name).resolve():
            raise ValueError(f"--output은 입력 오디오와 다른 파일이어야 합니다: {audio_dir / name}")
        if colab is None and not (audio_dir / name).is_file():
            raise ValueError(f"오디오를 찾을 수 없습니다: {audio_dir / name}. "
                             "원본 wav를 복원하거나 --audio-dir로 오디오 폴더를 지정하세요. "
                             "Colab 업로드용이면 --colab /content/<화자>/wavs를 지정하세요.")
        entries.append(f"{target / name}|{speaker}|ko|{text}")
    if not entries:
        raise ValueError(f"전사 데이터가 없습니다: {transcript}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return output, len(entries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path, help="dataset/<화자>/transcript.jsonl")
    parser.add_argument("--speaker", help="화자 이름 (기본: 전사 파일의 상위 폴더 이름)")
    parser.add_argument("--audio-dir", type=Path, help="로컬 wav 폴더 (기본: 전사 파일 옆 wavs/)")
    parser.add_argument("--output", type=Path, help="생성할 .list 파일")
    parser.add_argument("--colab", metavar="TARGET_AUDIO_DIR", help="Colab의 절대 wav 경로. 로컬 오디오 존재 검사를 생략합니다.")
    args = parser.parse_args(argv)
    try:
        output, count = rebuild_list(**vars(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{output}: {count}개 라벨 생성")


if __name__ == "__main__":
    main()
