"""Synthesize with the external GPT-SoVITS api_v2 server, optionally playing live."""
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import uuid
import wave

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


class TTSError(RuntimeError):
    """Configuration, transport, or invalid-audio failure."""


@dataclass(frozen=True)
class TTSConfig:
    server_url: str = "http://127.0.0.1:9880"
    ref_audio_path: str = ""
    ref_text: str = ""
    language: str = "ko"

    @classmethod
    def from_env(cls, env_file: Path | None = None):
        load_dotenv(env_file or ROOT / ".env", override=False)
        return cls(
            server_url=os.getenv("TTS_SERVER_URL", "http://127.0.0.1:9880").rstrip("/"),
            ref_audio_path=os.getenv("TTS_REF_AUDIO_PATH", ""),
            ref_text=os.getenv("TTS_REF_TEXT", ""),
            language=os.getenv("TTS_LANGUAGE", "ko"),
        )


def _finalize_wav(path: Path) -> None:
    """Repair the zero-length WAV header emitted before streaming PCM chunks."""
    size = path.stat().st_size
    with path.open("r+b") as audio:
        header = audio.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise TTSError("TTS 서버가 WAV 오디오를 반환하지 않았습니다.")
        while audio.tell() + 8 <= size:
            chunk_id, chunk_size = struct.unpack("<4sI", audio.read(8))
            if chunk_id == b"data":
                data_size = size - audio.tell()
                if data_size == 0:
                    raise TTSError("TTS 서버의 오디오가 비어 있습니다.")
                if chunk_size not in (0, data_size):
                    raise TTSError("TTS 서버의 WAV 데이터 길이가 올바르지 않습니다.")
                audio.seek(-4, 1)
                audio.write(struct.pack("<I", data_size))
                audio.seek(4)
                audio.write(struct.pack("<I", size - 8))
                break
            audio.seek(chunk_size + chunk_size % 2, 1)
        else:
            raise TTSError("WAV 데이터 구간을 찾을 수 없습니다.")
    with wave.open(str(path), "rb") as audio:
        frame_size = audio.getnchannels() * audio.getsampwidth()
        if data_size % frame_size or audio.getnframes() == 0:
            raise TTSError("WAV 오디오 프레임이 불완전합니다.")


def _close_player(player, abort: bool) -> None:
    if player is None:
        return
    try:
        if player.stdin:
            player.stdin.close()
    except OSError:
        pass
    try:
        if abort:
            player.terminate()
        player.wait(timeout=3 if abort else 120)
    except subprocess.TimeoutExpired:
        player.kill()
        player.wait(timeout=3)
    except OSError:
        pass


class TTSClient:
    def __init__(self, config: TTSConfig | None = None, *, output_dir=None, transport=None):
        self.config = config if config is not None else TTSConfig.from_env()
        self.output_dir = Path(output_dir) if output_dir is not None else ROOT / "runtime" / "audio"
        self.session_dir = self.output_dir / uuid.uuid4().hex
        self.transport = transport if transport is not None else requests

    def speak(self, text: str | None, turn: int, play: bool = True) -> str | None:
        """Best-effort audio for a chatbot response; keep the text conversation alive."""
        if not text or not text.strip():
            return None
        try:
            return str(self.synthesize(text, self.session_dir / f"turn_{turn:03d}.wav", play=play))
        except TTSError as exc:
            print(f"  [TTS 건너뜀: {exc}]")
            return None

    def synthesize(self, text: str, output: Path, *, play: bool = False) -> Path:
        """Write a complete WAV atomically, raising TTSError on failure.

        ref_audio_path is interpreted on the TTS server, which can be a remote host.
        Streaming always uses batch_size=1. File-only synthesis uses non-streaming mode.
        """
        config = self.config
        if not config.ref_audio_path.strip() or not config.ref_text.strip():
            raise TTSError("루트 .env에 TTS_REF_AUDIO_PATH와 TTS_REF_TEXT를 설정하세요.")
        if not text.strip():
            raise TTSError("합성할 텍스트가 비어 있습니다.")
        output = Path(output)
        response = player = partial = None
        complete = False
        try:
            response = self.transport.post(
                f"{config.server_url.rstrip('/')}/tts",
                json={
                    "text": text, "text_lang": config.language,
                    "ref_audio_path": config.ref_audio_path,
                    "prompt_text": config.ref_text, "prompt_lang": config.language,
                    "media_type": "wav", "streaming_mode": 3 if play else 0,
                    "batch_size": 1,
                },
                timeout=(5, 60), stream=True,
            )
            response.raise_for_status()
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".part", delete=False) as audio:
                partial = Path(audio.name)
                if play:
                    try:
                        player = subprocess.Popen(
                            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "-"],
                            stdin=subprocess.PIPE,
                        )
                    except OSError:
                        print("  [ffplay 실행 불가: 파일 저장만 진행합니다. ffmpeg를 설치하세요.]")
                player_writable = player is not None
                for chunk in response.iter_content(chunk_size=4096):
                    if not chunk:
                        continue
                    audio.write(chunk)
                    if player_writable:
                        try:
                            player.stdin.write(chunk)
                            player.stdin.flush()
                        except OSError:
                            player_writable = False
            _finalize_wav(partial)
            partial.replace(output)
            complete = True
            return output
        except (requests.RequestException, OSError, wave.Error, EOFError, struct.error) as exc:
            raise TTSError(f"음성 생성에 실패했습니다 ({type(exc).__name__}). 서버와 설정을 확인하세요.") from exc
        finally:
            if response is not None:
                response.close()
            _close_player(player, abort=not complete)
            if partial is not None:
                partial.unlink(missing_ok=True)
