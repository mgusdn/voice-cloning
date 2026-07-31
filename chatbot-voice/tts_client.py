"""GPT-SoVITS(my_voice) TTS 클라이언트.

files/GPT-SoVITS를 api_v2.py로 띄워둔 상태에서, chatbot의 bot_message를
클로닝된 목소리로 합성해 재생/저장한다.

chatbot 저장소(juminsuh/chatbot)를 순수 upstream 클론으로 유지하기 위해,
이 연동 코드는 일부러 chatbot/ 밖 이 폴더에 둔다. chatbot/cli.py에서는
sys.path로 이 폴더를 추가해서 import한다 (README 참고).

사전 준비 (별도 터미널):
    cd ../files/GPT-SoVITS
    source venv/bin/activate
    python api_v2.py -a 127.0.0.1 -p 9880
    curl "http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_weights_v2Pro/my_voice-e15.ckpt"
    curl "http://127.0.0.1:9880/set_sovits_weights?weights_path=SoVITS_weights_v2Pro/my_voice_e8_s352.pth"
"""
import os
import subprocess
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

TTS_SERVER_URL = os.environ.get("TTS_SERVER_URL", "http://127.0.0.1:9880")
TTS_REF_AUDIO_PATH = os.environ.get("TTS_REF_AUDIO_PATH")
TTS_REF_TEXT = os.environ.get("TTS_REF_TEXT")

AUDIO_OUT_DIR = Path(__file__).resolve().parent / "audio_out"


def speak(text: str, turn: int, play: bool = True) -> str | None:
    """text를 my_voice 목소리로 합성한다. TTS 서버가 안 떠있으면 조용히 건너뛴다.

    streaming_mode로 청크가 오는 대로 ffplay에 흘려보내 재생하면서 동시에
    파일로 저장한다 (afplay는 완성된 파일만 재생 가능해서 스트리밍엔 못 씀).
    batch_size는 streaming_mode와 같이 쓰면 GPT 디코딩 텐서 shape이 안 맞아
    서버가 에러를 내므로 절대 같이 쓰지 않는다.
    """
    if not text:
        return None
    if not TTS_REF_AUDIO_PATH or not TTS_REF_TEXT:
        print("  [TTS 건너뜀: .env에 TTS_REF_AUDIO_PATH / TTS_REF_TEXT를 설정하세요]")
        return None

    tts_start = time.perf_counter()
    try:
        resp = requests.get(
            f"{TTS_SERVER_URL}/tts",
            params={
                "text": text,
                "text_lang": "ko",
                "ref_audio_path": TTS_REF_AUDIO_PATH,
                "prompt_text": TTS_REF_TEXT,
                "prompt_lang": "ko",
                "media_type": "wav",
                "streaming_mode": 3,
            },
            timeout=60,
            stream=True,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [TTS 건너뜀: {TTS_SERVER_URL}에 연결 실패 - {e}]")
        return None

    AUDIO_OUT_DIR.mkdir(exist_ok=True)
    out_path = AUDIO_OUT_DIR / f"turn_{turn:03d}.wav"

    player = None
    if play:
        try:
            player = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "-"],
                stdin=subprocess.PIPE,
            )
        except FileNotFoundError:
            print("  [ffplay 없음: 실시간 재생 건너뜀, 파일 저장만 진행 - brew install ffmpeg]")

    first_chunk_at = None
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
                print(f"  [TTS 첫 응답까지: {first_chunk_at - tts_start:.2f}초]")
            f.write(chunk)
            if player is not None:
                try:
                    player.stdin.write(chunk)
                except BrokenPipeError:
                    player = None

    if player is not None:
        player.stdin.close()
        player.wait()

    tts_elapsed = time.perf_counter() - tts_start
    print(f"  [TTS 총 소요 시간: {tts_elapsed:.2f}초]")

    return str(out_path)
