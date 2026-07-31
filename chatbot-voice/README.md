# chatbot-voice

`chatbot`(juminsuh/chatbot, 순수 upstream 클론)이 `my_voice` 클로닝 목소리로 말하게 해주는 연동 코드. `chatbot/`은 **한 줄도 안 건드림** — `run_with_voice.py`가 `chatbot/graph.get_graph()`를 프록시로 감싸서 매 턴 `bot_message`를 가로채 TTS로 재생한 뒤 `cli.py`를 그대로 실행한다.

### 음성 출력 (my_voice TTS)
---

1. [pume-org/tts](https://github.com/pume-org/tts) 클론 (`--recursive`)
2. 별도 터미널에서 `files/GPT-SoVITS`에 들어가 venv 활성화 후 `python api_v2.py -a 127.0.0.1 -p 9880`
3. `curl "http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_weights_v2Pro/my_voice-e15.ckpt"`, `set_sovits_weights`도 동일하게
4. `chatbot/venv` 활성화한 상태로 이 폴더(`chatbot-voice/`)에서 `python cli.py` 대신 **`python run_with_voice.py`** 실행 → 상담사 메시지마다 자동으로 음성 재생됨 (TTS 서버 꺼져있으면 텍스트만 나옴)

실행하면 입력 방식을 고른다:
```
입력 방식을 선택하세요 [1] 텍스트  [2] 음성(마이크)
```
- `--mode text` (또는 `1`): 키보드로 타이핑 (마이크 없이 테스트용, 예전 cli.py 방식 그대로 재현)
- `--mode voice` (또는 `2`): `cli.py` 그대로 실행 — 실시간 마이크 STT (헤드셋 `WH-1000XM5` > `iphone` 순 자동 탐색, 안 잡히면 `--device` 옵션으로 지정)

`--debug True`는 voice 모드에만 그대로 전달됨 (`python run_with_voice.py --mode voice --debug True`).

이 폴더 자체엔 `requests`가 필요한데 `chatbot/requirements.txt`엔 없으니, `chatbot/venv`에 한 번 추가 설치: `pip install requests`

**음성 모드 쓰려면 추가로:**
- `chatbot/requirements.txt`에 STT 의존성(`faster-whisper`, `pywhispercpp`, `sounddevice`, `numpy`)이 추가됐으니 `chatbot/venv`에서 `pip install -r ../chatbot/requirements.txt` (처음 한 번, whisper 모델은 첫 실행 시 자동 다운로드)
- 스트리밍 재생에 `ffplay` 필요: `brew install ffmpeg` (없으면 파일 저장만 하고 재생은 건너뜀)

### 참조 음성/서버 주소 설정 (필수)

이 폴더에 `.env` 만들어서:
```
TTS_SERVER_URL="http://127.0.0.1:9880"
TTS_REF_AUDIO_PATH="/절대/경로/dataset/<화자이름>/wavs/<클립>.wav"
TTS_REF_TEXT="위 참조 클립의 실제 발화 내용"
```
`TTS_SERVER_URL`은 안 만들면 `http://127.0.0.1:9880`으로 기본 적용되지만, `TTS_REF_AUDIO_PATH`/`TTS_REF_TEXT`는 하드코딩된 기본값이 없으므로 반드시 설정해야 한다 — 없으면 TTS를 건너뛰고 텍스트만 나온다.

생성된 음성은 `audio_out/turn_XXX.wav`에 저장됨.
