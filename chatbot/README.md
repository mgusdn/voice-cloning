# 음성 상담 챗봇

저장소 루트에서 `python -m chatbot`으로 실행한다. 기본 입력은 텍스트이며, 마이크 입력과 클로닝 음성 출력을 각각 선택할 수 있다. 상담 상태와 질문 흐름은 기존 LangGraph 그래프를 사용한다.

## 설치와 설정

```bash
python3 -m venv .venv-chatbot
source .venv-chatbot/bin/activate
pip install -r chatbot/requirements.txt
cp .env.example .env
```

루트 `.env`에 `OPENAI_API_KEY`를 설정한다. `OPENAI_MODEL`로 모델을 바꿀 수 있고 기본값은 `gpt-5.4-mini`다. 이미 설정한 환경변수가 `.env`보다 우선한다. 이전 `chatbot/.env`, `chatbot-voice/.env` 설정은 루트 `.env`로 옮긴다.

```bash
python -m chatbot                          # 텍스트 입력/출력
python -m chatbot --mode text --debug      # 상담 상태와 처리 시간 표시
python -m chatbot --mode text --debug True # 기존 debug 값 표기도 지원
```

텍스트 입력에서 `quit`, `exit`, EOF 또는 Ctrl+C로 종료한다. 닉네임 입력 중 EOF/Ctrl+C도 정상 종료한다. `python -m chatbot --help`와 CLI 모듈 import는 모델이나 마이크 라이브러리를 불러오지 않는다.

## 마이크 입력

활성 STT 엔진은 **pywhispercpp (whisper.cpp)**이며 한국어 `small` 모델을 사용한다.

```bash
pip install -r chatbot/requirements-voice.txt
python -m chatbot.download_stt
python -m chatbot.stt --list-devices
python -m chatbot --mode voice
python -m chatbot --mode voice --device 3 --debug
```

`--device`에 입력 장치 인덱스 또는 이름 일부를 지정한다. 생략하면 `WH-1000XM5`, `iphone` 순으로 찾고 기본 입력 장치로 대체한다. 마이크 모드는 Ctrl+C로 종료한다. 모델 다운로드와 실제 마이크 사용은 해당 모드를 실행할 때만 필요하다. GPU 사용 여부는 whisper.cpp 빌드와 실행 환경에 따른다.

## 클로닝 목소리 출력

[학습 및 API 안내](../training/README.md)에 따라 GPT-SoVITS API 서버를 실행하고 루트 `.env`에 아래 값을 설정한다.

```dotenv
TTS_SERVER_URL=http://127.0.0.1:9880
TTS_REF_AUDIO_PATH=/절대/경로/참조음성.wav
TTS_REF_TEXT=참조 음성의 실제 발화 내용
```

`TTS_REF_AUDIO_PATH`는 **TTS 서버에서 읽을 수 있는 경로**여야 한다. 음성 재생에는 `ffplay`가 필요하다 (`brew install ffmpeg`).

```bash
python -m chatbot --tts
python -m chatbot --mode voice --tts --debug
```

공유 `voice_clone.tts_client.TTSClient`가 매 상담사 응답을 합성·저장·재생한다. 오디오 번호는 상담 상태의 턴 수와 별도로 증가하므로 초기 인사 단계에서도 파일을 덮어쓰지 않는다. TTS 설정이 없거나 서버 요청이 실패하면 텍스트 대화를 계속한다.
