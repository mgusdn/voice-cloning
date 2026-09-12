# voice-cloning

음원 수집 → 화자 데이터셋 → GPT-SoVITS 학습·API → 클로닝 음성으로 응답하는 상담 챗봇을 한 저장소에서 관리합니다. `testVoice`, `chatbot-voice`, `files`의 겹치는 코드를 정리한 통합본입니다.

## 빠른 시작: 텍스트 챗봇

Python 3.11 이상을 권장합니다. GPT-SoVITS는 별도 Python 3.10 환경을 사용합니다.

```bash
git clone https://github.com/mgusdn/voice-cloning.git
cd voice-cloning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# .env에 OPENAI_API_KEY를 설정하세요.
python -m chatbot --mode text
```

`OPENAI_MODEL`로 모델을 선택합니다. 기본값은 기존 프로젝트의 `gpt-5.4-mini`이며 해당 모델 사용 권한과 API 키가 필요합니다. 대화 시작 시 LLM API를 호출합니다. `--help`는 키·마이크·모델 없이 확인할 수 있습니다.

## 클로닝 목소리 연결

1. [학습·서버 안내](training/README.md)에 따라 GPT-SoVITS와 원하는 가중치를 준비하고 `api_v2.py`를 실행합니다.
2. 루트 `.env`에 `TTS_SERVER_URL`, `TTS_REF_AUDIO_PATH`, `TTS_REF_TEXT`를 설정합니다. 참조 오디오 경로는 **TTS 서버 컴퓨터 기준 절대경로**이고, 텍스트는 그 오디오의 실제 발화 내용입니다.
3. 루트의 챗봇 가상환경에서 실행합니다.

```bash
# 음성 출력: ffplay 필요 (macOS: brew install ffmpeg)
python -m chatbot --mode text --tts

# 마이크 입력까지 사용할 때만 추가 설치
python -m pip install -r chatbot/requirements-voice.txt
python -m chatbot --mode voice --tts

# 음성 파일만 생성 (같은 TTS 설정·클라이언트 사용)
python -m voice_clone.infer "안녕하세요." --output runtime/hello.wav
# 생성하며 재생: 위 명령에 --play 추가
```

음성 입력은 **pywhispercpp**, 데이터셋 전사는 **faster-whisper**를 사용합니다. macOS 마이크 권한 및 PortAudio 설정은 [챗봇 안내](chatbot/README.md)를 참고하세요. 챗봇 오디오는 `runtime/audio/<세션>/`에 저장됩니다. TTS 설정 누락이나 서버 오류가 발생해도 텍스트 대화는 계속됩니다.

## 새 화자 데이터 만들기

```bash
# 데이터 준비는 별도 가상환경 권장
python3 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements-data.txt
# ffmpeg/ffprobe 별도 설치. pyannote 모델 이용 조건 수락 후 토큰 설정.
export HF_TOKEN=토큰

python youtube.py "유튜브URL"
python diarize.py output.mp3
python build_voice_dataset.py segments/MERGED_SPEAKER_00.wav my_voice
```

결과는 `dataset/my_voice/`의 오디오, 전사문, GPT-SoVITS 목록입니다. 전사문을 검수한 후 [학습 단계](training/README.md)를 진행하세요. 이미 깨끗한 단일 화자 음원이 있다면 `training/01_slice_audio.py`와 `training/02_transcribe.py`로 분리·전사할 수 있습니다. 참조 음성 추출은 다음과 같습니다.

```bash
python make_reference.py source.wav runtime/ref.wav --start 10 --end 17
```

기존 전사문 240개는 `dataset/`에 그대로 보존했습니다. 원본 WAV와 모델 가중치는 저장소에 포함되지 않으므로 별도로 준비해야 합니다. 절대경로가 오래된 `.list`는 제거했으며 WAV를 복원한 뒤 다시 생성할 수 있습니다.

```bash
python tools/rebuild_lists.py dataset/droh/transcript.jsonl
```

[Colab 학습 안내](docs/colab.md) · [통합 내역과 이전 명령 대응표](docs/MIGRATION.md)

[검증 결과와 실행 범위](docs/VALIDATION.md)

## 구성

| 경로 | 역할 |
|---|---|
| `chatbot/` | 상담 그래프, 텍스트·마이크 입력 CLI |
| `voice_clone/` | 공통 HTTP TTS 클라이언트와 WAV 합성 CLI |
| `training/` | GPT-SoVITS 소스 준비, 오디오 슬라이싱·전사, 학습 안내 |
| `youtube.py`, `diarize.py`, `build_voice_dataset.py` | 음원 수집과 화자 데이터셋 생성 |
| `make_reference.py`, `tools/rebuild_lists.py` | 참조 오디오 추출, 학습 목록 복원 |
| `tools/build_dialogue_dataset.py` | 별도 LLM 파인튜닝용 대화 데이터셋 도구 |
| `dataset/` | 기존 전사문; 새 개인 데이터·오디오·가중치는 Git 제외 |

LLM용 대화 데이터셋 도구는 데이터용 환경에서 `python -m pip install -r tools/requirements-dialogue.txt`로 설치하고 `python tools/build_dialogue_dataset.py --help`를 참고하세요. 음성 학습 경로와 독립적인 선택 기능입니다.

## 검증

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m compileall -q chatbot voice_clone training tools
bash -n training/00_setup.sh
```

자동 테스트는 실제 모델 다운로드·학습·유료 API 호출 없이 실행합니다. 실제 음성 품질, 마이크 장치, GPT-SoVITS 가중치 로딩은 해당 환경과 데이터를 준비한 뒤 확인해야 합니다. 본인에게 사용 권한이 있는 음원·목소리만 사용하세요.
