# voice-cloning

유튜브 영상에서 특정 화자의 음성·텍스트를 추출해 **보이스 클로닝(GPT-SoVITS)** 모델을 만들고, 그 목소리로 말하는 **음성 상담 챗봇**까지 이어지는 통합 프로젝트.

기존 3개 저장소(`testVoice`, `test_ohdoc`, `tts`)를 하나로 합친 것으로, 전체 흐름은 다음과 같다:

```
[소스 수집]              [데이터셋 구축]                      [보이스 클로닝]           [응용]
youtube.py   →  diarize.py → build_voice_dataset.py → GPT-SoVITS 파인튜닝 → chatbot + chatbot-voice
(오디오 추출)     (화자분리)     (슬라이싱 + 대사)        (files/)               (클로닝 목소리로 응답)
```

## 폴더/파일 구성

| 경로 | 출처 | 역할 |
|---|---|---|
| `youtube.py` | tts | 유튜브 오디오 mp3 추출 + 불필요 구간 제외 (`yt-dlp` + `ffmpeg`) |
| `diarize.py`, `build_voice_dataset.py`, `dataset/`, `segments/`, `voice_clone/` | testVoice | 화자분리 → 타겟 화자 데이터셋 구축 → 클로닝 추론 |
| `build_dataset.py` | testVoice | (별개 목적) LLM 파인튜닝용 대화 데이터셋 `train.jsonl` 생성 |
| `files/` | test_ohdoc | GPT-SoVITS 본체 + 슬라이싱/전사/학습/추론 파이프라인 (`droh_weights/`는 예시로 남겨둔 참고용 가중치, 실사용과 무관) — 상세는 `files/README.md` |
| `chatbot/` | test_ohdoc | LangGraph 기반 음성 상담 챗봇 (STT: faster-whisper, LLM: OpenAI) — 상세는 `chatbot/README.md` |
| `chatbot-voice/` | test_ohdoc | 챗봇 응답을 클로닝 목소리(TTS 서버)로 재생하는 연동 레이어 — 상세는 `chatbot-voice/README.md` |
| `*.wav`, `*.mp3` | 각 폴더 | 작업 중이던 원본/중간 산출물 오디오 |

## 파이프라인 단계별 사용법

### 1. 가중치 만들기 (Finetuning)

```bash
# 1. 유튜브에서 원하는 부분 추출
python youtube.py "유튜브URL"

# 2. 화자 분리 + 슬라이싱 → voice cloning에 쓸 화자 결정
python diarize.py <입력.wav>

# 3. 대사 작성
python build_voice_dataset.py segments/MERGED_SPEAKER_XX.wav <화자이름>

# 4. SoVITS + GPT 학습 (webui)
cd files/GPT-SoVITS && source venv/bin/activate
python webui.py   # http://127.0.0.1:9874
```

webui(`http://127.0.0.1:9874`)에서:

1. "1-GPT-SOVITS-TTS" → Experiment name에 `<화자이름>` 입력, Version `v2Pro`
2. "1A-Dataset Formatting Tool" → train.list 경로에 `.../dataset/<화자이름>/<화자이름>.list` 입력, Audio dataset folder는 비워둠 → "1Aabc" 실행
3. "1B-Fine-Tuning" → SoVITS 학습 → 끝나면 GPT 학습

결과 가중치: `files/GPT-SoVITS/SoVITS_weights_v2Pro/`, `files/GPT-SoVITS/GPT_weights_v2Pro/`

### 2. 추론 (API 서버)

```bash
# chatbot-voice가 사용하는 TTS 서버 — 상세는 files/README.md
cd files/GPT-SoVITS && source venv/bin/activate
python api_v2.py -a 127.0.0.1 -p 9880
```

### 3. 음성 상담 챗봇

```bash
# 텍스트/음성 챗봇 자체 실행 (설치·마이크 설정은 chatbot/README.md)
cd chatbot && python cli.py

# 클로닝 목소리로 응답 재생 (3번 API 서버 켠 상태에서, chatbot/venv 활성화 후)
cd chatbot-voice && python run_with_voice.py   # 입력: [1] 텍스트 / [2] 음성(마이크)
```

## 환경 설정

⚠️ 이 폴더는 원본에서 **venv를 제외하고 복사**한 것이라 가상환경은 직접 다시 만들어야 한다. 컴포넌트별로 의존성이 달라 venv를 분리해서 쓴다:

| 용도 | 위치 | 설치 |
|---|---|---|
| 화자분리·데이터셋 (2번) | 루트 `.venv` | `python3 -m venv .venv && pip install -r requirements.txt` + `export HF_TOKEN=...` (pyannote용) |
| 제로샷 클로닝 (`tts_clone.py`) | 별도 venv | `requirements-tts.txt` (Python <3.12) |
| GPT-SoVITS (3번) | `files/GPT-SoVITS/venv` | `files/00_setup.sh` + `files/README.md`의 0-1~0-3 |
| 챗봇 (4번) | `chatbot/venv` | `chatbot/requirements.txt` + `.env`에 `OPENAI_API_KEY` |

공통: `brew install ffmpeg`

주의: `.list` 파일 중 일부(`dataset/droh/droh.list`, `files/train_droh.list` 등 예시 데이터)에는 **절대경로**가 들어있다 (현재 `voice-cloning` 경로 기준). 새로 만드는 `<화자이름>.list`는 `build_voice_dataset.py`가 현재 경로 기준으로 자동 생성하므로 문제없다. 폴더를 옮기거나 이름을 바꾸면 기존 `.list` 파일들은 다시 경로 수정 필요.

`chatbot-voice`에서 클로닝 목소리를 쓰려면 `chatbot-voice/.env`에 `TTS_REF_AUDIO_PATH`, `TTS_REF_TEXT`를 반드시 설정해야 한다 (하드코딩된 기본값 없음) — `chatbot-voice/README.md` 참고.

## 기타

- 대용량 파일(GPT-SoVITS 본체, 가중치 `*.pth/*.ckpt`, 오디오)은 git에 올리지 말 것 (`.gitignore` 참고)
- 음성권·초상권 유의 — 연구·실험 목적의 비공개 사용 전제
