# 네 저장소 통합 내역

통합 기준 저장소는 [mgusdn/voice-cloning](https://github.com/mgusdn/voice-cloning)입니다. 다른 세 저장소의 실행 코드는 대부분 이미 복사되어 있어, 같은 코드를 다시 들여오는 대신 중복 실행 경로와 설정을 정리했습니다. 원본 저장소와 Git 이력은 삭제하지 않았습니다.

## 조사한 원본

| 저장소 | 확인한 커밋 | 통합 판단 |
|---|---|---|
| voice-cloning | `513c953c0c0b7932f46bc5c71405cf987a7a0fd8` | 가장 넓은 기능 범위를 가진 기준 저장소 |
| testVoice | `10cf6098d3a97a674ca75484521a99922566bd88` | 공통 파일 13개 동일; 빌더와 목록 차이는 기본값·로컬 경로 중심 |
| chatbot-voice | `96b7fef0652d72ff8011e394a27836fa1923d6f6` | 실행 어댑터 동일; 통합본 TTS 설정이 개인 하드코딩을 제거한 버전 |
| files | `ec566e21cb30b15b5ad883873e05a8b03da7bed1` | 설치·슬라이싱·전사 코드 동일; 통합본 안내가 더 최신 |

## 이전 경로 → 통합 경로

| 이전 | 현재 |
|---|---|
| `chatbot-voice/run_with_voice.py --mode text` | `python -m chatbot --mode text --tts` |
| `chatbot-voice/run_with_voice.py --mode voice` | `python -m chatbot --mode voice --tts` |
| `chatbot/cli.py` | `python -m chatbot --mode voice` |
| `chatbot/.env`, `chatbot-voice/.env` | 루트 `.env` 한 곳; 기존 환경변수 우선 |
| `files/00_setup.sh` | `bash training/00_setup.sh` |
| `files/01_slice_audio.py`, `files/02_transcribe.py` | `training/`의 같은 파일명 |
| `files/GPT-SoVITS/` | `training/GPT-SoVITS/` (기존 설치는 직접 옮겨 재사용 가능) |
| `files/run_inference.py`, 개인 화자용 `voice_clone/infer.py` | `python -m voice_clone.infer TEXT --output FILE` |
| 개인 화자용 `voice_clone/demo_newlines.py` | 공통 합성 CLI를 필요한 문장마다 호출 |
| `build_dataset.py` | `tools/build_dialogue_dataset.py` (선택 기능) |
| 타임스탬프 고정 `make_reference.py` | `python make_reference.py INPUT OUTPUT --start SEC --end SEC` |
| `.list`, `.colab.list` | `tools/rebuild_lists.py`로 현재 경로에 맞게 생성 |
| 화자별 중복 Colab 노트북 | [공통 Colab 안내](colab.md) |

기존 GPT-SoVITS 설치를 옮길 때 가중치·참조 오디오·수정한 설정도 함께 보존하세요. `.env`의 참조 경로와 서버 설정 안의 파일 경로를 새 위치로 맞춥니다. 새 설치 스크립트는 기존 작업 트리를 강제로 덮어쓰지 않습니다.

## 보존·제거 근거

- `dataset/droh` 64개, `lecturer_sp01` 81개, `임커밋` 95개 전사문을 원문 그대로 보존했습니다. 오디오가 없는 상태에서 이를 재생성할 수 없으므로 제거하지 않았습니다.
- 예전 절대경로 학습 목록, ffmpeg concat 임시 목록, 출력 없는 화자별 중복 노트북은 재생성 가능한 중복입니다. `files/train_droh.list`의 전사 내용도 보존된 droh 데이터와 겹칩니다.
- 별도 `chatbot-voice`의 경로 주입·그래프 monkeypatch를 제거하고 상담 CLI가 공통 TTS 클라이언트를 직접 사용합니다.
- XTTS는 GPT-SoVITS와 다른 엔진이며 기존 통합본에는 구현이 없고 의존성 파일만 남아 있었습니다. 별도 Python 환경과 고정 참조 음성을 요구하던 `testVoice/tts_clone.py`는 들여오지 않고 불필요한 `requirements-tts.txt`를 제거했습니다. 원본 커밋에서 복원할 수 있습니다.
- GPT-SoVITS 본체, 녹음, 모델 가중치, 가상환경은 외부 자산으로 유지하며 Git에 추가하지 않습니다.

## 반영한 실행 오류 수정

중간 전송 실패 때 TTS가 대화를 종료시키는 문제, 스트리밍 WAV의 길이 헤더, 출력 파일 충돌, 학습 슬라이스의 최대 길이, 상대경로 전사 목록, 예전 작업 파일이 섞이는 문제, LLM 데이터 분할 시 샘플이 빠지는 문제를 테스트로 다룹니다.

GPT-SoVITS 설치 위치와 버전은 [training/README.md](../training/README.md)에 모았습니다. HTTP 요청 형식과 스트리밍 모드는 [공식 api_v2.py](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/api_v2.py)를 기준으로 합니다.
