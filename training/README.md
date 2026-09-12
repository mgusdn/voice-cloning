# GPT-SoVITS 학습과 API 준비

`음성 WAV → 슬라이싱 → 전사 검수 → GPT-SoVITS 학습 → HTTP TTS` 순서로 진행한다. 아래 명령은 별도 설명이 없으면 `voice-cloning` 저장소 루트에서 실행한다. 기존 `files/`의 학습 도구는 이 디렉터리로 통합했다.

이미 학습한 가중치가 있다면 환경과 공용 pretrained 모델을 준비한 뒤 **4. 추론**으로 이동한다. 화자별 학습을 다시 실행할 필요는 없다. 가중치 외에 참조 음성과 그 음성의 정확한 전사문도 필요하다.

## 0. 실행 환경

소스 기준은 upstream `48b1a0169a28582a8984402f82cf438d3bfa6aca`이다. 이는 재현 가능한 설치 시작점을 지정한 것이며, 이 통합 변경에서 실제 GPT-SoVITS 모델 학습이나 음질 검증까지 수행한 버전이라는 뜻은 아니다.

```bash
PYTHON=python3.10 bash training/00_setup.sh
```

스크립트는 실행한 디렉터리와 무관하게 `training/GPT-SoVITS/`에 지정 버전을 체크아웃하고 `venv/`를 만든다. Python 실행 파일은 `PYTHON`, 다른 upstream 커밋은 `GPT_SOVITS_REF`로 지정할 수 있다. 기존 체크아웃이 다른 버전이면 자동으로 변경하지 않고 종료한다. 소스와 가상환경 준비가 끝나도 **의존성과 pretrained 모델은 아직 설치되지 않은 상태**다.

### macOS 가상환경에서 수동 설치

```bash
brew install ffmpeg mecab-ko-dic
cd training/GPT-SoVITS
source venv/bin/activate
python -m pip install -r extra-req.txt --no-deps
python -m pip install -r requirements.txt
python -m pip install torchcodec faster-whisper
```

수동 설치 순서는 [고정한 upstream 설치 안내](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/README.md#install-manually)를 따른다. CUDA/Linux 환경은 해당 안내에서 장치에 맞는 PyTorch 설치 방법도 확인한다. 일반 `mecab`과 `mecab-ko`가 충돌하면 설치된 패키지를 먼저 확인한다. 스크립트가 기존 시스템 패키지를 삭제하거나 변경하지는 않는다.

수동 경로에서는 [공용 pretrained 모델](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)을 받아 `training/GPT-SoVITS/GPT_SoVITS/pretrained_models/` 아래에 upstream과 같은 디렉터리 구조로 배치해야 한다. 사용할 버전의 모델도 포함해야 한다. 예를 들어 v2Pro 계열은 `v2Pro/`와 `sv/`의 관련 모델이 필요하다. 추가 언어 모델과 선택 기능별 파일 위치는 [upstream pretrained 안내](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/README.md#pretrained-models)를 확인한다.

### Conda를 사용하는 경우

upstream의 자동 설치기를 선택할 수 있다. [해당 설치기](https://github.com/RVC-Boss/GPT-SoVITS/blob/48b1a0169a28582a8984402f82cf438d3bfa6aca/install.sh)는 Conda를 요구하며 의존성 설치와 모델 다운로드를 수행한다. 위 `venv`를 활성화한 채 두 환경을 섞지 않는다.

```bash
# venv가 활성화되어 있으면 먼저 deactivate
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
cd training/GPT-SoVITS
bash install.sh --device CPU --source HF
```

이 경로를 선택했다면 이후 `source .../venv/bin/activate` 대신 `conda activate GPTSoVits`를 사용한다. 설치 로그가 성공했는지와 필요한 모델 파일이 존재하는지 확인한 뒤 다음 단계로 진행한다.

## 1. 슬라이싱

```bash
source training/GPT-SoVITS/venv/bin/activate
python training/01_slice_audio.py /absolute/path/source.wav runtime/sliced/my_voice
```

출력은 32kHz 모노 WAV이며 `seg_0000.wav`부터 순서대로 저장한다. 무음을 기준으로 구간을 찾고, 긴 발화는 샘플을 빠뜨리지 않고 균등 분할해 각 클립이 10초를 넘지 않게 한다. 가까운 짧은 구간은 합친다. 3초 미만의 발화가 고립되어 10초 제한 안에서 합칠 수 없으면 삭제하지 않고 경고와 함께 보존한다. 학습 전에 해당 클립을 듣고 사용할지 결정한다. 3–10초는 목표 범위이며 이 짧은 클립은 명시적인 예외다.

빈 음성이나 완전 무음이면 WAV를 만들지 않고 CLI가 실패 코드로 종료한다. 이미 `seg_*.wav`가 있는 폴더에는 덮어쓰지 않으므로 새 입력마다 새 출력 폴더를 사용한다.

## 2. 전사와 검수

```bash
python training/02_transcribe.py runtime/sliced/my_voice my_voice runtime/lists/my_voice.list
# 모델과 언어를 바꿀 때:
python training/02_transcribe.py runtime/sliced/my_voice my_voice runtime/lists/my_voice.list \
  --language ko --model medium --device cpu --compute-type int8
```

기본값은 한국어, `large-v3`, CPU/int8이다. faster-whisper 모델이 로컬 캐시에 없으면 처음 실행할 때 다운로드한다. 출력 디렉터리는 자동 생성하고 WAV 경로는 절대경로로 기록한다.

```text
/absolute/path/seg_0000.wav|my_voice|ko|검수한 발화 내용
```

빈 전사는 건너뛴다. 전부 비어 있으면 기존 `.list`를 덮어쓰지 않고 오류를 반환한다. 화자명에는 `|`나 줄바꿈을 사용할 수 없다. 전사문 내부의 `|`와 줄바꿈은 공백으로 정리해 한 음성이 한 레코드로 남게 한다. ASR 결과와 짧은 클립은 학습 전에 직접 검수한다.

## 3. 학습

```bash
cd training/GPT-SoVITS
source venv/bin/activate
python webui.py
```

터미널에 표시된 WebUI 주소를 연다. 일반적인 학습 순서는 다음과 같다.

1. `1-GPT-SOVITS-TTS`에서 실험 이름과 사용할 모델 버전을 선택한다.
2. 데이터 포맷 도구에 `.list`의 절대경로를 입력하고 전처리를 실행한다. 목록에 절대경로가 있으므로 오디오 루트 입력란은 비워둔다.
3. SoVITS 학습을 실행한 다음 GPT 학습을 실행한다.
4. 터미널 로그와 실제 체크포인트 생성을 확인한다. v2Pro를 선택했다면 보통 `SoVITS_weights_v2Pro/`, `GPT_weights_v2Pro/`에 저장된다.

WebUI의 종료 메시지만으로 성공을 판단하지 않는다. 전처리와 학습의 에러 로그 및 결과 파일을 확인한다. 학습 시간과 CPU/GPU 성능은 장치, 버전, 데이터 길이에 따라 달라진다.

## 4. 추론

WebUI에서는 `1C-Inference`에서 가중치를 선택하고 추론 UI를 연다. 애플리케이션과 파일 합성 CLI는 같은 GPT-SoVITS HTTP API를 사용한다.

```bash
cd training/GPT-SoVITS
source venv/bin/activate
python api_v2.py -a 127.0.0.1 -p 9880
```

서버가 실행된 상태에서 다른 터미널로 학습 가중치를 선택한다. 경로는 **서버의 작업 디렉터리** 기준이며, 파일명은 실제 학습 결과로 바꾼다.

```bash
curl --fail-with-body -G http://127.0.0.1:9880/set_gpt_weights \
  --data-urlencode 'weights_path=GPT_weights_v2Pro/my_voice-e15.ckpt'
curl --fail-with-body -G http://127.0.0.1:9880/set_sovits_weights \
  --data-urlencode 'weights_path=SoVITS_weights_v2Pro/my_voice_e8_s352.pth'
```

저장소 루트의 `.env`에 `TTS_SERVER_URL`, `TTS_REF_AUDIO_PATH`, `TTS_REF_TEXT`를 설정한다. 참조 경로는 API 서버에서 읽을 수 있는 절대경로여야 한다. 서버와 클라이언트가 다른 컴퓨터라면 서버 쪽 경로를 사용한다. 이후 루트 애플리케이션 환경에서 실행한다.

```bash
python -m voice_clone.infer '안녕하세요. 오늘 하루도 고생 많으셨어요.' --output runtime/sample.wav
python -m voice_clone.infer '재생도 함께 합니다.' --output runtime/sample.wav --play
```

스트리밍은 첫 음성까지의 지연을 줄이는 데 사용할 수 있다. 배치 크기는 여러 텍스트 조각의 총 생성 시간에 영향을 준다. 기존 실험에서는 스트리밍과 큰 배치 크기를 함께 사용했을 때 서버 오류가 있었으므로, 먼저 배치 1로 동작을 확인하고 버전별로 측정한다. 과거 특정 Mac에서 측정한 시간이나 CPU/MPS 우열을 다른 장치의 보장값으로 사용하지 않는다.

## 5. 다른 환경으로 전달하기

화자별 전달 묶음에는 GPT `.ckpt`, SoVITS `.pth`, 참조 WAV, 참조 전사 TXT를 포함한다. 실험명, 모델 버전, upstream 커밋도 함께 기록한다. 공용 pretrained 모델과 실행 의존성은 받는 환경에도 필요하다. 대용량 가중치와 실제 음성은 Git에 추가하지 않고 별도로 보관한다.

이전 `files/train_droh.list`의 62개 전사 레코드는 기존 `dataset/droh` 전사 자료에도 모두 포함되어 있어 중복 절대경로 목록을 제거했다. 데이터셋의 현재 목록 재생성 방법은 루트 데이터셋 안내를 따른다.

## 문제 확인

| 증상 | 확인할 내용 |
|---|---|
| `mecab-config not found` | 한국어 mecab 설치와 실행 환경의 PATH |
| `torchcodec` 오디오 로딩 오류 | 현재 PyTorch와 호환되는 torchcodec 설치, 전처리 재실행 |
| 학습 종료 후 가중치 없음 | 터미널의 첫 에러, 전처리 산출물, 실제 로드된 데이터 수 |
| WebUI의 Gradio/FastAPI 오류 | 선택한 upstream의 요구사항과 설치된 버전 비교; 개별 패키지 무조건 업그레이드 금지 |
| 참조 음성을 찾지 못함 | API 서버에서 접근 가능한 경로인지 확인 |
| 새 작업에 이전 음성 조각이 섞임 | 입력마다 새 슬라이싱 폴더 사용 |
