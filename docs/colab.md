# Colab에서 GPT-SoVITS 학습하기

화자마다 복사하던 두 노트북의 데이터 업로드 → 환경 설치 → WebUI 학습 → 가중치 다운로드 절차를 공통 안내로 정리했다. 아래 `my_voice`를 실제 화자 폴더 이름으로 바꾼다. 기존 `dataset/`의 전사문 240개는 보존했으며, 원본 wav와 학습 가중치는 Git에 포함되어 있지 않다.

## 1. 로컬에서 데이터 준비

`dataset/my_voice/wavs/`에 전사문과 짝이 맞는 원본 wav를 복원하거나 `build_voice_dataset.py`로 새 데이터셋을 만든다. 전사문만으로 음성을 복원할 수는 없다.

```bash
# wav 존재를 검사하고 현재 컴퓨터의 절대경로 목록 생성
python tools/rebuild_lists.py dataset/my_voice/transcript.jsonl

# Colab에서 사용할 경로를 명시해 별도 목록 생성
python tools/rebuild_lists.py dataset/my_voice/transcript.jsonl --colab /content/my_voice/wavs

# my_voice/ 폴더를 루트로 포함하는 ZIP 생성
python -c "import shutil; shutil.make_archive('dataset/my_voice_colab', 'zip', 'dataset', 'my_voice')"
```

`rebuild_lists.py`는 `{file, text, ...}`와 기존 lecturer의 `{wav, text}` 형식을 모두 읽는다. 오래된 `wav` 절대경로에서는 파일 이름만 사용한다. 다른 곳에 오디오를 보관했다면 `--audio-dir /실제/wavs`로 지정하고 ZIP 안에는 `my_voice/wavs/`로 넣는다. `--speaker NAME`과 `--output FILE`로 화자 이름과 목록 저장 위치를 바꿀 수 있다. `--colab`에는 절대 대상 오디오 폴더가 필요하며, 이 모드만 로컬 오디오 존재 검사를 생략한다.

## 2. Colab 환경 설치

새 Colab 노트북에서 런타임 유형을 GPU로 선택하고 첫 셀에서 `!nvidia-smi`로 확인한다. [공식 Colab WebUI 노트북](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/Colab-WebUI.ipynb)을 이용할 수도 있다. 아래는 기존 프로젝트 노트북의 독립 Conda 환경 방식을 유지한 절차다.

```python
%pip install -q condacolab
import condacolab
condacolab.install()
```

런타임이 재시작되면 아래 셀부터 이어서 실행한다. 이 재시작 동작은 [condacolab 안내](https://github.com/conda-incubator/condacolab#usage)를 참고한다.

```bash
%%bash
set -e
cd /content
if [ ! -d GPT-SoVITS ]; then
  git clone https://github.com/RVC-Boss/GPT-SoVITS.git
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq GPTSoVITS; then
  conda create -n GPTSoVITS python=3.10 -y
fi
conda activate GPTSoVITS
cd GPT-SoVITS
bash install.sh --device CU126 --source HF
```

Python 3.10 환경과 설치 옵션은 [GPT-SoVITS의 Linux 설치 안내](https://github.com/RVC-Boss/GPT-SoVITS#linux)를 기준으로 한다. 설치 로그가 실패 없이 끝났는지 확인한다. 이 저장소 통합 검증에서는 Colab GPU 설치나 실제 학습을 실행하지 않았다.

## 3. 데이터 업로드

```python
from google.colab import files
from pathlib import Path
import zipfile

speaker = "my_voice"
uploaded = files.upload()  # my_voice_colab.zip 하나 선택
archive = next(iter(uploaded))
with zipfile.ZipFile(archive) as package:
    package.extractall("/content")
dataset = Path("/content") / speaker
labels = dataset / f"{speaker}.colab.list"
assert labels.is_file(), f"목록이 없습니다: {labels}"
for line in labels.read_text(encoding="utf-8").splitlines():
    audio_path = Path(line.split("|", 1)[0])
    assert audio_path.is_file(), f"오디오가 없습니다: {audio_path}"
print(f"준비 완료: {labels}")
```

ZIP에는 `my_voice/wavs/*.wav`와 `my_voice/my_voice.colab.list`가 들어 있어야 한다. 직접 준비한 데이터 ZIP을 사용한다.

## 4. WebUI 학습

```python
!cd /content/GPT-SoVITS && is_share=True conda run --no-capture-output -n GPTSoVITS python webui.py
```

실행 로그의 Gradio 공유 URL을 열어 다음 순서로 진행한다.

1. **Version**을 학습하려는 모델에 맞게 선택한다. 기존 두 노트북은 `v2`였다. `v2Pro`로 학습한다면 로컬 추론 서버에도 같은 버전의 가중치를 사용한다.
2. **1A Dataset Formatting**에서 텍스트 라벨 파일을 `/content/my_voice/my_voice.colab.list`, 실험 이름을 `my_voice`로 설정한다. 목록에 절대경로가 있으므로 Audio dataset folder는 비워둔다.
3. 일괄 포맷팅 **1Aabc**를 실행하고 텍스트·SSL·시맨틱 전처리 로그에 에러가 없는지 확인한다.
4. **1B Fine-Tuning**에서 SoVITS 학습 후 GPT 학습을 실행한다. 기존 T4/v2 설정은 GPT batch 4와 낮은 에폭 수에서 시작했으며, 데이터 양과 메모리 사용량에 맞게 조절한다.
5. 완료 표시뿐 아니라 로그의 가중치 저장 경로와 실제 파일을 확인한다. 필요한 출력 폴더 생성 실패가 나오면 해당 경로를 만든 뒤 다시 학습한다.

## 5. 가중치와 참조 음성 가져오기

WebUI 셀을 중지한 뒤 실행한다.

```python
from google.colab import files
from pathlib import Path

speaker = "my_voice"
root = Path("/content/GPT-SoVITS")
weights = sorted(root.glob(f"SoVITS_weights*/{speaker}*.pth"))
weights += sorted(root.glob(f"GPT_weights*/{speaker}*.ckpt"))
assert weights, "가중치가 없습니다. 실험 이름과 학습 로그의 저장 위치를 확인하세요."
for weight in weights:
    files.download(str(weight))
```

로컬 `training/GPT-SoVITS/`의 해당 버전 가중치 폴더에 파일을 넣고, API 서버에서 GPT `.ckpt`와 SoVITS `.pth`를 선택한다. 학습에 사용한 깨끗한 음성 한 구간과 정확한 전사문도 함께 준비한다.

```bash
python make_reference.py original.wav reference.wav --start 12.5 --end 19.5
```

생성된 24 kHz mono 참조 wav의 실제 발화 내용을 확인한 뒤 `TTS_REF_AUDIO_PATH`와 `TTS_REF_TEXT`에 설정한다. 참조 경로는 API 서버가 실행되는 컴퓨터에서 접근 가능해야 한다. 서버 설정은 [학습 안내](../training/README.md), 클라이언트 실행은 [루트 README](../README.md)를 따른다.
