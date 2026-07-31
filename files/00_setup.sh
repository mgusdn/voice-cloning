#!/bin/bash
# GPT-SoVITS 설치 스크립트 (macOS / Apple Silicon 기준)
# 사용법: ./00_setup.sh

set -e

WORKDIR="$(pwd)"
REPO_DIR="$WORKDIR/GPT-SoVITS"

echo "[1/4] GPT-SoVITS 레포 클론 중..."
if [ -d "$REPO_DIR" ]; then
  echo "이미 존재함, 스킵: $REPO_DIR"
else
  git clone https://github.com/RVC-Boss/GPT-SoVITS.git "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "[2/4] Python 가상환경 생성 (venv)..."
python3 -m venv venv
source venv/bin/activate

echo "[3/4] 의존성 설치 중... (시간 좀 걸립니다)"
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] 사전학습 모델 다운로드 중..."
# 공식 repo 안내에 따른 pretrained 모델 다운로드 스크립트
bash tools/download_models.sh || echo "download_models.sh 없으면 README의 huggingface 링크에서 수동 다운로드 필요"

echo ""
echo "설치 완료! 다음 단계:"
echo "  source $REPO_DIR/venv/bin/activate"
echo "  cd $REPO_DIR"
echo "  python webui.py   # GUI로 데이터 전처리/학습 진행"
