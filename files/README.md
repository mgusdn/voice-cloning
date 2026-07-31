# GPT-SoVITS 보이스 클로닝 파이프라인

```
원본.wav → 01_slice_audio.py → 02_transcribe.py → webui.py 학습 → 추론
```

## 클론

```bash
git clone --recursive https://github.com/pume-org/tts.git
cd tts
# --recursive 빠뜨렸으면:
git submodule update --init
```

## 빠른 시작 (가중치 zip 있음)

`droh_weights.zip` 있으면 학습 스킵, 0 → 5 → 4 순서로.

## 0. 설치

```bash
chmod +x 00_setup.sh
./00_setup.sh
```

```bash
# 0-1. mecab-ko (requirements 설치 전에 먼저)
brew uninstall mecab   # 이미 깔려있으면
brew install mecab-ko-dic

# 0-2. pretrained 모델
cd GPT-SoVITS
curl -L -o pretrained_models.zip "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
unzip -q -o pretrained_models.zip -d GPT_SoVITS
rm pretrained_models.zip

# 0-3. torchcodec
source venv/bin/activate
pip install torchcodec
```

`requirements.txt`의 `fastapi[standard]==0.115.6` 버전 고정은 이미 반영됨 (건드리지 말 것).

## 1. 슬라이싱

```bash
source GPT-SoVITS/venv/bin/activate
python 01_slice_audio.py 원본음성.wav ./sliced_output
```

## 2. 전사

```bash
source GPT-SoVITS/venv/bin/activate
pip install faster-whisper   # 최초 1회
python 02_transcribe.py ./sliced_output {화자명} ./train.list
```

포맷: `/절대/경로/seg_0000.wav|{화자명}|ko|텍스트`

## 3. 학습

```bash
cd GPT-SoVITS
source venv/bin/activate
python webui.py   # http://127.0.0.1:9874
```

1. "1-GPT-SOVITS-TTS" → Experiment name = `{화자명}`, Version = `v2Pro`
2. "1A-Dataset Formatting Tool" → train.list 절대경로 입력, Audio dataset folder 비워둠 → "1Aabc" 클릭
3. "1B-Fine-Tuning" → "1Ba-SoVITS Training" → 끝나면 "1Bb-GPT Training"
4. 결과: `SoVITS_weights_v2Pro/`, `GPT_weights_v2Pro/`

소요시간 (M-series Mac, CPU, 데이터량 비례): SoVITS 에폭당 ~90초, GPT 에폭당 ~45초

## 4. 추론

**webui**: `python webui.py` → "1C-Inference" → weight 선택 → "Open TTS Inference WebUI" (`:9872`) → 참조 음성/텍스트 + 생성할 텍스트 입력

**스크립트** (`run_inference.py`):
```bash
cd GPT-SoVITS
source venv/bin/activate
python ../run_inference.py
```
`gpt_path`/`sovits_path`/`ref_wav_path`/`ref_text`/`target_text` 값만 수정.

**API 서버**:
```bash
python api_v2.py -a 127.0.0.1 -p 9880
curl "http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_weights_v2Pro/{name}-eXX.ckpt"
curl "http://127.0.0.1:9880/set_sovits_weights?weights_path=SoVITS_weights_v2Pro/{name}_eXX_sXXX.pth"
curl "http://127.0.0.1:9880/tts?text=텍스트&text_lang=ko&ref_audio_path=참조.wav&prompt_text=참조텍스트&prompt_lang=ko"
```

## 4-1. 추론 속도 최적화 (M-series Mac, API 서버 기준)

**스트리밍 (효과 큼, 권장)** — 첫 응답까지 5초대 → 0.4초대로 감소. 총 생성시간은 비슷하거나 소폭 증가하지만 체감 지연이 크게 줄어듦.
```bash
curl -G "http://127.0.0.1:9880/tts" \
  --data-urlencode "text=텍스트" --data-urlencode "text_lang=ko" \
  --data-urlencode "ref_audio_path=참조.wav" --data-urlencode "prompt_text=참조텍스트" \
  --data-urlencode "prompt_lang=ko" --data-urlencode "media_type=wav" \
  --data-urlencode "streaming_mode=3"   # 0=끔, 1=최고품질/최느림, 2=중간, 3=최저품질/최속
```
클라이언트는 `requests.get(..., stream=True)` + `iter_content()`로 청크 수신, 재생은 `afplay`(완성 파일만 가능) 대신 `ffplay -i -`(stdin 파이프)로 실시간 재생.

**batch_size (문장 조각 여러 개일 때만 효과)** — `text_split_method`(기본 `cut5`)가 쉼표까지 끊어서 문장 하나도 보통 여러 조각으로 나뉨. 조각 수만큼 `batch_size`를 주면 그만큼 병렬 처리돼 총 생성시간 단축 (테스트: 5조각 문장 기준 batch_size=1 대비 batch_size=5가 약 25% 빠름, 음질 차이 없음).
```bash
curl -G "http://127.0.0.1:9880/tts" ... --data-urlencode "batch_size=5"
```
⚠️ **`batch_size`와 `streaming_mode`를 동시에 켜면 서버가 죽는다** (`RuntimeError: Sizes of tensors must match... Expected size 1 but got size 3` — 스트리밍용 GPT 디코딩이 배치를 지원 안 함). 실시간 재생이 중요하면 스트리밍만, 총 시간이 중요하면(파일로 미리 생성해두는 경우 등) batch_size만 — 둘 중 하나만 선택.

**MPS(Mac GPU) — 시도했으나 역효과, 미적용.** `torch.backends.mps.is_available()`는 `True`지만, 실측 결과 CPU(5.4초)가 MPS(웜업 후 7.6~7.9초, 첫 호출은 17초)보다 빠름 — 이 자기회귀(autoregressive) 워크로드는 매 스텝이 작은 연산이라 MPS의 커널 디스패치 오버헤드가 CPU보다 크게 나타남. `tts_infer.yaml`/`inference_webui.py`는 `device: cpu`로 유지.

## 5. 가중치만 전달하기

zip에 4개 포함 (가중치만 있고 참조 음성/텍스트 없으면 추론 불가):

```bash
mkdir -p /tmp/pkg
cp GPT-SoVITS/GPT_weights_v2Pro/{name}-eXX.ckpt /tmp/pkg/
cp GPT-SoVITS/SoVITS_weights_v2Pro/{name}_eXX_sXXX.pth /tmp/pkg/
cp {참조음성.wav} /tmp/pkg/ref.wav
echo "{참조텍스트}" > /tmp/pkg/ref.txt
cd /tmp/pkg && zip -r ../{name}_weights.zip . && mv ../{name}_weights.zip /path/to/files/
```

현재 있는 예시: `droh_weights.zip` (268MB, git 미포함 — 100MB 제한, 별도 전달 필요)

받는 사람: 0-1~0-3 설치 → zip 풀어서 `*.ckpt`/`*.pth` 해당 폴더에 → 참조 wav/txt 경로 지정 → 4번 추론

## 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `RuntimeError: mecab-config not found` | mecab-ko 미설치 | `brew install mecab-ko-dic` |
| webui `TypeError: unhashable type: 'dict'` | fastapi/starlette 버전 불일치 | `fastapi[standard]==0.115.6` 유지 |
| SoVITS "Finished"인데 weights 폴더 비어있음 | torchcodec 없어서 데이터 0개 로드 | `pip install torchcodec` → "1Aabc" 재실행 → 재학습 |
| webui가 "완료"라고 뜨는데 실제 결과물 없음 | webui는 프로세스 종료만 확인, 성공 여부 미확인 | 터미널에서 직접 실행해서 에러 확인 |
| 파일 없음 에러 | 절대경로가 다른 컴퓨터 기준 | 현재 프로젝트 경로로 수정 |
| webui 새로고침 버튼 멈춤 | gradio 큐 이슈 | 페이지 새로고침(F5) 후 재시도 |
