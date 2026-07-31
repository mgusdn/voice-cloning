"""
학습된 GPT-SoVITS v2 가중치로 클로닝 음성 합성 (단일 문장).

레포 루트의 GPT-SoVITS/ 와 dataset/lecturer_sp01/trained/ 의 학습 가중치를 사용한다.
가중치(*.pth/*.ckpt)와 wav 는 .gitignore 로 빠지므로, 학습/다운로드 후에만 동작한다.

사용법:
    python voice_clone/infer.py "합성할 문장" [출력.wav]
"""
import os
import sys

# 레포 루트 = 이 파일의 두 단계 상위
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(BASE, "GPT-SoVITS")
os.chdir(GS)
sys.path.insert(0, GS)
sys.path.insert(0, os.path.join(GS, "GPT_SoVITS"))

import soundfile as sf
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

TRAINED = os.path.join(BASE, "dataset/lecturer_sp01/trained")
DS = os.path.join(BASE, "dataset/lecturer_sp01")
pre = "GPT_SoVITS/pretrained_models"

# 레퍼런스: 학습 데이터 중 깨끗한 7초 클립 + 정확한 전사
REF = os.environ.get("REF_AUDIO", os.path.join(DS, "wavs/sp01_023_07.0s.wav"))
REF_TEXT = os.environ.get(
    "REF_TEXT",
    "그런데 이 생활에 너무너무나 익숙해지다 보면 그것이 문제라는 것을 잊어버립니다.",
)
TEXT = sys.argv[1] if len(sys.argv) > 1 else "안녕하세요. 학습된 목소리로 합성한 음성입니다."
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(TRAINED, "infer_out.wav")

config = TTS_Config({"custom": {
    "device": os.environ.get("SOVITS_DEVICE", "cpu"),
    "is_half": False,
    "version": "v2",
    "t2s_weights_path": os.path.join(TRAINED, "lecturer-e15.ckpt"),       # 학습된 GPT
    "vits_weights_path": os.path.join(TRAINED, "lecturer_e10_s240.pth"),  # 학습된 SoVITS
    "bert_base_path": f"{pre}/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": f"{pre}/chinese-hubert-base",
}})

print("[1/2] 학습 가중치 로딩...")
tts = TTS(config)
print(f"[2/2] 합성: {TEXT}")
gen = tts.run({
    "text": TEXT, "text_lang": "ko",
    "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "ko",
    "top_k": 15, "top_p": 1, "temperature": 1,
    "text_split_method": "cut4", "speed_factor": 1.0, "fragment_interval": 0.3,
})
sr, audio = list(gen)[-1]
sf.write(OUT, audio, sr)
print(f"완료 -> {OUT}  ({len(audio)/sr:.1f}s, {sr}Hz)")
