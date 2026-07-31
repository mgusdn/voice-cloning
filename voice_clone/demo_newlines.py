"""
일반화 성능 데모: 학습 데이터에 전혀 없는 새 문장들을 클로닝 목소리로 합성.
모델 1회 로딩 후 여러 문장 합성 + 합본(demo_newlines_all.wav) 생성.

사용법:
    python voice_clone/demo_newlines.py
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS = os.path.join(BASE, "GPT-SoVITS")
os.chdir(GS)
sys.path.insert(0, GS)
sys.path.insert(0, os.path.join(GS, "GPT_SoVITS"))

import numpy as np
import soundfile as sf
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

DS = os.path.join(BASE, "dataset/lecturer_sp01")
TRAINED = os.path.join(DS, "trained")
pre = "GPT_SoVITS/pretrained_models"
OUTDIR = os.path.join(TRAINED, "newlines")
os.makedirs(OUTDIR, exist_ok=True)

REF = os.path.join(DS, "wavs/sp01_023_07.0s.wav")
REF_TEXT = "그런데 이 생활에 너무너무나 익숙해지다 보면 그것이 문제라는 것을 잊어버립니다."

# 학습 데이터에 없는 새 문장 (주제/문형/숫자/길이 다양화)
JOBS = [
    ("01_daily", "오늘 날씨가 참 맑네요. 점심으로 김치찌개를 먹을까 합니다."),
    ("02_question", "그래서 여러분은 과연 어떤 선택을 하시겠어요?"),
    ("03_numbers", "회의는 오후 세 시 삼십 분에 시작하고, 약 두 시간 정도 진행됩니다."),
    ("04_long", "처음에는 누구나 어렵습니다. 하지만 매일 조금씩 연습하다 보면, 어느새 익숙해진 자신을 발견하게 될 거예요."),
    ("05_command", "자, 이제 천천히 숨을 들이쉬고, 다시 길게 내쉬어 보세요."),
    ("06_mixed", "인공지능 기술은 정말 빠르게 발전하고 있습니다. 우리도 변화에 잘 적응해야겠죠."),
]

config = TTS_Config({"custom": {
    "device": os.environ.get("SOVITS_DEVICE", "cpu"), "is_half": False, "version": "v2",
    "t2s_weights_path": os.path.join(TRAINED, "lecturer-e15.ckpt"),
    "vits_weights_path": os.path.join(TRAINED, "lecturer_e10_s240.pth"),
    "bert_base_path": f"{pre}/chinese-roberta-wwm-ext-large",
    "cnhuhbert_base_path": f"{pre}/chinese-hubert-base",
}})
print("모델 로딩...")
tts = TTS(config)

clips, SR = [], None
for name, txt in JOBS:
    print(f"\n=== {name} ===\n{txt}")
    gen = tts.run({
        "text": txt, "text_lang": "ko",
        "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "ko",
        "top_k": 15, "top_p": 1, "temperature": 1,
        "text_split_method": "cut4", "speed_factor": 1.0, "fragment_interval": 0.3,
    })
    sr, audio = list(gen)[-1]
    SR = sr
    out = os.path.join(OUTDIR, f"{name}.wav")
    sf.write(out, audio, sr)
    clips.append(audio)
    print(f"  -> {out}  ({len(audio)/sr:.1f}s)")

gap = np.zeros(int(SR * 0.5), dtype=clips[0].dtype)
allaudio = np.concatenate([c for pair in zip(clips, [gap] * len(clips)) for c in pair])
sf.write(os.path.join(TRAINED, "demo_newlines_all.wav"), allaudio, SR)
print(f"\n합본 -> {os.path.join(TRAINED, 'demo_newlines_all.wav')}  ({len(allaudio)/SR:.1f}s)")
