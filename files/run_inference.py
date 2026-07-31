# 아래 값들을 본인이 학습한 화자명/경로로 수정 후 실행
import os
import sys

os.environ["gpt_path"] = "GPT_weights_v2Pro/{name}-eXX.ckpt"
os.environ["sovits_path"] = "SoVITS_weights_v2Pro/{name}_eXX_sXXX.pth"
os.environ["is_half"] = "False"

sys.path.insert(0, ".")
sys.path.insert(0, "GPT_SoVITS")

import soundfile as sf
from GPT_SoVITS.inference_webui import get_tts_wav, i18n

ref_wav_path = "{name}/wavs/{name}_001_x.xs.wav"
ref_text = "참조 음성의 실제 발화 내용"
target_text = "합성하고 싶은 문장을 여기에 입력"

result = get_tts_wav(
    ref_wav_path=ref_wav_path,
    prompt_text=ref_text,
    prompt_language=i18n("韩文"),
    text=target_text,
    text_language=i18n("韩文"),
    top_p=1,
    temperature=1,
)

result_list = list(result)
if result_list:
    sr, audio = result_list[-1]
    out_path = "output.wav"
    sf.write(out_path, audio, sr)
    print(f"Saved: {out_path}")
else:
    print("No audio produced")
