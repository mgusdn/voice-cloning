"""
Qwen3 파인튜닝용 대화 데이터셋 빌더 (v1: Q&A 쌍만).

raw_media/ 안의 여러 영상/오디오를 일괄 처리하여:
  1) 화자분리(pyannote)로 "누가 언제" 말했는지
  2) 자막(faster-whisper)으로 "무슨 말"을 했는지
  3) 알려진 박사님 음성(REF_VOICE)과 성문 비교로 "어느 화자가 박사님"인지 식별
  4) 시간순으로 [상대방 발화] -> [박사님 발화] 쌍을 만들어
  5) mlx-lm 호환 챗 JSONL (train/valid)로 저장

독백(박사님 혼자 길게)은 v1에서 건너뜀 -> 나중에 질문 역생성으로 보강.

사용법 (.venv 활성화 후):
    export HF_TOKEN=hf_xxxxx
    python tools/build_dialogue_dataset.py --reference <참조음성.wav>
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict


# ---------------- 설정 ----------------
BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "raw_media"           # 원본 영상/오디오를 여기 넣으세요
OUT_DIR = BASE / "dataset"
REF_VOICE = BASE / "cand3.wav"          # 알려진 박사님 음성(성문 기준)
WORK = BASE / "_ds_work"                # 변환 중간파일

SR = 16000
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
SYSTEM_PROMPT = "너는 오은영 박사처럼 따뜻하고 차분하게, 공감을 먼저 표현하고 핵심을 짚어 상담하는 전문가야."

# 박사님 화자 식별: 절대 바닥값 + 2등과의 격차(margin)로 판단.
# BGM 잔존 등으로 절대 유사도는 낮아질 수 있어, 화자 간 상대 격차를 함께 본다.
SPK_SIM_FLOOR = float(os.environ.get("SPK_SIM_FLOOR", "0.60"))    # 최소 절대 유사도
SPK_SIM_MARGIN = float(os.environ.get("SPK_SIM_MARGIN", "0.04"))  # 2등과 벌어져야 하는 격차

# 필터
MIN_ASSIST_CHARS = 15   # 박사님 답변 최소 길이(맞장구 제거)
MIN_USER_CHARS = 4      # 상대방 질문 최소 길이
MAX_CHARS = 600         # 너무 긴 턴 컷
BACKCHANNEL = {"네", "예", "음", "그쵸", "그죠", "맞아요", "아", "어", "응", "네네", "그래요"}

MEDIA_EXT = {".mp3", ".mp4", ".wav", ".m4a", ".mov", ".mkv", ".webm", ".aac", ".flac"}
VALID_RATIO = 0.05      # 검증셋 비율
# --------------------------------------

def to_wav(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)],
        check=True, capture_output=True,
    )


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def is_backchannel(text: str) -> bool:
    t = text.strip().strip(".?!~ ")
    return t in BACKCHANNEL or len(t) < 2


def cosine(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def process_file(path: Path, pipeline, whisper, encoder, ref_embed):
    """한 파일 -> [(role, text), ...] 시간순 턴 리스트(병합 완료) 반환. 박사님 없으면 None."""
    import numpy as np
    import soundfile as sf

    WORK.mkdir(parents=True, exist_ok=True)
    wav_path = WORK / (path.stem + "_16k.wav")
    to_wav(path, wav_path)

    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 1) 화자분리
    out = pipeline(str(wav_path))
    diar = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
    spk_turns = [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]
    if not spk_turns:
        return None

    # 2) 박사님 화자 식별: 화자별 오디오를 모아 임베딩 -> ref와 코사인
    spk_audio = defaultdict(list)
    for s, e, spk in spk_turns:
        spk_audio[spk].append(audio[int(s * sr):int(e * sr)])
    spk_sim = {}
    for spk, chunks in spk_audio.items():
        merged = np.concatenate(chunks)
        if len(merged) < sr * 1.0:   # 1초 미만이면 신뢰 불가
            spk_sim[spk] = -1.0
            continue
        emb = encoder.embed_utterance(merged.astype(np.float32))
        spk_sim[spk] = cosine(ref_embed, emb)
    ranked = sorted(spk_sim.items(), key=lambda x: -x[1])
    doctor, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else -1.0
    print(f"    화자 유사도: " + ", ".join(f"{k}={v:.2f}" for k, v in ranked))
    # 절대 바닥값 미달이면 스킵. 단, 2등과 충분히 벌어져 있으면 바닥값 약간 아래도 허용.
    margin_ok = (top - second) >= SPK_SIM_MARGIN
    if top < SPK_SIM_FLOOR and not (margin_ok and top >= SPK_SIM_FLOOR - 0.05):
        print(f"    -> 박사님 식별 실패(최고 {top:.2f}, 격차 {top-second:.2f}). 이 파일 스킵.")
        return None
    if not margin_ok:
        print(f"    -> 주의: 1·2등 격차 작음({top-second:.2f}). 식별 신뢰도 낮음.")
    print(f"    -> 박사님 = {doctor} (유사도 {top:.2f}, 격차 {top-second:.2f})")

    # 3) 자막
    segments, _ = whisper.transcribe(str(wav_path), language="ko", vad_filter=True)
    asr = [(seg.start, seg.end, seg.text.strip()) for seg in segments if seg.text.strip()]

    # 4) 각 자막 segment를 겹침이 가장 큰 화자에 할당
    labeled = []
    for s, e, text in asr:
        best, best_ov = None, 0.0
        for ts, te, spk in spk_turns:
            ov = overlap(s, e, ts, te)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None:
            continue
        role = "assistant" if best == doctor else "user"
        labeled.append((s, role, text))

    # 5) 연속 동일 역할 병합
    turns = []
    for s, role, text in labeled:
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + " " + text)
        else:
            turns.append((role, text))
    return turns


def turns_to_pairs(turns):
    """[user -> assistant] 인접 쌍만 추출."""
    pairs = []
    for i in range(1, len(turns)):
        prev_role, prev_text = turns[i - 1]
        role, text = turns[i]
        if prev_role == "user" and role == "assistant":
            u = prev_text.strip()
            a = text.strip()
            if is_backchannel(a) or len(a) < MIN_ASSIST_CHARS or len(u) < MIN_USER_CHARS:
                continue
            pairs.append((u[:MAX_CHARS], a[:MAX_CHARS]))
    return pairs


def split_pairs(pairs):
    """Keep every unique sample in exactly one deterministic train/valid partition."""
    ordered = sorted(pairs, key=lambda pair: (len(pair[1]), pair[0]))
    n_valid = max(1, int(len(ordered) * VALID_RATIO)) if len(ordered) > 20 else 0
    if n_valid:
        step = max(1, len(ordered) // n_valid)
        valid_idx = set(list(range(0, len(ordered), step))[:n_valid])
    else:
        valid_idx = set()
    valid = [ordered[i] for i in sorted(valid_idx)]
    train = [pair for i, pair in enumerate(ordered) if i not in valid_idx]
    return train, valid


def prepare_reference(source, workdir):
    """Resample references to the same 16 kHz used for candidate embeddings."""
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "reference_16k.wav"
    to_wav(source, target)
    return target


def main(argv=None):
    global RAW_DIR, OUT_DIR, REF_VOICE, WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--reference", type=Path, required=True, help="화자 식별용 참조 음성")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    RAW_DIR, OUT_DIR, REF_VOICE = args.raw_dir, args.output_dir, args.reference
    WORK = BASE / "_ds_work" / "dialogue"
    if not RAW_DIR.is_dir():
        parser.error(f"원본 미디어 폴더가 없습니다: {RAW_DIR}")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        parser.error("환경변수 HF_TOKEN을 설정하세요.")
    if not REF_VOICE.is_file():
        sys.exit(f"박사님 기준 음성이 없습니다: {REF_VOICE}")
    files = sorted(f for f in RAW_DIR.iterdir() if f.suffix.lower() in MEDIA_EXT)
    if not files:
        sys.exit(f"{RAW_DIR}/ 에 처리할 미디어 파일이 없습니다. mp3/mp4 등을 넣어주세요.")
    print(f"처리 대상 {len(files)}개: " + ", ".join(f.name for f in files))

    import numpy as np
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline
    from faster_whisper import WhisperModel
    from resemblyzer import VoiceEncoder

    print("\n[모델 로딩] pyannote / whisper / 성문인코더 ...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    encoder = VoiceEncoder()

    ref_audio, rsr = sf.read(str(prepare_reference(REF_VOICE, WORK)))
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    ref_embed = encoder.embed_utterance(ref_audio.astype(np.float32))

    all_pairs = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        try:
            turns = process_file(f, pipeline, whisper, encoder, ref_embed)
        except Exception as ex:
            print(f"    !! 오류로 스킵: {ex}")
            continue
        if not turns:
            continue
        pairs = turns_to_pairs(turns)
        print(f"    -> Q&A 쌍 {len(pairs)}개 추출")
        all_pairs.extend(pairs)

    # 중복 제거
    seen, uniq = set(), []
    for u, a in all_pairs:
        key = (u, a)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((u, a))

    train, valid = split_pairs(uniq)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path, pairs):
        with open(path, "w", encoding="utf-8") as fh:
            for u, a in pairs:
                rec = {"messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_jsonl(OUT_DIR / "train.jsonl", train)
    if valid:
        write_jsonl(OUT_DIR / "valid.jsonl", valid)

    print(f"\n===== 완료 =====")
    print(f"  총 Q&A 쌍: {len(uniq)}개 (중복 제거 후)")
    print(f"  train: {len(train)}개 -> {OUT_DIR/'train.jsonl'}")
    if valid:
        print(f"  valid: {len(valid)}개 -> {OUT_DIR/'valid.jsonl'}")
    print("\n샘플 미리보기:")
    for u, a in uniq[:3]:
        print(f"  [user] {u[:40]}...\n  [박사님] {a[:50]}...\n")


if __name__ == "__main__":
    main()
