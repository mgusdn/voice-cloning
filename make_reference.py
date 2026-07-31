"""
타겟 화자(SPEAKER_01)의 깨끗한 긴 구간만 원본 mp3(고음질)에서 다시 추출해
TTS 보이스 클로닝용 레퍼런스 wav를 만든다.

- 입력: test.mp3 (원본 44.1kHz)
- 사용 구간: diarize 결과 중 길이가 긴 SPEAKER_01 겹침제거 구간
- 출력: reference_speaker01.wav (24kHz mono, TTS 클로닝에 적합)
"""

import subprocess
from pathlib import Path

SRC = "test.mp3"
OUT = "reference_speaker01.wav"

# diarize 결과에서 고른 SPEAKER_01의 "긴+깨끗한" 구간 (start, end)
# 너무 짧은 토막은 제외하고, 발음이 충분히 담긴 3초 이상 구간 위주로 선별
GOOD = [
    (0.27, 7.96),    # 7.7s
    (17.87, 22.71),  # 4.8s
    (38.71, 44.06),  # 5.3s
    (64.19, 69.05),  # 4.9s
    (81.40, 84.91),  # 3.5s  (01:21.40~01:24.91)
    (85.47, 90.08),  # 4.6s  (01:25.47~01:30.08)
    (90.70, 112.67), # 22.0s (01:30.70~01:52.67)
    (113.21, 120.23),# 7.0s  (01:53.21~02:00.23)
    (120.65, 148.18),# 27.5s (02:00.65~02:28.18)
]

tmp = Path("_ref_parts")
tmp.mkdir(exist_ok=True)
parts = []
for i, (s, e) in enumerate(GOOD):
    p = tmp / f"part_{i:02d}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", SRC, "-ss", str(s), "-to", str(e),
         "-ac", "1", "-ar", "24000", str(p)],
        check=True, capture_output=True,
    )
    parts.append(p)

listfile = tmp / "list.txt"
listfile.write_text("".join(f"file '{p.name}'\n" for p in parts))
subprocess.run(
    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
     "-c", "copy", OUT],
    check=True, capture_output=True,
)

dur = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", OUT],
    capture_output=True, text=True,
).stdout.strip()
print(f"레퍼런스 생성 완료: {OUT}  (길이 {float(dur):.1f}초, 24kHz mono)")
