# 통합 검증 기록

로컬 검증 환경: macOS Apple Silicon, Python 3.13.14. 작업용 가상환경에 `requirements-dev.txt`에 해당하는 라이브러리를 설치했다.

## 통과한 검증

- `python -m unittest discover -s tests -v`: **45개 통과, 실패·건너뜀 없음**.
- Python 컴파일: chatbot, voice_clone, training, tools 및 루트 오디오 도구.
- `bash -n training/00_setup.sh`, `git diff --check`.
- `python -S`에서 주요 CLI 9개의 `--help`: 외부 패키지·API 키·마이크 없이 통과.
- `pip check`: 의존성 충돌 없음.
- 원본과 보존된 세 `transcript.jsonl` 파일을 바이트 단위 비교: 64 + 81 + 95 = **240개 전사문 동일**.
- 기존 전사 스키마 두 종류의 목록 복원과 새 절대경로 검증.
- 레거시 GPT-SoVITS 설치, 새 설치, 원본 음원, `.env`의 Git 제외 확인. `.env.example`은 추적 대상.

실제 LangGraph를 사용하고 외부 LLM 호출만 대체해 텍스트 세션과 음성 출력 순서를 검증했다. 테스트는 합성 WAV와 실제 ffmpeg/librosa/soundfile로 구간 추출·리샘플링·길이 제한·무음 처리를 확인한다. TTS는 제어된 HTTP 응답 경계로 정상 WAV, 스트리밍 헤더, 전송 실패, 플레이어 정리, 파일 보존과 CLI 종료 코드를 확인한다.

## 리뷰에서 수정한 문제

1. 이전 `files/GPT-SoVITS/` 설치가 새 브랜치에서 Git 추적 대상으로 노출될 수 있어 이전 제외 규칙도 유지했다.
2. 원본 길이 밖 참조 구간이 빈 WAV를 성공으로 저장하던 문제를 수정했다. 임시 파일의 실제 오디오 프레임을 확인한 뒤 교체한다.
3. 학습 목록 출력 경로가 원본 WAV를 덮어쓸 수 있어 `.list` 확장자와 원본 파일 충돌을 사전에 검사한다.

## 외부 실행 범위

실제 GPT-SoVITS 모델 다운로드·설치·학습·음질 평가, pyannote/Whisper/Demucs 모델 추론, 실시간 마이크·스피커, 유료 LLM 호출은 실행하지 않았다. 이 검증만으로 특정 가중치나 장치에서의 음성 품질·성능을 보장하지 않는다. 해당 데이터와 자격 증명을 준비한 뒤 README의 단계로 확인해야 한다.

GitHub Actions는 Ubuntu/Python 3.11·3.13에서 같은 테스트, 컴파일, shell 문법 및 CLI 확인을 실행하도록 구성했다. 실제 실행 결과는 PR의 Checks에서 확인할 수 있다.
