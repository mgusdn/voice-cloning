"""chatbot/(juminsuh/chatbot 순수 클론)를 한 글자도 안 건드리고 my_voice 음성을 붙여서 실행.

chatbot/ 안의 graph.get_graph()가 반환하는 객체를 감싸서, 매 턴 invoke() 결과의
bot_message를 TTS로 재생한 뒤 그대로 돌려준다. cli.py는 이 프록시 객체를
원본 graph인 줄 알고 그대로 쓴다.

cli.py가 마이크 음성 입력(STT)만 지원하도록 바뀌어서, 마이크 없이 테스트할 수
있게 텍스트 입력 모드를 여기서 별도로 재현한다 (예전 cli.py의 input() 루프와 동일).

사용법 (chatbot/venv 활성화한 상태에서):
    python run_with_voice.py            # 실행할 때 텍스트/음성 중 선택
    python run_with_voice.py --mode text
    python run_with_voice.py --mode voice --debug True
"""
import argparse
import sys
import time
from pathlib import Path

CHATBOT_DIR = Path(__file__).resolve().parent.parent / "chatbot"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tts_client
sys.path.insert(0, str(CHATBOT_DIR))  # chatbot의 top-level 모듈들 (graph, cli, ...)

from tts_client import speak
import graph as graph_module


class _VoiceGraph:
    """graph.get_graph()가 반환하는 CompiledGraph를 감싸는 얇은 프록시.

    chatbot의 turn_count는 rapport 단계에서 안 올라가서 파일명으로 쓰기엔
    부적합 (초반 여러 턴이 같은 값을 가짐) -> 여기서 별도로 turn을 센다.
    """

    def __init__(self, inner):
        self._inner = inner
        self._turn = 0

    def invoke(self, state, *args, **kwargs):
        gen_start = time.perf_counter()
        result = self._inner.invoke(state, *args, **kwargs)
        gen_elapsed = time.perf_counter() - gen_start
        print(f"  [응답 생성 시간: {gen_elapsed:.2f}초]")
        speak(result.get("bot_message"), self._turn)
        self._turn += 1
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


_original_get_graph = graph_module.get_graph


def _patched_get_graph():
    return _VoiceGraph(_original_get_graph())


graph_module.get_graph = _patched_get_graph


def _run_text_mode() -> None:
    """마이크 없이 텍스트로 테스트 (예전 cli.py의 input() 루프를 그대로 재현)."""
    from state import new_session

    graph = graph_module.get_graph()
    name = input("닉네임을 입력해주세요: ").strip()
    state = new_session(name)

    print("[상담을 시작합니다 - 종료하려면 'quit' 입력]\n")
    state = graph.invoke(state)
    print(f"상담사: {state['bot_message']}\n")

    while state["stage"] != "done":
        try:
            user_input = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit"):
            break
        state["user_input"] = user_input
        state = graph.invoke(state)
        print(f"\n상담사: {state['bot_message']}\n")

    if state["stage"] == "done":
        print("[상담이 마무리되었습니다. 대화를 종료합니다]")


def _run_voice_mode(extra_argv: list[str]) -> None:
    """cli.py 그대로 실행 (마이크 STT 입력)."""
    import cli

    sys.argv = [sys.argv[0]] + extra_argv
    cli.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "voice"], default=None)
    args, remaining = parser.parse_known_args()

    mode = args.mode
    if mode is None:
        choice = input("입력 방식을 선택하세요 [1] 텍스트  [2] 음성(마이크): ").strip()
        mode = "voice" if choice == "2" else "text"

    if mode == "text":
        _run_text_mode()
    else:
        _run_voice_mode(remaining)
