"""Text or microphone counseling sessions, with optional cloned speech output."""
import argparse
import time

from .debug import debug_print, set_debug
from .state import new_session


def _str2bool(value: str) -> bool:
    if value.strip().lower() in ("true", "1", "yes"):
        return True
    if value.strip().lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("true 또는 false를 입력하세요")


def _print_debug(state: dict) -> None:
    print(
        f"  [stage={state['stage']} turn_count={state['turn_count']} "
        f"gate={state['gate']} selected_values={state.get('selected_values')}]"
    )
    for slot, values in state["slots"].items():
        content = " / ".join(values) if values else "X"
        print(f"    - {slot}: {content}")
    print(f"  [pending(target_slot/question_intent)={state.get('pending')}]")
    print(f"  [asked_slots={state.get('asked_slots')}]")


def _respond(graph, state, speaker, turn: int, debug: bool):
    started = time.perf_counter()
    state = graph.invoke(state)
    debug_print(f"[TIMING DEBUG] 응답 생성까지 걸린 시간: {time.perf_counter() - started:.2f}초")
    print(f"\n상담사: {state['bot_message']}\n")
    if debug:
        _print_debug(state)
    if speaker is not None:
        speaker.speak(state.get("bot_message"), turn, play=True)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("text", "voice"), default="text")
    parser.add_argument("--tts", action="store_true", help="클로닝 목소리로 응답 재생")
    parser.add_argument("--debug", nargs="?", const=True, type=_str2bool, default=False)
    parser.add_argument(
        "--device", default=None,
        help="음성 모드 입력 장치 이름 일부 또는 인덱스 (기본: 헤드셋 > 아이폰 순 탐색)",
    )
    args = parser.parse_args(argv)
    set_debug(args.debug)

    try:
        name = input("닉네임을 입력해주세요: ").strip()
    except (EOFError, KeyboardInterrupt):
        return 0

    # Help, module imports, and text input never load microphone libraries.
    from .graph import get_graph

    graph = get_graph()
    state = new_session(name)
    speaker = None
    if args.tts:
        from voice_clone.tts_client import TTSClient

        speaker = TTSClient()

    listen = None
    if args.mode == "voice":
        from . import stt

        if args.device is not None:
            device = int(args.device) if args.device.isdigit() else stt.find_input_device(args.device)
        else:
            device = stt.find_priority_input_device(stt.DEVICE_NAME_PRIORITY)
        listen = stt.transcribe_stream(stt.load_model(), device=device, debug=args.debug)

    print("[상담을 시작합니다 - 텍스트 모드: quit / 음성 모드: Ctrl+C로 종료]\n")
    # Rapport keeps state['turn_count'] at zero, so audio needs its own counter.
    turn = 0
    try:
        state = _respond(graph, state, speaker, turn, args.debug)
        turn += 1
        while state["stage"] != "done":
            last_loud_time = None
            if listen is None:
                user_input = input("나: ").strip()
                if user_input.lower() in ("quit", "exit"):
                    break
            else:
                print("나: (말씀해주세요...)")
                try:
                    user_input, last_loud_time = next(listen)
                except StopIteration:
                    break
                print(f"나: {user_input}\n")

            state["user_input"] = user_input
            state = _respond(graph, state, speaker, turn, args.debug)
            turn += 1
            if last_loud_time is not None:
                elapsed = time.perf_counter() - last_loud_time
                debug_print(f"[TIMING DEBUG] 음성 입력부터 응답 완료까지: {elapsed:.2f}초")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if listen is not None:
            listen.close()

    if state["stage"] == "done":
        print("[상담이 마무리되었습니다. 대화를 종료합니다]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
