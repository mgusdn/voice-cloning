"""Exercise the chatbot entry point without network or microphone access."""
import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ChatbotEntryPointTests(unittest.TestCase):
    def test_help_runs_without_any_installed_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "chatbot", "--help"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mode", result.stdout)
        self.assertIn("--tts", result.stdout)

    def test_import_does_not_load_graph_or_microphone_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-S", "-c", (
                "import sys; import chatbot.cli; "
                "assert 'chatbot.graph' not in sys.modules; "
                "assert 'chatbot.stt' not in sys.modules; "
                "assert 'sounddevice' not in sys.modules"
            )],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_eof_at_nickname_exits_without_loading_models(self):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "chatbot"],
            cwd=ROOT, input="", capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class ChatbotSessionTests(unittest.TestCase):
    def setUp(self):
        self.cli = importlib.import_module("chatbot.cli")
        self.nodes = importlib.import_module("chatbot.nodes")
        self.model_patch = patch.object(
            self.nodes, "call_openai_json", return_value={"reflection": "이해했어요."}
        )
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def run_cli(self, args, answers):
        answers = iter(answers)

        def read_answer(prompt):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError from None

        output = io.StringIO()
        with patch("builtins.input", side_effect=read_answer), contextlib.redirect_stdout(output):
            status = self.cli.main(args)
        return status, output.getvalue()

    def test_text_session_speaks_each_response_with_distinct_rapport_turns(self):
        speech = []
        instances = []

        class Speaker:
            def __init__(self):
                instances.append(self)

            def speak(self, text, turn, play=True):
                speech.append((text, turn, play))
                return None

        tts_module = types.ModuleType("voice_clone.tts_client")
        tts_module.TTSClient = Speaker
        with patch.dict(sys.modules, {"chatbot.stt": None, "voice_clone.tts_client": tts_module}):
            status, output = self.run_cli(
                ["--tts", "--debug", "True"], ["테스터", "안녕하세요", "괜찮아요", "quit"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(len(instances), 1)
        self.assertEqual([turn for _, turn, _ in speech], [0, 1, 2])
        self.assertTrue(all(play for _, _, play in speech))
        self.assertIn("테스터", speech[0][0])
        self.assertIn("이해했어요.", speech[1][0])
        self.assertEqual(output.count("상담사:"), 3)
        self.assertEqual(output.count("stage=rapport turn_count=0"), 3)

    def test_text_input_eof_ends_session_after_greeting(self):
        with patch.dict(sys.modules, {"chatbot.stt": None, "voice_clone.tts_client": None}):
            status, output = self.run_cli(["--mode", "text"], ["테스터"])
        self.assertEqual(status, 0)
        self.assertEqual(output.count("상담사:"), 1)

    def test_voice_mode_uses_selected_device_and_closes_audio_stream(self):
        observed = {}
        stt_module = types.ModuleType("chatbot.stt")
        stt_module.DEVICE_NAME_PRIORITY = ["headset"]
        stt_module.find_input_device = lambda name: 8
        stt_module.find_priority_input_device = lambda hints: 9
        stt_module.load_model = lambda: "test-model"

        def audio_stream(model, device=None, debug=False):
            observed.update(model=model, device=device, debug=debug)
            try:
                yield "오늘은 괜찮아요", time.perf_counter()
            finally:
                observed["closed"] = True

        stt_module.transcribe_stream = audio_stream
        with patch.dict(sys.modules, {"chatbot.stt": stt_module, "voice_clone.tts_client": None}):
            status, output = self.run_cli(
                ["--mode", "voice", "--device", "3", "--debug"], ["테스터"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(observed, {"model": "test-model", "device": 3, "debug": True, "closed": True})
        self.assertIn("오늘은 괜찮아요", output)
        self.assertEqual(output.count("상담사:"), 2)


if __name__ == "__main__":
    unittest.main()
