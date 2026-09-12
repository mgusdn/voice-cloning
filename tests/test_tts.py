import io
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import requests

from voice_clone.tts_client import TTSClient, TTSConfig, TTSError


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(32000)
        wav.writeframes(b"\x01\x00" * 100)
    return buffer.getvalue()


class Response:
    def __init__(self, chunks, failure=None):
        self.chunks = chunks
        self.failure = failure
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self.chunks
        if self.failure:
            raise self.failure

    def close(self):
        self.closed = True


class Transport:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def post(self, url, **kwargs):
        self.payload = kwargs["json"]
        return self.response


class Player:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.terminated = False
        self.waited = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        self.waited = True


class TTSTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.config = TTSConfig(ref_audio_path="/server/ref.wav", ref_text="참조 문장")

    def client(self, response):
        transport = Transport(response)
        return TTSClient(self.config, output_dir=self.output, transport=transport), transport

    def test_synthesis_saves_complete_wav_and_closes_response(self):
        body = wav_bytes()
        response = Response([body[:5], b"", body[5:]])
        client, transport = self.client(response)
        path = client.speak("안녕하세요", 0, play=False)
        self.assertIsNotNone(path)
        self.assertEqual(Path(path).read_bytes(), body)
        self.assertTrue(response.closed)
        self.assertEqual(transport.payload["ref_audio_path"], "/server/ref.wav")
        self.assertEqual(transport.payload["streaming_mode"], 0)

    def test_streaming_wav_header_is_finalized_for_replay(self):
        body = bytearray(wav_bytes())
        struct.pack_into("<I", body, 4, 36)
        struct.pack_into("<I", body, 40, 0)
        response = Response([bytes(body[:44]), bytes(body[44:])])
        client, transport = self.client(response)
        with patch("voice_clone.tts_client.subprocess.Popen", side_effect=FileNotFoundError):
            path = client.speak("안녕하세요", 1)
        with wave.open(path, "rb") as wav:
            self.assertEqual(wav.getnframes(), 100)
        self.assertEqual(transport.payload["streaming_mode"], 3)
        self.assertEqual(transport.payload["batch_size"], 1)

    def test_disconnect_preserves_old_output_and_removes_partial_file(self):
        target = self.output / "result.wav"
        target.write_bytes(b"existing")
        response = Response([wav_bytes()[:20]], requests.ConnectionError("dropped"))
        client, _ = self.client(response)
        with self.assertRaises(TTSError):
            client.synthesize("hello", target)
        self.assertEqual(target.read_bytes(), b"existing")
        self.assertEqual(list(self.output.iterdir()), [target])
        self.assertTrue(response.closed)

    def test_speak_degrades_to_text_on_midstream_error(self):
        response = Response([b"RIFF"], requests.ConnectionError("dropped"))
        client, _ = self.client(response)
        self.assertIsNone(client.speak("hello", 0, play=False))
        self.assertFalse(list(self.output.rglob("*.wav")))
        self.assertTrue(response.closed)

    def test_missing_config_does_not_make_request(self):
        transport = Transport(Response([]))
        client = TTSClient(TTSConfig(), output_dir=self.output, transport=transport)
        self.assertIsNone(client.speak("hello", 0, play=False))
        self.assertIsNone(transport.payload)

    def test_json_or_empty_audio_is_not_saved_as_success(self):
        for body in [b'{"error":"failed"}', wav_bytes()[:44], b""]:
            with self.subTest(body=body):
                response = Response([body])
                client, _ = self.client(response)
                self.assertIsNone(client.speak("hello", 0, play=False))
                self.assertTrue(response.closed)

    def test_environment_takes_priority_over_dotenv(self):
        env_file = self.output / ".env"
        env_file.write_text("TTS_REF_TEXT=from-file\nTTS_REF_AUDIO_PATH=/file.wav\n")
        with patch.dict(os.environ, {"TTS_REF_TEXT": "from-env"}, clear=True):
            config = TTSConfig.from_env(env_file)
        self.assertEqual(config.ref_text, "from-env")
        self.assertEqual(config.ref_audio_path, "/file.wav")

    def test_sessions_do_not_overwrite_same_turn_number(self):
        first, _ = self.client(Response([wav_bytes()]))
        second, _ = self.client(Response([wav_bytes()]))
        self.assertNotEqual(first.speak("hi", 0, False), second.speak("hi", 0, False))

    def test_interrupted_stream_closes_and_reaps_player(self):
        player = Player()
        response = Response([wav_bytes()[:44]], requests.ConnectionError("dropped"))
        client, _ = self.client(response)
        with patch("voice_clone.tts_client.subprocess.Popen", return_value=player):
            self.assertIsNone(client.speak("hello", 0))
        self.assertTrue(player.stdin.closed)
        self.assertTrue(player.terminated)
        self.assertTrue(player.waited)

    def test_cli_failure_has_nonzero_exit_status(self):
        from voice_clone.infer import main
        with patch("voice_clone.tts_client.TTSClient.synthesize", side_effect=TTSError("unavailable")):
            with self.assertRaises(SystemExit) as result:
                main(["hello", "--output", str(self.output / "result.wav")])
        self.assertEqual(result.exception.code, 1)
