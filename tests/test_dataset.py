"""Dataset regressions; no model downloads or machine-specific audio needed."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave


ROOT = Path(__file__).resolve().parents[1]


def run_cli(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        capture_output=True, text=True,
    )


def write_wav(path, seconds=3, sample_rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x01\x00" * int(seconds * sample_rate))


class DatasetTests(unittest.TestCase):
    def load_module(self, relative):
        path = ROOT / relative
        self.assertTrue(path.is_file(), f"Missing importable dataset tool: {relative}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except (ImportError, SystemExit) as exc:
            self.fail(f"Dataset helpers must import without loading ML dependencies: {exc}")
        return module

    def test_dialogue_split_retains_every_sample_exactly_once(self):
        module = self.load_module("tools/build_dialogue_dataset.py")
        for count in (0, 20, 21, 41, 61, 99, 101, 121):
            with self.subTest(count=count):
                pairs = [(f"question {i}", f"answer {i}") for i in range(count)]
                train, valid = module.split_pairs(pairs)
                self.assertCountEqual(train + valid, pairs)
                self.assertFalse(set(train) & set(valid))
        _, valid = module.split_pairs([(str(i), str(i)) for i in range(41)])
        self.assertEqual(len(valid), 2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for audio integration")
    def test_dialogue_reference_is_resampled_before_embedding(self):
        module = self.load_module("tools/build_dialogue_dataset.py")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "reference_24k.wav"
            write_wav(source, sample_rate=24000)
            prepared = module.prepare_reference(source, Path(tmp))
            with wave.open(str(prepared), "rb") as wav:
                self.assertEqual(wav.getframerate(), 16000)
                self.assertAlmostEqual(wav.getnframes() / wav.getframerate(), 3, places=2)

    def test_lists_reconstruct_both_transcript_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp).resolve() / "화자"
            audio = dataset / "wavs"
            audio.mkdir(parents=True)
            (audio / "first.wav").touch()
            (audio / "second.wav").touch()
            transcript = dataset / "transcript.jsonl"
            transcript.write_text(
                json.dumps({"file": "first.wav", "text": "첫 문장"}, ensure_ascii=False) + "\n"
                + json.dumps({"wav": "/obsolete/second.wav", "text": "둘째 문장"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = run_cli("tools/rebuild_lists.py", transcript)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((dataset / "화자.list").read_text(encoding="utf-8"),
                             f"{audio / 'first.wav'}|화자|ko|첫 문장\n"
                             f"{audio / 'second.wav'}|화자|ko|둘째 문장\n")

    def test_missing_audio_is_reported_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            transcript.write_text('{"file":"missing.wav","text":"hello"}\n')
            output = Path(tmp) / "train.list"
            result = run_cli("tools/rebuild_lists.py", transcript, "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing.wav", result.stderr)
            self.assertIn("--audio-dir", result.stderr)
            self.assertFalse(output.exists())

    def test_explicit_colab_target_does_not_require_local_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "transcript.jsonl"
            transcript.write_text('{"wav":"/old/clip.wav","text":"hello"}\n')
            output = Path(tmp) / "train.list"
            result = run_cli("tools/rebuild_lists.py", transcript, "--speaker", "speaker",
                             "--colab", "/content/speaker/wavs", "--output", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), "/content/speaker/wavs/clip.wav|speaker|ko|hello\n")

    def test_list_output_cannot_replace_source_audio_or_use_wrong_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "wavs"
            audio.mkdir()
            source = audio / "clip.wav"
            write_wav(source)
            original = source.read_bytes()
            transcript = root / "transcript.jsonl"
            transcript.write_text('{"file":"clip.wav","text":"hello"}\n')
            alias = root / "alias.list"
            alias.symlink_to(source)
            for output in (source, root / "labels.txt", alias):
                with self.subTest(output=output.name):
                    result = run_cli("tools/rebuild_lists.py", transcript, "--output", output)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(source.read_bytes(), original)
            self.assertFalse((root / "labels.txt").exists())

    def test_list_output_cannot_replace_transcript_through_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.jsonl"
            transcript.write_text('{"file":"clip.wav","text":"hello"}\n')
            original = transcript.read_bytes()
            alias = root / "labels.list"
            alias.symlink_to(transcript)
            result = run_cli("tools/rebuild_lists.py", transcript, "--colab", "/content/voice/wavs",
                             "--output", alias)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(transcript.read_bytes(), original)

    def test_reference_rejects_reversed_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            write_wav(source)
            result = run_cli("make_reference.py", source, Path(tmp) / "reference.wav",
                             "--start", "2", "--end", "1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--end", result.stderr)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for audio integration")
    def test_reference_extracts_only_requested_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.wav", Path(tmp) / "reference.wav"
            write_wav(source)
            result = run_cli("make_reference.py", source, output, "--start", "1", "--end", "2")
            self.assertEqual(result.returncode, 0, result.stderr)
            with wave.open(str(output), "rb") as wav:
                self.assertEqual(wav.getframerate(), 24000)
                self.assertAlmostEqual(wav.getnframes() / wav.getframerate(), 1, places=2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for audio integration")
    def test_reference_past_end_preserves_existing_output_without_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source.wav", root / "reference.wav"
            write_wav(source, seconds=3)
            write_wav(output, seconds=1)
            original = output.read_bytes()
            result = run_cli("make_reference.py", source, output, "--start", "10", "--end", "11")
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(output.read_bytes(), original)
            self.assertCountEqual([path.name for path in root.iterdir()], ["source.wav", "reference.wav"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for audio integration")
    def test_reference_invalid_source_leaves_existing_output_and_no_temporary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "broken.wav", root / "reference.wav"
            source.write_bytes(b"not a valid WAV")
            write_wav(output, seconds=1)
            original = output.read_bytes()
            result = run_cli("make_reference.py", source, output, "--start", "0", "--end", "1")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            self.assertCountEqual([path.name for path in root.iterdir()], ["broken.wav", "reference.wav"])

    def test_demucs_result_belongs_to_current_source(self):
        module = self.load_module("build_voice_dataset.py")
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            old = work / "htdemucs" / "old" / "vocals.wav"
            old.parent.mkdir(parents=True)
            old.write_text("old audio")

            def run_demucs(command):
                target = Path(command[command.index("-o") + 1]) / "htdemucs" / "new" / "vocals.wav"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("current audio")

            with patch.object(module, "run", run_demucs):
                result = module.separate_vocals(Path("new.wav"), work)
            self.assertEqual(result.read_text(), "current audio")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for audio integration")
    def test_diarization_merge_excludes_previous_run_clips(self):
        module = self.load_module("diarize.py")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            outdir = Path(tmp) / "segments"
            outdir.mkdir()
            write_wav(source)
            write_wav(outdir / "SPEAKER_00_999_old.wav", seconds=2)
            module.export_speaker_audio(source, outdir, [(0.0, 1.0, "SPEAKER_00")])
            with wave.open(str(outdir / "MERGED_SPEAKER_00.wav"), "rb") as wav:
                self.assertAlmostEqual(wav.getnframes() / wav.getframerate(), 1, places=2)


if __name__ == "__main__":
    unittest.main()
