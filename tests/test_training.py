"""Training regressions; model/audio decoding are the external test boundaries."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_helper(name):
    path = ROOT / "training" / name
    spec = importlib.util.spec_from_file_location("training_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SlicingTests(unittest.TestCase):
    def run_slice(self, samples, intervals, destination):
        # Decoding and encoding need optional native libraries; range decisions stay real.
        def save(path, audio, sr):
            Path(path).write_text(str(len(audio) / sr))
        audio_modules = {
            "numpy": types.ModuleType("numpy"),
            "librosa": types.SimpleNamespace(
                load=lambda *args, **kwargs: (samples, 10),
                effects=types.SimpleNamespace(split=lambda *args, **kwargs: intervals),
            ),
            "soundfile": types.SimpleNamespace(write=save),
        }
        with patch.dict(sys.modules, audio_modules), contextlib.redirect_stdout(io.StringIO()):
            return load_helper("01_slice_audio.py").slice_audio("source.wav", destination)

    def test_long_utterance_keeps_every_sample_with_bounded_balanced_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.run_slice([1] * 205, [(0, 205)], tmp)
            durations = [float(Path(p).read_text()) for p in paths]
            self.assertTrue(all(3 <= duration <= 10 for duration in durations), durations)
            self.assertAlmostEqual(sum(durations), 20.5)
            self.assertEqual([Path(p).name for p in paths],
                             ["seg_0000.wav", "seg_0001.wav", "seg_0002.wav"])

    def test_short_clip_merge_never_crosses_maximum(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.run_slice([1] * 200, [(0, 20), (110, 200)], tmp)
            self.assertEqual([float(Path(p).read_text()) for p in paths], [2.0, 9.0])

    def test_neighboring_short_clips_merge_without_dropping_speech(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.run_slice([1] * 60, [(0, 20), (30, 60)], tmp)
            self.assertEqual([float(Path(p).read_text()) for p in paths], [6.0])

    def test_empty_and_silent_input_emit_no_training_clips(self):
        for samples, intervals in [([], []), ([0] * 100, [(0, 100)])]:
            with self.subTest(samples=len(samples)), tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(self.run_slice(samples, intervals, tmp), [])
                self.assertEqual(list(Path(tmp).glob("*.wav")), [])

    def test_isolated_short_speech_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.run_slice([1] * 5, [(0, 5)], tmp)
            self.assertEqual([float(Path(p).read_text()) for p in paths], [0.5])

    def test_existing_segments_are_not_overwritten_or_mixed_with_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "seg_0000.wav"
            existing.write_text("original")
            with self.assertRaises(FileExistsError):
                self.run_slice([1] * 60, [(0, 60)], tmp)
            self.assertEqual(existing.read_text(), "original")


@unittest.skipUnless(importlib.util.find_spec("librosa") and importlib.util.find_spec("soundfile"),
                     "optional audio libraries are not installed")
class RealAudioSlicingTests(unittest.TestCase):
    def test_synthetic_wav_is_resampled_bounded_and_not_truncated(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            audio = 0.5 * np.sin(2 * np.pi * 220 * np.arange(164000) / 8000)
            sf.write(source, audio, 8000)
            with contextlib.redirect_stdout(io.StringIO()):
                paths = load_helper("01_slice_audio.py").slice_audio(source, Path(tmp) / "segments")
            clips = [sf.info(path) for path in paths]
            self.assertEqual([clip.samplerate for clip in clips], [32000, 32000, 32000])
            self.assertTrue(all(3 <= clip.duration <= 10 for clip in clips))
            self.assertAlmostEqual(sum(clip.duration for clip in clips), 20.5, places=3)

    def test_silent_wav_generates_no_clips(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "silent.wav"
            sf.write(source, np.zeros(32000), 8000)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(load_helper("01_slice_audio.py").slice_audio(
                    source, Path(tmp) / "segments"), [])


class TranscriptionTests(unittest.TestCase):
    def run_transcription(self, source, speaker, output, transcripts):
        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass
            def transcribe(self, wav_path, **kwargs):
                return iter([types.SimpleNamespace(text=transcripts[Path(wav_path).name])]), None
        with patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeModel)}), contextlib.redirect_stdout(io.StringIO()):
            return load_helper("02_transcribe.py").transcribe_segments(source, speaker, output)

    def test_relative_input_produces_absolute_sorted_paths_and_creates_output_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve() / "audio"
            source.mkdir()
            for filename in ["b.wav", "a.wav"]:
                (source / filename).touch()
            output = Path(tmp) / "lists" / "train.list"
            relative = os.path.relpath(source, Path.cwd())
            try:
                self.run_transcription(relative, "speaker", output,
                                       {"a.wav": " hello ", "b.wav": "world"})
            except FileNotFoundError:
                self.fail("The transcription helper must create the list output directory")
            self.assertEqual(output.read_text().splitlines(), [
                f"{source / 'a.wav'}|speaker|ko|hello",
                f"{source / 'b.wav'}|speaker|ko|world",
            ])

    def test_invalid_speaker_cannot_corrupt_list_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve()
            (source / "a.wav").touch()
            for speaker in ["", "  ", "a|b", "a\nb"]:
                with self.subTest(speaker=speaker), self.assertRaises(ValueError):
                    self.run_transcription(source, speaker, source / "train.list", {"a.wav": "hello"})

    def test_invalid_output_is_rejected_before_loading_model_and_preserves_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve()
            audio = source / "a.wav"
            audio.write_bytes(b"original audio")
            alias = source / "alias.list"
            alias.symlink_to(audio)
            module = load_helper("02_transcribe.py")
            for output in (audio, source / "labels.txt", alias):
                with self.subTest(output=output.name), patch.dict(sys.modules, {"faster_whisper": None}):
                    with self.assertRaises(ValueError):
                        module.transcribe_segments(source, "speaker", output)
                    self.assertEqual(audio.read_bytes(), b"original audio")
            self.assertFalse((source / "labels.txt").exists())

    def test_uppercase_list_suffix_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve()
            (source / "a.wav").touch()
            output = source / "train.LIST"
            self.run_transcription(source, "speaker", output, {"a.wav": "hello"})
            self.assertEqual(output.read_text(), f"{source / 'a.wav'}|speaker|ko|hello\n")

    def test_all_empty_transcripts_fail_without_overwriting_existing_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve()
            (source / "a.wav").touch()
            output = source / "train.list"
            output.write_text("previous valid data")
            with self.assertRaises(ValueError):
                self.run_transcription(source, "speaker", output, {"a.wav": "  "})
            self.assertEqual(output.read_text(), "previous valid data")

    def test_empty_transcripts_are_skipped_and_delimiters_cannot_create_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp).resolve()
            for filename in ["a.wav", "b.wav"]:
                (source / filename).touch()
            output = source / "train.list"
            self.run_transcription(source, "speaker", output,
                                   {"a.wav": "  ", "b.wav": "one | two\nthree"})
            self.assertEqual(output.read_text().splitlines(),
                             [f"{source / 'b.wav'}|speaker|ko|one two three"])

    def test_no_audio_is_a_clear_validation_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                self.run_transcription(tmp, "speaker", Path(tmp) / "train.list", {})
            except ValueError:
                pass
            except SystemExit:
                self.fail("A library call must report missing audio as ValueError, not terminate the caller")
            else:
                self.fail("Missing audio must fail before creating a training list")


class SetupTests(unittest.TestCase):
    def test_setup_anchors_checkout_and_venv_to_script_and_prints_remaining_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            training = home / "project" / "training"
            training.mkdir(parents=True)
            source = ROOT / "training" / "00_setup.sh"
            script = training / "00_setup.sh"
            script.write_text(source.read_text())
            binaries = home / "bin"
            binaries.mkdir()
            (binaries / "git").write_text(r"""#!/bin/bash
set -eu
if [[ "$1" == clone ]]; then
    target="${@: -1}"
    mkdir -p "$target/.git"
    exit 0
fi
if [[ "$1" == -C ]]; then
    cd "$2"
    shift 2
fi
case "$1" in
    checkout) printf '%s' "${@: -1}" > .git/test-head ;;
    rev-parse) if [[ "${@: -1}" == HEAD ]]; then cat .git/test-head; else printf '%s' "${@: -1}" | sed 's/\^{commit}//'; fi ;;
    *) exit 9 ;;
esac
""")
            (binaries / "selected-python").write_text("""#!/bin/bash
set -eu
[[ "$1" == -m && "$2" == venv ]] || exit 9
mkdir -p "$3/bin"
printf 'prepared' > "$3/bin/python"
chmod +x "$3/bin/python"
""")
            (binaries / "git").chmod(0o755)
            (binaries / "selected-python").chmod(0o755)
            (binaries / "python3").symlink_to(binaries / "selected-python")
            (binaries / "pip").write_text("#!/bin/bash\nexit 98\n")
            (binaries / "pip").chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                       PYTHON=str(binaries / "selected-python"), GPT_SOVITS_REF="test-pinned-ref")
            result = subprocess.run(["bash", str(script)], cwd=home, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue((training / "GPT-SoVITS" / "venv" / "bin" / "python").exists())
            self.assertEqual((training / "GPT-SoVITS" / ".git" / "test-head").read_text(),
                             "test-pinned-ref")
            self.assertFalse((home / "GPT-SoVITS").exists())
            self.assertIn("pretrained", result.stdout)
            second = subprocess.run(["bash", str(script)], cwd=home, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            head = training / "GPT-SoVITS" / ".git" / "test-head"
            head.write_text("local-custom-revision")
            mismatched = subprocess.run(["bash", str(script)], cwd=home, env=env,
                                        capture_output=True, text=True)
            self.assertNotEqual(mismatched.returncode, 0)
            self.assertEqual(head.read_text(), "local-custom-revision")


if __name__ == "__main__":
    unittest.main()
