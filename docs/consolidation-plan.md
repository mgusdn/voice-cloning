# Voice repository consolidation implementation plan

**Goal:** One runnable, documented voice-cloning repository with useful source preserved and duplicate runtime removed.

**Architecture:** Python packages `chatbot` and `voice_clone`, standalone dataset CLIs, `training` helpers. Share one HTTP TTS client instead of importing GPT-SoVITS internals or monkeypatching the chatbot graph.

**Spec:** `docs/consolidation-design.md`

## Tasks

- [x] Chatbot: use relative imports; consolidate text/voice CLI with optional TTS; defer STT imports; root environment configuration; preserve graph behavior. Verify no-mic text mode and one speech call per response with independent turn numbers.
- [x] Training: move `files` to `training`; correct slicing maximum duration and absolute transcription list paths; make setup independent of caller cwd and report missing model setup accurately. Add synthetic audio/path regression tests.
- [x] Dataset: preserve all transcript records; replace duplicate personal notebooks with generic Colab guide and generated absolute lists with a reconstruction CLI; make reference extraction explicit and remove hardcoded paths in retained tools. Add path/CLI regression tests.
- [x] Shared TTS: build `voice_clone.tts_client.TTSClient.speak(text, turn, play=True) -> str | None`, safe streaming cleanup and configurable reference; `python -m voice_clone.infer TEXT --output FILE`; remove old inference/demo and XTTS dependency file. Test real file results and transport failures using controlled HTTP responses.
- [x] Integration: root env sample, dependencies, ignore rules, README, provenance/migration map, unittest CI. Verify compile, shell syntax, unit tests, and CLIs with installed lightweight dependencies.
- [x] Review the complete diff and fix actionable regressions. Record verification evidence and runtime limitations in docs/VALIDATION.md; publish via the consolidation PR.

## Boundaries

No destructive repository/history deletion, no model download or paid API call. Keep the 240 existing transcript rows. Tests must exercise observable behavior; use fakes only at external model/network/audio boundaries. Agents own disjoint files and do not commit until the root integrates changes.
