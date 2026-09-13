"""Tests for the Gemini Live engine: tool schema conversion, the half-duplex

echo gate, server-decided barge-in, tool dispatch and the confirm/cancel gate,
turn-detection fallback, and check_ready.

Mirrors tests/test_realtime.py in structure and fixture style, adapted to
gemini_live's message shapes: `realtimeInput`/`serverContent`/`toolCall`/
`toolResponse`/`clientContent` instead of realtime's typed `type` field, and
one shared `_kill_mic` workaround documented in MicrophoneGateTests below.

Run with: python3 -m unittest discover -s tests
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omarchy_voice import feedback, gemini_live
from omarchy_voice.config import Config
from omarchy_voice.realtime import GATE_TOOLS
from omarchy_voice.tools import TOOL_SCHEMAS


class FakeSocket:
    """Records what would have gone over the wire.

    Gemini's protocol has no `type` field — each message is one top-level key
    (`realtimeInput`, `toolResponse`, `clientContent`, ...) — so `events` keys
    off that instead of a `type` value the way test_realtime.py's does.
    """

    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    def events(self, key):
        return [e[key] for e in self.sent if key in e]


def patch_feedback_paths(testcase):
    """Redirect feedback's on-disk state into a temp dir for the test's life."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for name, value in (("LOG_FILE", root / "session.log"),
                        ("STATE_FILE", root / "state.json"),
                        ("STATE_DIR", root),
                        ("RUNTIME_DIR", root)):
        patcher = mock.patch.object(feedback, name, value)
        patcher.start()
        testcase.addCleanup(patcher.stop)
    testcase.addCleanup(tmp.cleanup)
    return root


# --- tool schema conversion --------------------------------------------------

class ToolConversionTests(unittest.TestCase):
    def setUp(self):
        self.tools = gemini_live.to_gemini_tools()
        self.declarations = self.tools[0]["functionDeclarations"]
        self.by_name = {d["name"]: d for d in self.declarations}

    def test_everything_is_grouped_under_one_function_declarations_entry(self):
        self.assertEqual(len(self.tools), 1)
        self.assertEqual(set(self.tools[0]), {"functionDeclarations"})

    def test_every_executor_tool_is_present(self):
        for schema in TOOL_SCHEMAS:
            with self.subTest(tool=schema["name"]):
                self.assertIn(schema["name"], self.by_name)
                self.assertEqual(self.by_name[schema["name"]]["description"],
                                 schema["description"])

    def test_gate_tools_are_included_alongside_the_executor_tools(self):
        for gate in GATE_TOOLS:
            with self.subTest(tool=gate["name"]):
                self.assertIn(gate["name"], self.by_name)
        self.assertNotIn("confirm_last", {s["name"] for s in TOOL_SCHEMAS})

    def test_object_types_are_upper_cased(self):
        self.assertEqual(self.by_name["hypr_query"]["parameters"]["type"], "OBJECT")

    def test_additional_properties_is_stripped_from_every_tool(self):
        for declaration in self.declarations:
            with self.subTest(tool=declaration["name"]):
                self.assertNotIn("additionalProperties", declaration["parameters"])

    def test_nested_array_and_object_schemas_convert_recursively(self):
        panes = self.by_name["compose_windows"]["parameters"]["properties"]["panes"]
        self.assertEqual(panes["type"], "ARRAY")
        self.assertEqual(panes["items"]["type"], "OBJECT")
        self.assertEqual(panes["items"]["properties"]["kind"]["type"], "STRING")
        self.assertNotIn("additionalProperties", panes["items"])
        # required/enum/description are untouched — only `type` and
        # `additionalProperties` are load-bearing adjustments.
        self.assertEqual(panes["items"]["required"], ["kind", "target"])

    def test_json_serialisable(self):
        json.dumps(self.tools)  # must not raise

    def test_a_flat_schema_gets_only_the_two_adjustments(self):
        converted = gemini_live._schema_for_gemini({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        })
        self.assertEqual(converted, {
            "type": "OBJECT",
            "properties": {"x": {"type": "STRING"}},
            "required": ["x"],
        })

    def test_custom_schema_list_is_honoured_instead_of_the_default(self):
        tools = gemini_live.to_gemini_tools([])
        names = {d["name"] for d in tools[0]["functionDeclarations"]}
        # No executor tools were passed in, but the gate tools are always added.
        self.assertEqual(names, {g["name"] for g in GATE_TOOLS})


# --- turn-detection / activity config ----------------------------------------

class ActivityDetectionTests(unittest.TestCase):
    def setUp(self):
        self.log = patch_feedback_paths(self) / "session.log"

    def session_with(self, turn_detection):
        return gemini_live.GeminiLiveSession(
            Config(dry_run=True, notify=False, gemini_turn_detection=turn_detection))

    def test_high_sensitivity(self):
        result = self.session_with("high")._activity_detection()
        self.assertEqual(result, {
            "automaticActivityDetection": {
                "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
                "endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
            },
        })

    def test_low_sensitivity(self):
        result = self.session_with("low")._activity_detection()
        self.assertEqual(result, {
            "automaticActivityDetection": {
                "startOfSpeechSensitivity": "START_SENSITIVITY_LOW",
                "endOfSpeechSensitivity": "END_SENSITIVITY_LOW",
            },
        })

    def test_off_falls_back_to_high_and_logs_a_warning(self):
        session = self.session_with("off")
        result = session._activity_detection()
        self.assertEqual(result["automaticActivityDetection"]["startOfSpeechSensitivity"],
                         "START_SENSITIVITY_HIGH")
        self.assertIn('gemini_turn_detection = "off" is not implemented yet',
                      self.log.read_text())

    def test_off_is_case_and_whitespace_insensitive(self):
        session = self.session_with("  OFF  ")
        result = session._activity_detection()
        self.assertEqual(result["automaticActivityDetection"]["startOfSpeechSensitivity"],
                         "START_SENSITIVITY_HIGH")
        self.assertIn("not implemented yet", self.log.read_text())

    def test_an_unrecognised_value_falls_back_to_high_silently(self):
        """Unlike "off", a typo or unknown value is not called out — it just
        lands on the SENSITIVITY dict's default via .get(). Worth pinning down
        as a real behavior, not assuming it warns the same way "off" does."""
        session = self.session_with("medium")
        result = session._activity_detection()
        self.assertEqual(result["automaticActivityDetection"]["startOfSpeechSensitivity"],
                         "START_SENSITIVITY_HIGH")
        self.assertFalse(self.log.exists())


# --- session setup message ----------------------------------------------------

class SetupMessageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patch_feedback_paths(self)

    async def test_setup_message_carries_model_persona_tools_and_activity(self):
        config = Config(dry_run=True, notify=False, gemini_turn_detection="low",
                        tasks_enabled=False, vision_enabled=False)
        session = gemini_live.GeminiLiveSession(config)
        with mock.patch.object(gemini_live.capabilities, "manifest", return_value="MANIFEST TEXT"), \
             mock.patch.object(gemini_live.capabilities, "live_state", return_value="DESKTOP STATE"):
            setup = await session._setup_message()
        body = setup["setup"]
        self.assertEqual(body["model"], f"models/{config.gemini_model}")
        text = body["systemInstruction"]["parts"][0]["text"]
        self.assertIn("You are the voice control layer", text)
        self.assertIn("MANIFEST TEXT", text)
        self.assertIn("DESKTOP STATE", text)
        names = {d["name"] for d in body["tools"][0]["functionDeclarations"]}
        self.assertIn("confirm_last", names)
        self.assertIn("hypr_dispatch", names)
        self.assertEqual(body["realtimeInputConfig"], {
            "automaticActivityDetection": {
                "startOfSpeechSensitivity": "START_SENSITIVITY_LOW",
                "endOfSpeechSensitivity": "END_SENSITIVITY_LOW",
            },
        })
        self.assertEqual(
            body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
            config.gemini_voice)
        self.assertEqual(body["generationConfig"]["responseModalities"], ["AUDIO"])
        json.dumps(setup)  # the whole thing must go over the wire as JSON

    async def test_voice_style_is_appended_to_the_system_instruction(self):
        config = Config(dry_run=True, notify=False, tasks_enabled=False, vision_enabled=False,
                        gemini_voice_style="Speak like a calm radio announcer.")
        session = gemini_live.GeminiLiveSession(config)
        with mock.patch.object(gemini_live.capabilities, "manifest", return_value=""), \
             mock.patch.object(gemini_live.capabilities, "live_state", return_value=""):
            setup = await session._setup_message()
        text = setup["setup"]["systemInstruction"]["parts"][0]["text"]
        self.assertIn("# Delivery", text)
        self.assertIn("Speak like a calm radio announcer.", text)

    async def test_no_voice_style_means_no_delivery_section(self):
        config = Config(dry_run=True, notify=False, tasks_enabled=False, vision_enabled=False)
        session = gemini_live.GeminiLiveSession(config)
        with mock.patch.object(gemini_live.capabilities, "manifest", return_value=""), \
             mock.patch.object(gemini_live.capabilities, "live_state", return_value=""):
            setup = await session._setup_message()
        text = setup["setup"]["systemInstruction"]["parts"][0]["text"]
        self.assertNotIn("# Delivery", text)

    async def test_disabled_features_drop_their_tools_from_the_setup_message(self):
        config = Config(dry_run=True, notify=False, tasks_enabled=False, vision_enabled=False)
        session = gemini_live.GeminiLiveSession(config)
        with mock.patch.object(gemini_live.capabilities, "manifest", return_value=""), \
             mock.patch.object(gemini_live.capabilities, "live_state", return_value=""):
            setup = await session._setup_message()
        names = {d["name"] for d in setup["setup"]["tools"][0]["functionDeclarations"]}
        self.assertNotIn("task_submit", names)
        self.assertNotIn("camera_view", names)

    async def test_enabled_features_add_their_tools_to_the_setup_message(self):
        config = Config(dry_run=True, notify=False, tasks_enabled=True, vision_enabled=True)
        session = gemini_live.GeminiLiveSession(config)
        with mock.patch.object(gemini_live.capabilities, "manifest", return_value=""), \
             mock.patch.object(gemini_live.capabilities, "live_state", return_value=""):
            setup = await session._setup_message()
        names = {d["name"] for d in setup["setup"]["tools"][0]["functionDeclarations"]}
        self.assertIn("task_submit", names)
        self.assertIn("camera_view", names)


# --- barge-in via serverContent.interrupted ----------------------------------

class BargeInTests(unittest.IsolatedAsyncioTestCase):
    """Architecturally simpler than realtime.py's offset-truncation barge-in:
    there is no audio-item/byte-offset tracking, just a flag to react to."""

    def setUp(self):
        patch_feedback_paths(self)
        self.session = gemini_live.GeminiLiveSession(Config(dry_run=True, notify=False))
        self.session.speaker.interrupt = mock.AsyncMock()
        self.session.speaker.write = mock.AsyncMock()
        self.socket = FakeSocket()
        self.session.ws = self.socket

    async def test_interrupted_flag_calls_speaker_interrupt(self):
        await self.session._on_server_content({"interrupted": True})
        self.session.speaker.interrupt.assert_awaited_once()

    async def test_a_false_interrupted_flag_does_not_call_interrupt(self):
        await self.session._on_server_content({"interrupted": False})
        self.session.speaker.interrupt.assert_not_awaited()

    async def test_content_with_no_interrupted_key_does_not_call_interrupt(self):
        await self.session._on_server_content({"modelTurn": {"parts": []}})
        self.session.speaker.interrupt.assert_not_awaited()

    async def test_interrupted_alongside_fresh_audio_still_drops_the_old_queue(self):
        pcm = base64.b64encode(b"\x00\x01").decode()
        await self.session._on_server_content({
            "interrupted": True,
            "modelTurn": {"parts": [{"inlineData": {"data": pcm}}]},
        })
        self.session.speaker.interrupt.assert_awaited_once()
        self.session.speaker.write.assert_awaited_once_with(b"\x00\x01")


# --- session-level dispatch, confirm/cancel gate, budget ---------------------

class GeminiSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = patch_feedback_paths(self)
        self.log = root / "session.log"
        self.config = Config(dry_run=True, notify=False)
        self.session = gemini_live.GeminiLiveSession(self.config)
        self.session.speaker.write = mock.AsyncMock()
        self.session.speaker.interrupt = mock.AsyncMock()
        self.socket = FakeSocket()
        self.session.ws = self.socket

    def hold_a_reboot(self):
        """Put a confirm-gated action into the pending slot, the real way."""
        result = self.session.executor.call("omarchy_cli", {"command": "reboot"})
        self.assertFalse(result.ok)
        self.assertIsNotNone(self.session.executor.pending)

    def call(self, identity, name, args):
        return {"id": identity, "name": name, "args": args}

    def only_response(self):
        replies = self.socket.events("toolResponse")
        self.assertEqual(len(replies), 1)
        return replies[0]["functionResponses"]

    async def test_a_single_call_gets_a_batched_reply_of_one(self):
        await self.session._on_tool_call({"functionCalls": [
            self.call("c1", "hypr_dispatch", {"lua": 'hl.dsp.focus({ workspace = "3" })'}),
        ]})
        [entry] = self.only_response()
        self.assertEqual(entry["id"], "c1")
        self.assertEqual(entry["name"], "hypr_dispatch")
        self.assertIn("dry-run", entry["response"]["result"])

    async def test_several_calls_in_one_toolCall_get_one_toolResponse(self):
        await self.session._on_tool_call({"functionCalls": [
            self.call("c1", "hypr_query", {"kind": "workspaces"}),
            self.call("c2", "hypr_query", {"kind": "clients"}),
        ]})
        entries = self.only_response()
        self.assertEqual([e["id"] for e in entries], ["c1", "c2"])

    async def test_an_empty_functionCalls_list_sends_nothing(self):
        await self.session._on_tool_call({"functionCalls": []})
        self.assertEqual(self.socket.sent, [])

    async def test_denied_actions_are_still_denied(self):
        await self.session._on_tool_call({"functionCalls": [
            self.call("c1", "run_shell", {"command": "sudo rm -rf /"}),
        ]})
        [entry] = self.only_response()
        self.assertTrue(entry["response"]["result"].startswith("ERROR:")
                        or "refused" in entry["response"]["result"])
        self.assertIsNone(self.session.executor.pending)

    async def test_same_batch_confirm_last_does_not_release_the_gate(self):
        self.session._user_turn_since_hold = True  # a new turn already happened
        await self.session._on_tool_call({"functionCalls": [
            self.call("hold", "omarchy_cli", {"command": "reboot"}),
            self.call("yes", "confirm_last", {"heard_phrase": "confirm"}),
        ]})
        self.assertIsNotNone(self.session.executor.pending)
        entries = {e["id"]: e["response"]["result"] for e in self.only_response()}
        self.assertIn("new user turn", entries["yes"])

    async def test_confirm_without_a_new_turn_after_the_hold_is_rejected(self):
        self.hold_a_reboot()
        self.assertFalse(self.session._user_turn_since_hold)
        await self.session._on_tool_call({"functionCalls": [
            self.call("yes", "confirm_last", {"heard_phrase": "confirm"}),
        ]})
        self.assertIsNotNone(self.session.executor.pending)
        [entry] = self.only_response()
        self.assertIn("new user turn", entry["response"]["result"])

    async def test_confirm_after_a_genuinely_new_turn_releases_the_hold(self):
        self.hold_a_reboot()
        await self.session._on_server_content({"turnComplete": True})
        self.assertTrue(self.session._user_turn_since_hold)
        await self.session._on_tool_call({"functionCalls": [
            self.call("yes", "confirm_last", {"heard_phrase": "confirm"}),
        ]})
        self.assertIsNone(self.session.executor.pending)

    async def test_wrong_phrase_does_not_release_the_gate(self):
        self.hold_a_reboot()
        self.session._user_turn_since_hold = True
        for phrase in ["they said yes", "the user confirmed", "ok", "do it", ""]:
            with self.subTest(phrase=phrase):
                output = await self.session._confirm(phrase)
                self.assertTrue(output.startswith("ERROR:"), output)
                self.assertIsNotNone(self.session.executor.pending)

    async def test_a_real_confirmation_phrase_releases_it(self):
        self.hold_a_reboot()
        output = await self.session._confirm("confirm")
        self.assertFalse(output.startswith("ERROR:"), output)
        self.assertIsNone(self.session.executor.pending)

    async def test_confirming_nothing_is_an_error(self):
        output = await self.session._confirm("confirm")
        self.assertTrue(output.startswith("ERROR:"), output)

    async def test_cancel_drops_the_held_action(self):
        self.hold_a_reboot()
        output = self.session._cancel()
        self.assertIn("Cancelled", output)
        self.assertIsNone(self.session.executor.pending)

    async def test_local_confirm_bypasses_phrase_matching_entirely(self):
        """The control-socket path (`omarchy-voice listen confirm`) trusts the
        caller and never looks at confirm_words — unlike `_confirm`."""
        self.hold_a_reboot()
        output = await self.session._local_confirm()
        self.assertFalse(output.startswith("ERROR:"), output)
        self.assertIsNone(self.session.executor.pending)

    async def test_local_cancel_bypasses_phrase_matching_entirely(self):
        self.hold_a_reboot()
        output = await self.session._local_cancel()
        self.assertIn("cancelled", output.lower())
        self.assertIsNone(self.session.executor.pending)

    async def test_local_confirm_with_nothing_pending_says_so(self):
        self.assertEqual(await self.session._local_confirm(), "nothing to confirm")

    async def test_local_cancel_with_nothing_pending_says_so(self):
        self.assertEqual(await self.session._local_cancel(), "nothing to cancel")

    async def test_tool_loop_stops_at_max_turns(self):
        self.session.config.max_turns = 2
        for i in range(4):
            await self.session._on_tool_call({"functionCalls": [
                self.call(f"c{i}", "hypr_query", {"kind": "workspaces"}),
            ]})
        self.assertEqual(self.session._tool_rounds, 4)
        self.assertIn("guard   stopped", self.log.read_text())

    async def test_a_new_user_turn_refills_the_budget(self):
        self.session._tool_rounds = 5
        await self.session._on_server_content({"turnComplete": True})
        self.assertEqual(self.session._tool_rounds, 0)

    async def test_typed_turn_is_injected_as_client_content(self):
        self.session._tool_rounds = 5
        result = await self.session._inject("switch to workspace four")
        self.assertEqual(result, "sent")
        [content] = self.socket.events("clientContent")
        self.assertEqual(content["turns"], [{"role": "user",
                                             "parts": [{"text": "switch to workspace four"}]}])
        self.assertTrue(content["turnComplete"])
        self.assertTrue(self.session._user_turn_since_hold)
        self.assertEqual(self.session._tool_rounds, 0)

    async def test_injecting_blank_text_sends_nothing(self):
        self.assertEqual(await self.session._inject("   "), "nothing to say")
        self.assertEqual(self.socket.sent, [])


# --- the microphone loop's echo gate ------------------------------------------

class MicrophoneGateTests(unittest.IsolatedAsyncioTestCase):
    """The half-duplex echo gate as gemini_live's `_mic_loop` actually applies
    it: mic frames dropped while `speaker.is_playing()` and `barge_in` is off,
    sent otherwise. Adapted from test_realtime.py's MicrophoneGateTests to
    16kHz capture and the `realtimeInput.audio` message shape.

    `_mic_loop`'s `finally` block calls `self._kill_mic()`, which calls a name
    `_kill_mic_process` that is not defined or imported anywhere in
    gemini_live.py (confirmed: `grep -rn _kill_mic_process src/` finds only
    the call site) — every real call to `_kill_mic()` raises `NameError`.
    `_kill_mic` is stubbed out below to isolate the echo-gate logic under test
    from that unrelated, pre-existing bug; see the final report for details.
    """

    # 100ms of 16kHz PCM16 mono, loud enough to clear frame_level's noise gate.
    FRAME = b"\x00\x20" * 1600

    def setUp(self):
        self.log = patch_feedback_paths(self) / "session.log"

    async def run_mic(self, frames, config=None, speaking=0.0):
        session = gemini_live.GeminiLiveSession(config or Config(dry_run=True, notify=False))
        session._kill_mic = mock.AsyncMock()  # sidestep the _kill_mic_process bug
        socket = FakeSocket()
        session.ws = socket
        session._active_event.set()
        if speaking:
            session.speaker._plays_until = time.monotonic() + speaking

        levels = []
        session.feedback.level = lambda mic, voice=0.0: levels.append((mic, voice))

        queue = list(frames)

        class Stdout:
            async def read(self, _n):
                while queue:
                    item = queue.pop(0)
                    if item == "quiet":
                        session.speaker._plays_until = 0.0
                        continue
                    return item
                session._stop.set()
                return b""

        class Proc:
            returncode = None
            stdout = Stdout()

        with mock.patch.object(gemini_live.asyncio, "create_subprocess_exec",
                               new=mock.AsyncMock(return_value=Proc())):
            await asyncio.wait_for(session._mic_loop(), timeout=5)
        appended = socket.events("realtimeInput")
        return session, appended, levels

    async def test_her_own_voice_never_reaches_the_wire(self):
        session, appended, _ = await self.run_mic([self.FRAME] * 4, speaking=5.0)
        self.assertEqual(appended, [])
        self.assertEqual(session._held_frames, 4)

    async def test_a_quiet_room_is_sent_normally(self):
        _, appended, _ = await self.run_mic([self.FRAME] * 4, speaking=0.0)
        self.assertEqual(len(appended), 4)

    async def test_the_microphone_reopens_when_she_stops(self):
        session, appended, _ = await self.run_mic(
            [self.FRAME, self.FRAME, "quiet", self.FRAME, self.FRAME], speaking=5.0)
        self.assertEqual(len(appended), 2)
        self.assertEqual(session._held_frames, 0)
        self.assertIn("mic     held 2 frames while speaking", self.log.read_text())

    async def test_sent_frames_use_the_realtime_input_audio_shape(self):
        _, appended, _ = await self.run_mic([self.FRAME], speaking=0.0)
        [frame] = appended
        audio = frame["audio"]
        self.assertEqual(base64.b64decode(audio["data"]), self.FRAME)
        self.assertEqual(audio["mimeType"], "audio/pcm;rate=16000")

    async def test_the_orb_does_not_twitch_on_her_own_voice(self):
        _, _, levels = await self.run_mic([self.FRAME] * 3, speaking=5.0)
        self.assertEqual([mic for mic, _ in levels if mic > 0.0], [])

    async def test_barge_in_hands_the_interruption_back(self):
        """With barge_in on, the mic stays open even while she is speaking —
        there is no leak to gate against (headphones, or echo-cancelled)."""
        config = Config(dry_run=True, notify=False, barge_in=True)
        session, appended, _ = await self.run_mic(
            [self.FRAME] * 4, config=config, speaking=5.0)
        self.assertEqual(len(appended), 4)
        self.assertEqual(session._held_frames, 0)


# --- check_ready ---------------------------------------------------------------

class CheckReadyTests(unittest.TestCase):
    """Same shape as realtime.check_ready: a list of human-readable problems,
    checked here in isolation from any real environment or session."""

    def setUp(self):
        self.config = Config()

    def healthy_env(self, **overrides):
        env = {"GEMINI_API_KEY": "test-key"}
        env.update(overrides)
        return env

    def test_missing_api_key_is_reported(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("shutil.which", return_value="/usr/bin/pw-record"), \
             mock.patch.object(gemini_live, "default_source", return_value="alsa_input.builtin"):
            problems = gemini_live.check_ready(self.config)
        self.assertTrue(any("GEMINI_API_KEY is not set" in p for p in problems), problems)

    def test_a_custom_key_env_name_is_reported_by_name(self):
        config = Config(gemini_api_key_env="MY_GEMINI_KEY")
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("shutil.which", return_value="/usr/bin/pw-record"), \
             mock.patch.object(gemini_live, "default_source", return_value="alsa_input.builtin"):
            problems = gemini_live.check_ready(config)
        self.assertTrue(any("MY_GEMINI_KEY is not set" in p for p in problems), problems)

    def test_missing_pipewire_tools_are_reported(self):
        with mock.patch.dict(os.environ, self.healthy_env(), clear=True), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch.object(gemini_live, "default_source", return_value="alsa_input.builtin"):
            problems = gemini_live.check_ready(self.config)
        self.assertTrue(any("pw-record is missing" in p for p in problems), problems)
        self.assertTrue(any("pw-cat is missing" in p for p in problems), problems)

    def test_no_input_device_is_reported(self):
        with mock.patch.dict(os.environ, self.healthy_env(), clear=True), \
             mock.patch("shutil.which", return_value="/usr/bin/x"), \
             mock.patch.object(gemini_live, "default_source", return_value=""):
            problems = gemini_live.check_ready(self.config)
        self.assertTrue(any("no audio input" in p for p in problems), problems)

    def test_a_loopback_input_is_reported(self):
        with mock.patch.dict(os.environ, self.healthy_env(), clear=True), \
             mock.patch("shutil.which", return_value="/usr/bin/x"), \
             mock.patch.object(gemini_live, "default_source",
                               return_value="alsa_output.x.monitor"):
            problems = gemini_live.check_ready(self.config)
        self.assertTrue(any("loopback" in p for p in problems), problems)

    def test_a_healthy_setup_reports_nothing(self):
        with mock.patch.dict(os.environ, self.healthy_env(), clear=True), \
             mock.patch("shutil.which", return_value="/usr/bin/x"), \
             mock.patch.object(gemini_live, "default_source", return_value="alsa_input.builtin"):
            problems = gemini_live.check_ready(self.config)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
