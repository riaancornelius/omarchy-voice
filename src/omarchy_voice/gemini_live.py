"""A second speech-to-speech engine: Gemini Live instead of OpenAI Realtime.

    pw-record ──▶ websocket ──▶ gemini-live ──▶ audio ──▶ pw-cat
                                     │
                                     ▼  toolCall
                               policy gate ──▶ denied / held
                                     │
                                     ▼
                        hyprctl · omarchy · wtype · uwsm-app

Same pipeline shape as realtime.py, different wire protocol. Two real
differences from OpenAI's Realtime API drove the design here:

  - Audio is asymmetric: 16kHz PCM16 in, 24kHz PCM16 out, rather than one
    shared sample rate for both directions.
  - Barge-in is server-decided. There is no truncate-by-audio-offset message
    to send; the server itself notices the interruption and reports it as
    `serverContent.interrupted`. The client's whole job is to react to that by
    dropping whatever `Speaker` still has queued.

Protocol details here (message field names, the setup handshake, the
function-declaration schema shape) come from Google's published Live API
reference and a handful of worked examples, not from a client SDK — no SDK
here either, matching every other network call in this codebase. Two things
are flagged inline as unverified rather than asserted as fact: the exact
manual/"off" VAD handshake, and `toolCallCancellation`. Both are handled
conservatively (logged, not acted on) until confirmed against a live session.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any

from . import capabilities
from .config import Config, ENV_FILE
from .feedback import Feedback
from .network import monitored_socket
from .persona import PERSONA
from .realtime import (
    ECHO_TAIL_SECONDS,
    FRAME_BYTES,
    GATE_TOOLS,
    REALTIME_PERSONA,
    RECONNECT_ATTEMPTS,
    RECONNECT_BASE_DELAY,
    RECONNECT_HEALTHY_SECONDS,
    RECONNECT_MAX_DELAY,
    Speaker,
    _run_until_done,
    _terminate,
    default_source,
    frame_level,
)
from .session import ControlServer
from .tools import TOOL_SCHEMAS, Executor, tools_for

GEMINI_WS_HOST = "wss://generativelanguage.googleapis.com"
GEMINI_WS_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

# Errors the server reports that mean "you were slightly late", the same idea
# as realtime.py's BENIGN_ERRORS — kept separate because Gemini's own error
# vocabulary has not been exercised enough yet to know if it overlaps.
BENIGN_ERRORS: set[str] = set()

SENSITIVITY = {
    "high": ("START_SENSITIVITY_HIGH", "END_SENSITIVITY_HIGH"),
    "low": ("START_SENSITIVITY_LOW", "END_SENSITIVITY_LOW"),
}


class GeminiUnavailable(RuntimeError):
    """Something the Gemini Live path needs is missing; the message says what."""


# --- tool schema conversion ---------------------------------------------------

def _schema_for_gemini(node: Any) -> Any:
    """OpenAI/JSON-Schema shapes, recursively, into Gemini's Schema object.

    Three adjustments, all load-bearing: `type` values are upper-cased
    (`"object"` -> `"OBJECT"`), `additionalProperties` is dropped since
    Gemini's function-declaration Schema does not support it, and empty-string
    `enum` members are dropped — confirmed against a live setup call, which
    rejected `reveal_window`'s `panel` enum with "cannot be empty" for its ""
    member. That member only ever meant "nothing to dismiss", equivalent to
    omitting an already-optional field, so dropping it changes nothing a
    well-behaved caller would notice. This targets the documented `parameters`
    field; confirmed against a live `setup` call (2026-09) that the resulting
    schema passes validation for the full tool set — a `parametersJsonSchema`
    passthrough may still exist as a simpler alternative, but this conversion
    is verified working, not just a guess.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "additionalProperties":
                continue
            if key == "type" and isinstance(value, str):
                out[key] = value.upper()
            elif key == "enum" and isinstance(value, list):
                out[key] = [item for item in value if item != ""]
            else:
                out[key] = _schema_for_gemini(value)
        return out
    if isinstance(node, list):
        return [_schema_for_gemini(item) for item in node]
    return node


def to_gemini_tools(schemas: list[dict] | None = None) -> list[dict]:
    """Executor tool schemas -> one Gemini Tool with all functionDeclarations."""
    declarations = []
    for schema in schemas if schemas is not None else TOOL_SCHEMAS:
        declarations.append({
            "name": schema["name"],
            "description": schema["description"],
            "parameters": _schema_for_gemini(schema["input_schema"]),
        })
    for gate in GATE_TOOLS:
        declarations.append({
            "name": gate["name"],
            "description": gate["description"],
            "parameters": _schema_for_gemini(gate["parameters"]),
        })
    return [{"functionDeclarations": declarations}]


# --- the session --------------------------------------------------------------

class GeminiLiveSession:
    def __init__(self, config: Config):
        self.config = config
        self.feedback = Feedback(config)
        self.executor = Executor(config, on_action=self._on_action)
        self.speaker = Speaker(config.gemini_output_sample_rate)
        self.active = False
        self.ws: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._send_lock = asyncio.Lock()
        self._active_event = asyncio.Event()
        self._stop = asyncio.Event()
        self._mic: asyncio.subprocess.Process | None = None
        self._held_frames = 0
        self._user_turn_since_hold = False
        self._tool_rounds = 0
        self._user_quit = False
        self._dropped = False
        self._exit_code = 0
        self._input_transcript = ""
        self._output_transcript = ""
        # Set from sessionResumptionUpdate events; not yet used to actually
        # resume a session on reconnect (see _serve()), just captured for now.
        self._resumption_handle: str | None = None
        # Connect only while this is set — while actively listening, or for
        # the duration of a typed `say` turn. An idle Gemini Live connection
        # gets closed by the server roughly every 2-3 minutes ("policy
        # violation"), confirmed live (2026-09); holding one open all day
        # regardless of use just churns reconnects for no benefit.
        self._want_connection = asyncio.Event()
        # Set once a connection's setup handshake has completed; _inject()
        # waits on this before sending a typed turn.
        self._connected = asyncio.Event()
        # Set on turnComplete; lets a typed turn know when it's safe to
        # close the connection it opened for itself.
        self._turn_done = asyncio.Event()
        # Fresh Event per connection attempt (see _serve()); sema for "this
        # particular connection should end", distinct from _stop (the whole
        # daemon shutting down).
        self._session_should_end: asyncio.Event | None = None

    # -- plumbing -------------------------------------------------------------
    def _on_action(self, name: str, description: str) -> None:
        self.feedback.state("acting", description)
        self.feedback.log(f"action  {description}")

    async def _send(self, payload: dict) -> None:
        if self.ws is None:
            return
        async with self._send_lock:
            try:
                await self.ws.send(json.dumps(payload))
            except Exception as exc:
                self.feedback.log(f"error   send failed: {type(exc).__name__}: {exc}")
                self._dropped = True
                self._exit_code = 1
                self._stop.set()

    # -- session configuration --------------------------------------------------
    async def _instructions(self) -> str:
        from .tasks import ROUTING
        from .vision import ROUTING as VISION_ROUTING
        manifest, live = await asyncio.gather(
            asyncio.to_thread(capabilities.manifest),
            asyncio.to_thread(capabilities.live_state),
        )
        style = self.config.gemini_voice_style.strip()
        return "\n\n".join([
            PERSONA, REALTIME_PERSONA,
            f"# Delivery\n\n{style}" if style else "",
            ROUTING if self.config.tasks_enabled else "",
            VISION_ROUTING if self.config.vision_enabled else "", manifest,
            "# The desktop right now\n\n" + live,
        ])

    def _activity_detection(self) -> dict:
        kind = (self.config.gemini_turn_detection or "high").strip().lower()
        if kind == "off":
            # Manual/no-VAD mode needs an explicit activityStart/activityEnd
            # handshake on the client side that has not been verified against
            # a live session yet. Falling back to "high" rather than sending
            # a half-implemented manual mode that would silently never end a
            # turn.
            self.feedback.log(
                "warn    gemini_turn_detection = \"off\" is not implemented yet; using \"high\"")
            kind = "high"
        start, end = SENSITIVITY.get(kind, SENSITIVITY["high"])
        return {
            "automaticActivityDetection": {
                "startOfSpeechSensitivity": start,
                "endOfSpeechSensitivity": end,
            },
        }

    async def _setup_message(self) -> dict:
        return {
            "setup": {
                "model": f"models/{self.config.gemini_model}",
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": self.config.gemini_voice},
                        },
                    },
                },
                "systemInstruction": {"parts": [{"text": await self._instructions()}]},
                "tools": to_gemini_tools(tools_for(self.config)),
                "realtimeInputConfig": self._activity_detection(),
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            },
        }

    # -- microphone -----------------------------------------------------------
    async def _mic_loop(self) -> None:
        """Capture only while active. Same echo-gate logic as realtime.py."""
        while not self._stop.is_set():
            waiter = asyncio.create_task(self._active_event.wait())
            stopper = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait(
                {waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if self._stop.is_set():
                return

            rate = self.config.gemini_input_sample_rate
            cmd = ["pw-record", "--rate", str(rate), "--channels", "1",
                   "--format", "s16", "--latency", "20ms"]
            if self.config.device:
                cmd += ["--target", self.config.device]
            cmd.append("-")
            try:
                self._mic = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL)
            except FileNotFoundError:
                self.feedback.log("error   pw-record is missing — install pipewire-audio")
                self._stop.set()
                self._exit_code = 1
                return

            self.feedback.log("mic     capturing")
            stdout = self._mic.stdout
            assert stdout is not None
            frame_bytes = int(rate * 0.1) * 2  # 100ms of 16-bit mono, this rate
            try:
                while self._active_event.is_set() and not self._stop.is_set():
                    chunk = await stdout.read(frame_bytes or FRAME_BYTES)
                    if not chunk:
                        if self._active_event.is_set() and not self._stop.is_set():
                            self.feedback.log("error   pw-record ended unexpectedly")
                            self._exit_code = 1
                            self._stop.set()
                        break
                    if not self.config.barge_in and self.speaker.is_playing(
                            ECHO_TAIL_SECONDS):
                        self._held_frames += 1
                        self.feedback.level(0.0, self.speaker.level_now())
                        continue
                    if self._held_frames:
                        self.feedback.log(
                            f"mic     held {self._held_frames} frames while speaking")
                        self._held_frames = 0
                    self.feedback.level(frame_level(chunk), self.speaker.level_now())
                    await self._send({
                        "realtimeInput": {
                            "audio": {
                                "data": base64.b64encode(chunk).decode(),
                                "mimeType": f"audio/pcm;rate={rate}",
                            },
                        },
                    })
            finally:
                await self._kill_mic()
                self.feedback.level(0.0)
                self.feedback.log("mic     stopped")

    async def _kill_mic(self) -> None:
        proc, self._mic = self._mic, None
        if proc is None or proc.returncode is not None:
            return
        await _terminate(proc)

    # -- control socket ---------------------------------------------------------
    def _control(self, command: str) -> str:
        if self.loop is None:
            return "not ready"
        verb, _, rest = command.partition(" ")
        if verb == "toggle":
            future = asyncio.run_coroutine_threadsafe(self._toggle(), self.loop)
        elif verb in ("start", "stop"):
            future = asyncio.run_coroutine_threadsafe(
                self._set_active(verb == "start"), self.loop)
        elif verb == "say":
            future = asyncio.run_coroutine_threadsafe(self._inject(rest), self.loop)
        elif verb == "confirm":
            future = asyncio.run_coroutine_threadsafe(self._local_confirm(), self.loop)
        elif verb == "cancel":
            future = asyncio.run_coroutine_threadsafe(self._local_cancel(), self.loop)
        elif verb == "quit":
            self._user_quit = True
            self.loop.call_soon_threadsafe(self._stop.set)
            # Wakes _lifecycle() if it's idling with no connection open.
            self.loop.call_soon_threadsafe(self._want_connection.set)
            return "stopping"
        else:
            return f"unknown command {verb!r}"
        try:
            return future.result(timeout=10)
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"

    async def _toggle(self) -> str:
        return await self._set_active(not self.active)

    async def _set_active(self, active: bool) -> str:
        self.active = active
        if active:
            self._active_event.set()
            self._want_connection.set()
        else:
            self._active_event.clear()
            await self._kill_mic()
            await self.speaker.interrupt()
            await asyncio.to_thread(self.executor.vision.stop_owned)
            self._want_connection.clear()
            if self._session_should_end is not None:
                self._session_should_end.set()
        self.feedback.state("listening" if active else "idle")
        self.feedback.notify("Listening" if active else "Sleeping")
        self.feedback.log(f"gate    {'listening' if active else 'muted'}")
        return "listening" if active else "idle"

    async def _inject(self, text: str) -> str:
        """`omarchy-voice listen say ...` — a typed turn, mic untouched.

        The `clientContent`/`turns`/`parts`/`text` shape is confirmed working
        against a live session (2026-09). If nothing is connected yet, this
        opens a connection just for this turn and closes it again once the
        reply finishes — unless listening got turned on in the meantime.
        """
        text = text.strip()
        if not text:
            return "nothing to say"
        self._user_turn_since_hold = True
        self._tool_rounds = 0
        self.feedback.log(f"typed   {text!r}")
        opened_for_this = not self._want_connection.is_set()
        self._want_connection.set()
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=15)
        except asyncio.TimeoutError:
            if opened_for_this:
                self._want_connection.clear()
            return "error: could not connect to gemini live"
        self._turn_done.clear()
        await self._send({
            "clientContent": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turnComplete": True,
            },
        })
        if opened_for_this:
            asyncio.create_task(self._disconnect_after_typed_turn())
        return "sent"

    async def _disconnect_after_typed_turn(self) -> None:
        """Close a connection _inject() opened for itself, once the reply
        finishes — unless listening got turned on while it was in flight."""
        try:
            await asyncio.wait_for(self._turn_done.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass
        if not self.active:
            self._want_connection.clear()
            if self._session_should_end is not None:
                self._session_should_end.set()

    async def _local_confirm(self) -> str:
        """Keybind / CLI confirm — does not trust the model."""
        if not self.executor.pending:
            return "nothing to confirm"
        held = self.executor.describe(*self.executor.pending)
        self.feedback.log(f"confirm local release: {held}")
        result = await asyncio.to_thread(self.executor.run_pending)
        self._settle()
        return result.as_tool_result()

    async def _local_cancel(self) -> str:
        held = self.executor.drop_pending()
        if held is None:
            return "nothing to cancel"
        self.feedback.log(f"cancel  {held}")
        self._settle()
        return f"cancelled: {held}"

    # -- events -----------------------------------------------------------------
    async def _on_event(self, event: dict) -> None:
        if "setupComplete" in event:
            self.feedback.log("start   gemini live session")
            return

        content = event.get("serverContent")
        if content is not None:
            await self._on_server_content(content)
            return

        tool_call = event.get("toolCall")
        if tool_call is not None:
            await self._on_tool_call(tool_call)
            return

        # Documented as a way for the server to withdraw a pending tool call;
        # not exercised against a live session yet, so this only logs. If it
        # turns out to matter, the fix is to drop the matching id from
        # whatever is queued in `executor.pending` before it resolves.
        cancellation = event.get("toolCallCancellation")
        if cancellation is not None:
            self.feedback.log(f"note    toolCallCancellation: {cancellation}")
            return

        # Confirmed on a live session (2026-09): {"newHandle": <uuid>,
        # "resumable": true}, arriving dozens of times per single turn — near
        # continuous, not periodic-while-idle as first assumed. Stored rather
        # than logged (logging every one would flood the session log); not
        # yet used to actually resume a session on reconnect, but the shape is
        # now real, not a guess, for whoever wires that up.
        resumption = event.get("sessionResumptionUpdate")
        if resumption is not None:
            if resumption.get("resumable"):
                self._resumption_handle = resumption.get("newHandle")
            return

        error = event.get("error")
        if error is not None:
            code = error.get("code") if isinstance(error, dict) else None
            message = str(error.get("message") if isinstance(error, dict) else error)
            if code in BENIGN_ERRORS:
                self.feedback.log(f"note    {code}: {message}")
                return
            self.feedback.log(f"error   {message}")
            self.feedback.state("error", message)
            self.feedback.notify("Voice error", message, urgency="normal")

    async def _on_server_content(self, content: dict) -> None:
        # `content` has also been observed carrying a `voiceActivity` key on a
        # live session, shape not yet examined. Falls through unhandled below
        # — safe (nothing here breaks on an unknown key) but worth knowing
        # it's a real, undocumented-here field before assuming this handler
        # covers everything serverContent can carry.
        if content.get("interrupted"):
            # The server decided this was a barge-in. There is no offset to
            # truncate at — dropping what Speaker still has queued is the
            # entire client-side job.
            await self.speaker.interrupt()

        # Deltas carry their own inter-word spacing — stripping each one
        # individually (rather than only the final accumulated string) ate
        # that spacing and produced log lines like "couldnotconnectto",
        # confirmed against a live session.
        #
        # `heard` is not a trustworthy transcript, confirmed live: a user who
        # said "focus the other monitor" in English (correctly understood and
        # acted on — the right monitor got focused) was logged here as
        # "Cette fois que tu as demandé un autre." — fabricated French,
        # unrelated to the actual audio. `inputTranscription` is itself
        # model-generated commentary alongside the real audio understanding,
        # not a dedicated ASR pass, and can hallucinate independently of
        # whether the assistant actually understood you correctly. Treat
        # `heard` log lines as a rough diagnostic hint, never as evidence of
        # what was actually said or whether the assistant understood it.
        transcription = content.get("inputTranscription") or {}
        heard = transcription.get("text") or ""
        if heard:
            self._input_transcript += heard

        out_transcription = content.get("outputTranscription") or {}
        said_delta = out_transcription.get("text") or ""
        if said_delta:
            self._output_transcript += said_delta

        model_turn = content.get("modelTurn") or {}
        for part in model_turn.get("parts", []):
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                try:
                    pcm = base64.b64decode(inline["data"])
                except (ValueError, TypeError) as exc:
                    self.feedback.log(f"error   bad audio part: {exc}")
                    continue
                await self.speaker.write(pcm)

        if content.get("turnComplete"):
            heard_final = self._input_transcript.strip()
            said_final = self._output_transcript.strip()
            if heard_final:
                self.feedback.log(f"heard   {heard_final!r}")
            if said_final:
                self.feedback.log(f"reply   {said_final!r}")
                self.feedback.notify(said_final)
            self._input_transcript = ""
            self._output_transcript = ""
            # A fresh turn boundary is the closest confirmed signal to "the
            # user spoke again" available on this protocol — used the same
            # way realtime.py resets its budget on speech_started.
            self._tool_rounds = 0
            self._user_turn_since_hold = True
            self._settle()
            self._turn_done.set()

    async def _on_tool_call(self, tool_call: dict) -> None:
        calls = tool_call.get("functionCalls") or []
        if not calls:
            return
        responses = []
        created_hold = False
        for call in calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            if name == "confirm_last":
                if created_hold or not self._user_turn_since_hold:
                    output = (
                        "ERROR: confirmation must come from a new user turn after "
                        "the action was held. Do not call confirm_last in the same "
                        "response as the gated tool."
                    )
                    self.feedback.log("reject  same-batch or no-new-turn confirm_last")
                else:
                    output = await self._confirm(str(args.get("heard_phrase", "")))
            elif name == "cancel_last":
                output = self._cancel()
            else:
                output = await asyncio.to_thread(self.executor.call, name, args)
                output = output.as_tool_result()
                if self.executor.pending:
                    created_hold = True
                    self._user_turn_since_hold = False
            responses.append({
                "id": call.get("id", ""),
                "name": name,
                "response": {"result": output},
            })

        await self._send({"toolResponse": {"functionResponses": responses}})

        self._tool_rounds += 1
        if self._tool_rounds >= self.config.max_turns:
            self.feedback.log(
                f"guard   stopped after {self._tool_rounds} tool rounds "
                f"with no new user turn (max_turns={self.config.max_turns})")
            self.feedback.notify(
                "OMA stopped", "Too many steps without a new instruction.")
        self._settle()

    def _settle(self) -> None:
        if self.executor.pending:
            held = self.executor.describe(*self.executor.pending)
            self.feedback.state("confirm", held)
            self.feedback.notify("Waiting for confirmation", held, urgency="normal")
        else:
            self.feedback.state("listening" if self.active else "idle")

    async def _confirm(self, heard_phrase: str) -> str:
        from .session import _matches
        if not self.executor.pending:
            return "ERROR: nothing is waiting for confirmation. Do not call this again."
        held = self.executor.describe(*self.executor.pending)
        if not _matches(heard_phrase, self.config.confirm_words, allow_negation=False):
            self.feedback.log(f"reject  {heard_phrase!r} is not a confirmation of: {held}")
            phrases = ", ".join(f'"{w}"' for w in self.config.confirm_words)
            return (f"ERROR: {heard_phrase!r} is not a confirmation phrase, so {held} is "
                    f"still held. Ask the user to say one of: {phrases}.")
        self.feedback.log(f"confirm {heard_phrase!r} released: {held}")
        result = await asyncio.to_thread(self.executor.run_pending)
        self._settle()
        return result.as_tool_result()

    def _cancel(self) -> str:
        held = self.executor.drop_pending()
        if held is None:
            return "Nothing was being held."
        self.feedback.log(f"cancel  {held}")
        self._settle()
        return f"Cancelled: {held}. It was not run."

    # -- main loop ----------------------------------------------------------
    async def run(self) -> int:
        self.loop = asyncio.get_running_loop()
        key = os.environ.get(self.config.gemini_api_key_env, "")
        if not key:
            raise GeminiUnavailable(
                f"{self.config.gemini_api_key_env} is not set — "
                "the gemini_live engine needs a Gemini API key")

        url = f"{GEMINI_WS_HOST}{GEMINI_WS_PATH}?key={key}"

        control = ControlServer(self._control)
        control.start()
        self.feedback.state("idle")
        self.feedback.log(f"start   engine=gemini_live model={self.config.gemini_model} "
                          f"voice={self.config.gemini_voice} "
                          f"dry_run={self.config.dry_run} (connects on demand)")
        self.feedback.log("gate    muted — press SUPER + SHIFT + V to start listening")

        try:
            await self._lifecycle(url)
        except GeminiUnavailable:
            raise
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.feedback.state("error", str(exc))
            self.feedback.log(f"error   {type(exc).__name__}: {exc}")
            print(f"gemini live session failed: {type(exc).__name__}: {exc}")
            return 1
        finally:
            self._stop.set()
            await self._kill_mic()
            await self.speaker.close()
            await asyncio.to_thread(control.stop)
            self.feedback.state("idle")
            self.ws = None
        return 0 if self._user_quit else self._exit_code

    async def _lifecycle(self, url: str) -> None:
        """Connect only while wanted: actively listening, or a typed `say`
        turn in flight. An idle Gemini Live connection gets closed by the
        server roughly every 2-3 minutes ("policy violation"), confirmed
        live (2026-09) — holding one open all day regardless of use just
        churns reconnects (and, apparently, 409s on the AI Studio dashboard)
        for no benefit.
        """
        while not self._user_quit:
            await self._want_connection.wait()
            if self._user_quit:
                return
            await self._serve(url)

    async def _serve(self, url: str) -> None:
        """Hold a session open for as long as it's wanted, rebuilding it if
        the socket dies unexpectedly. Returns (not as a failure) as soon as
        _want_connection is cleared. Otherwise the same reconnect-with-backoff
        shape as realtime.py's _serve — see that file's long comment for why
        an unexpected drop must not end the run.
        """
        delay = RECONNECT_BASE_DELAY
        attempt = 0
        while not self._user_quit and self._want_connection.is_set():
            self._dropped = False
            self._session_should_end = asyncio.Event()
            connected_at = time.monotonic()
            try:
                await self._open_one(url)
            except (GeminiUnavailable, asyncio.CancelledError):
                raise
            except Exception as exc:
                self._dropped = True
                self.feedback.log(f"error   {type(exc).__name__}: {exc}")

            if time.monotonic() - connected_at >= RECONNECT_HEALTHY_SECONDS:
                attempt = 0
                delay = RECONNECT_BASE_DELAY

            if self._user_quit or not self._dropped or not self._want_connection.is_set():
                return
            attempt += 1
            if attempt > RECONNECT_ATTEMPTS:
                self.feedback.log(f"stop    gave up after {RECONNECT_ATTEMPTS} reconnects")
                self.feedback.notify("Voice control stopped",
                                     "Lost the connection and could not get it back.",
                                     urgency="normal")
                self._exit_code = 1
                return
            was_listening, self.active = self.active, False
            self._active_event.clear()
            self.feedback.state("error", f"reconnecting ({attempt}/{RECONNECT_ATTEMPTS})")
            self.feedback.log(f"retry   reconnecting in {delay:.0f}s "
                              f"({attempt}/{RECONNECT_ATTEMPTS})")
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)
            if was_listening:
                self.active = True
                self._active_event.set()
                self.feedback.state("listening")
            else:
                self.feedback.state("idle")

    async def _open_one(self, url: str) -> None:
        mic_task: asyncio.Task | None = None
        try:
            async with monitored_socket(_open_socket(url), self.config,
                                        self.feedback, "gemini_live", endpoint=url) as ws:
                self.ws = ws
                await self._send(await self._setup_message())
                self._connected.set()
                mic_task = asyncio.create_task(self._mic_loop())
                stopper = asyncio.create_task(self._stop.wait())
                ender = asyncio.create_task(self._session_should_end.wait())
                reader = asyncio.create_task(self._read(ws))
                done, pending = await asyncio.wait(
                    {stopper, ender, reader}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                if reader in done and not reader.cancelled():
                    reader.result()
                    if not self._user_quit and ender not in done:
                        self._dropped = True
                        self.feedback.log("stop    the server closed the connection")
        finally:
            self._connected.clear()
            if mic_task is not None:
                mic_task.cancel()
            await self._kill_mic()
            self.ws = None
            await asyncio.to_thread(self.executor.vision.stop_owned)

    async def _read(self, ws) -> None:
        async for raw in ws:
            if self._stop.is_set():
                return
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if self.config.verbose:
                print(f"  << {list(event.keys())}", flush=True)
            try:
                await self._on_event(event)
            except Exception as exc:
                self.feedback.log(f"error   event {list(event.keys())}: {type(exc).__name__}: {exc}")


# --- helpers -------------------------------------------------------------------

def _open_socket(url: str):
    """Return an async context manager for the websocket.

    Same version-compatibility shim as realtime.py's _open_socket, minus the
    auth header — Gemini's API key rides in the URL's query string instead.
    """
    try:
        from websockets.asyncio.client import connect
    except ImportError:
        try:
            from websockets.legacy.client import connect as legacy_connect  # type: ignore
        except ImportError as exc:
            raise GeminiUnavailable(
                "python-websockets is not installed — "
                "run: sudo pacman -S python-websockets") from exc
        return legacy_connect(url, max_size=None, ping_interval=20, ping_timeout=20)
    return connect(url, max_size=None, ping_interval=20, ping_timeout=20)


def check_ready(config: Config) -> list[str]:
    """Everything standing between here and a working gemini_live session."""
    problems = []
    try:
        import websockets  # noqa: F401
    except ImportError:
        problems.append("python-websockets is not installed (sudo pacman -S python-websockets)")
    if not os.environ.get(config.gemini_api_key_env):
        problems.append(f"{config.gemini_api_key_env} is not set")
    import shutil
    for tool, package in (("pw-record", "pipewire-audio"), ("pw-cat", "pipewire-audio")):
        if not shutil.which(tool):
            problems.append(f"{tool} is missing (install {package})")
    source = default_source()
    if not source:
        problems.append("PipeWire reports no audio input — is a microphone plugged in?")
    elif source.endswith(".monitor"):
        problems.append(
            f"the default input is {source}, which is a loopback of speaker "
            "output, not a microphone — plug one in, or set `device` in config")
    return problems


def run(config: Config) -> int:
    """Entry point used by `omarchy-voice run` when engine = "gemini_live"."""
    problems = check_ready(config)
    soft = ("audio input", "loopback")
    hard = [p for p in problems if not any(s in p for s in soft)]
    unconfigured = [p for p in hard if config.gemini_api_key_env in p]
    if unconfigured:
        note = f"{config.gemini_api_key_env} is not set — put it in {ENV_FILE}"
        print(note)
        print("then: systemctl --user restart omarchy-voice")
        Feedback(config).state("unconfigured", note)
        return 0
    if hard:
        for problem in hard:
            print(f"cannot start gemini_live engine: {problem}")
        return 1
    for problem in problems:
        print(f"warning: {problem}")
    try:
        return _run_until_done(GeminiLiveSession(config))
    except GeminiUnavailable as exc:
        print(f"cannot start gemini_live engine: {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
