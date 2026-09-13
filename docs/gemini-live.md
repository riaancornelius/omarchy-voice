# Gemini Live backend

Gemini Live is a third speech-to-speech engine, alongside Realtime and Live. It
talks to Google's `BidiGenerateContent` API instead of OpenAI's Realtime API,
using the same pipeline shape (microphone in, tool calls to the shared executor
and policy gate, audio back out) with a different wire protocol.

The reason to reach for it is cost: `gemini-3.1-flash-live-preview` is priced
at roughly $3/$12 per 1M audio input/output tokens, against $32/$64 for
`gpt-realtime-2.1`, the default. That is a genuine like-for-like swap of the
always-on speech-to-speech engine, not just a cheaper model name — same shape
of session, same tool-calling loop, same policy gate. It is not a drop-in
guarantee of equal quality; treat it as a cost/quality trade to evaluate for
your own use, not an assumed upgrade.

## Select the engine

Stop the existing daemon before running another one in the foreground:

```sh
omarchy-voice listen quit
omarchy-voice run --engine gemini_live
```

To choose it persistently, edit `~/.config/omarchy-voice/config.toml` and
restart the idle user service:

```toml
[openai]
engine = "gemini_live"
```

`engine` lives under `[openai]` regardless of which engine you pick — see
[Live setup](live.md) for the same convention with `engine = "live"`.
Gemini-specific settings (model, voice, sample rates, turn detection) go in
their own `[gemini]` section, described below. `omarchy-voice doctor` reports
the configured engine and, when it is `gemini_live`, a dedicated "gemini"
section with the API key check, model/voice, and configured sample rates.

## Required API key

Gemini Live needs a `GEMINI_API_KEY` (the env var name itself is
configurable via `gemini_api_key_env`, see below). Add it to
`~/.config/omarchy-voice/env`, the same file used for `OPENAI_API_KEY` and
loaded the same way — merged into the process environment at startup, with a
real environment variable always taking precedence, and a warning if the file
is readable by group or other:

```sh
GEMINI_API_KEY=your-api-key
```

Keep that file private (`chmod 600 ~/.config/omarchy-voice/env`). If the key
is missing, `omarchy-voice run --engine gemini_live` prints a note telling you
where to put it and where to restart, rather than starting muted with no way
to connect.

## Configuration

See `[gemini]` in the [configuration example](../share/config.example.toml)
for the full section; the fields and their defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| `gemini_model` | `gemini-3.1-flash-live-preview` | Model used for the live session |
| `gemini_api_key_env` | `GEMINI_API_KEY` | Name of the environment variable holding the API key |
| `gemini_voice` | `Puck` | Prebuilt voice name sent in `speechConfig.voiceConfig.prebuiltVoiceConfig` |
| `gemini_input_sample_rate` | `16000` | Microphone capture rate (Hz), sent to `pw-record` |
| `gemini_output_sample_rate` | `24000` | Playback rate (Hz) for audio parts the server sends back |
| `gemini_turn_detection` | `high` | `"high"` or `"low"` per the Live API's Start/EndSensitivity enums; `"off"` is accepted but not yet implemented — see limitations below |
| `gemini_voice_style` | `""` | Free-text delivery direction appended to the system instruction, e.g. `"Speak like a calm, dry-witted radio announcer."`. `gemini_live` only — `realtime.py`/`live.py` have no equivalent field |

Fields are prefixed the same way as `[realtime]`, `[live]`, `[vision]`,
`[tasks]`, and `[network]`: a `[gemini]` section's keys get a `gemini_`
prefix when the loader reads `config.toml`, so `model = "..."` under
`[gemini]` becomes `gemini_model`.

## How it differs from the OpenAI engines

**Asymmetric audio.** Realtime and Live each use one shared sample rate for
both directions (`realtime_sample_rate` / `live_sample_rate`). Gemini Live is
asymmetric: 16kHz PCM16 microphone input, 24kHz PCM16 speaker output. That is
why it gets its own pair of fields (`gemini_input_sample_rate`,
`gemini_output_sample_rate`) instead of reusing a single-rate setting.

**Server-decided barge-in.** `realtime.py` truncates playback by sending an
offset-based `conversation.item.truncate` message when the user interrupts —
the client has to know how much of the queued audio was actually heard.
Gemini Live's server instead decides the interruption itself and reports it
as `serverContent.interrupted: true`; the client's entire job on receiving
that is to drop whatever the local `Speaker` still has queued. There is no
offset to compute or truncate message to send.

**Same tool-calling shape, different wire format.** Executor tool schemas are
converted to Gemini's function-declaration `Schema` object at setup time
(uppercasing JSON-Schema `type` values, dropping `additionalProperties`,
which Gemini's schema does not support). The confirm/cancel/policy-gate logic
itself — held actions, `confirm_last`/`cancel_last`, the `max_turns` guard —
is unchanged from the other engines.

**Voice style comes from instructions, not a parameter — and one relevant
feature doesn't apply to this model.** There is no structured style/emotion
parameter for prebuilt voices; the model follows plain-language delivery
direction placed in `systemInstruction`, which is what `gemini_voice_style`
(see below) feeds into. The Live API also has a dedicated
`enable_affective_dialog` setup flag that adapts tone to match the user's own
input automatically, with no instruction text needed — checked and ruled
out: Google's own docs state it is **not supported on
`gemini-3.1-flash-live-preview`**, the model configured here, only on
`gemini-2.5-flash-live-preview`. Not implemented for that reason; revisit if
this engine ever moves to a 2.5-series model.

## Verified against a live session (2026-09)

A real `setup` handshake against `generativelanguage.googleapis.com` with a
valid `GEMINI_API_KEY` confirmed the connection URL, the full `setup` message
shape (model, `generationConfig`, `systemInstruction`, `tools`,
`realtimeInputConfig`), and the tool-schema conversion — including one real
bug the probe caught and fixed: `reveal_window`'s `panel` parameter had an
empty-string `enum` member (meaning "nothing to dismiss"), which Gemini's
schema validator rejects outright with "cannot be empty". `_schema_for_gemini`
now strips empty-string enum members; since `panel` was already optional,
this changes nothing a well-behaved caller would notice. The handshake got
as far as actually starting a session, blocked only by the test account's
billing state (depleted prepayment credits) — not by anything in this
engine's code.

A second, deeper probe (billing since activated) drove a full round trip
through the real `GeminiLiveSession` code — not a hand-rolled script — using
`omarchy-voice listen say`'s injection path:

- **Typed injection (`clientContent`) is confirmed working.** An earlier
  attempt appeared to fail with "Request contains an invalid argument", but
  that was a race in the *test harness* (it sent the injected turn before
  confirming `setupComplete`, so the client's own message briefly overtook its
  own `setup`) — not a protocol bug. Once the probe waited for `setupComplete`
  first, `clientContent` worked exactly as documented.

- **The full tool-calling round trip is confirmed working, end to end.** The
  injected turn ("call `hypr_query`, then report the active window") produced
  a real `toolCall`, dispatched through the real `Executor`. It reported "no
  desktop service" in that run — an artifact of the probe sandboxing
  `XDG_RUNTIME_DIR` to avoid touching real state, which also broke
  `hyprctl`'s own socket discovery, not a real Hyprland integration problem
  (confirmed separately: `hyprctl` works fine outside the sandbox). The model
  correctly reported the failure it was actually given rather than inventing
  a window name, which is the behavior that matters here.

- **A real bug was found and fixed in transcript logging.** `_on_server_content`
  was `.strip()`-ing each incoming `inputTranscription`/`outputTranscription`
  delta before concatenating it, which ate the inter-word spacing Gemini
  actually sends between chunks — producing log lines like `reply
  'I couldnotconnectto thedesktopserviceto checkthe activewindow.'`. Fixed to
  strip only the final accumulated string, at `turnComplete`. Re-verified with
  a known sentence: logged and spoken back with correct spacing.

- **`sessionResumptionUpdate` is a real message type, previously unknown to
  this code.** Confirmed shape: `{"newHandle": <uuid>, "resumable": true}`.
  It is *not* occasional — it arrived dozens of times within a single turn,
  effectively continuously, not periodically-while-idle as first guessed.
  Logging every occurrence would have flooded the session log, so it is now
  stored on the session (`_resumption_handle`) rather than logged. Not yet
  used to actually resume a session on reconnect — a real enhancement
  opportunity for `_serve()`/`_open_one()`, whenever someone wants it.

### A real microphone/speaker session, on a live Hyprland desktop (2026-09)

With billing active, the daemon was run for real — `omarchy-voice run --engine
gemini_live`, real microphone, real speakers, a real Hyprland session, no
sandboxing — and driven with actual spoken commands:

- **A real, consequential tool call executed correctly.** Asked "which
  monitor is focused?", it answered correctly from its desktop-state snapshot
  with no tool call needed. Told to focus the other monitor, it called
  `hl.dsp.focus({ direction = "r" })` for real, and Hyprland actually moved
  focus. First real (not scripted, not text-injected) tool dispatch this
  engine has done.

- **The echo gate held up in a real conversation.** `mic held 497 frames
  while speaking` / `mic held 198 frames while speaking` appeared in the log
  exactly when the assistant's own reply was playing back, confirming the
  microphone is genuinely gated shut during playback (the default,
  `barge_in = false` behavior) rather than just passing unit tests against a
  fake clock.

- **A genuine reconnect happened mid-session, unprompted, and recovered
  cleanly.** After roughly 4 minutes connected, the socket closed with
  `close_code: 1008` (Policy Violation — most likely a session-duration cap
  on Google's side, not investigated further). `_serve()`'s reconnect logic
  fired, reconnected in 2 seconds, and correctly resumed in the same (muted)
  state the session was in before the drop — the "come back the way we left"
  logic in `_serve()` held up against a real, not simulated, disconnect.

- **`inputTranscription` hallucinated outright — while the assistant's actual
  understanding was correct.** Asked (in English) to "focus the other
  monitor" — correctly understood and acted on — the `heard` log line showed
  fabricated French text unrelated to what was said. The transcription is
  itself model-generated commentary alongside the real audio understanding,
  not a dedicated ASR pass, and can be wrong independently of whether the
  assistant actually understood you. Treat `heard` log lines as a rough
  diagnostic hint only, never as evidence of what was said or of whether the
  assistant understood correctly.

- **A new, unexamined field showed up**: `serverContent.voiceActivity`. Not
  investigated or handled — falls through harmlessly, but it's a real field
  this code doesn't yet know the shape of.

### Barge-in (2026-09) — the interrupt mechanism works; the risk it warns about is real

A follow-up session ran with `barge_in = true` (config override, no
microphone/speaker isolation — same analog device for both, no headphones,
`echo_risk()` had already flagged this before starting). Two things
confirmed:

- **The interrupt mechanism itself works correctly.** Replies were repeatedly
  cut off mid-word ("What's", "September") exactly when new audio was
  detected, confirming `serverContent.interrupted` → `speaker.interrupt()`
  fires and genuinely stops playback immediately, not just in the unit-test
  fakes.

- **Without echo isolation, this becomes a self-sustaining loop, with no
  automatic circuit breaker.** The assistant's own voice leaked back into the
  mic, was heard as new input, interrupted its own reply mid-sentence, and
  produced an increasingly disconnected reply to its own echoed fragment —
  six times in a row, each one a fresh `turnComplete` cycle. Nothing in this
  engine stopped it automatically: `max_turns` only bounds tool-call rounds
  *within* one turn, and each echo-triggered exchange looked like a brand new
  legitimate turn, so that guard never engaged. The loop only ended because a
  human muted it manually (`omarchy-voice listen quit`) — left alone, it
  would have kept going (and kept costing API usage) indefinitely.

  This is not a `gemini_live`-specific bug — `echo_risk()` in `realtime.py`
  describes the identical failure mode for the OpenAI engine under the same
  conditions, and is engine-agnostic. What's new here is empirical
  confirmation the loop is real and has no built-in stop condition on this
  engine, plus a related gap worth knowing: `echo_risk()` is only ever
  surfaced by `omarchy-voice doctor` — it is not printed at `run()` startup
  on *either* engine, so skipping `doctor` means no warning until you're
  already in the loop. Fixing that is a shared `realtime.py`/`gemini_live.py`
  concern, not attempted here since it wasn't the point of this test — worth
  a deliberate follow-up if this trips someone up in practice.

### Voice selection (2026-09) — confirmed working for this setup

A developer forum report specific to `gemini-3.1-flash-live-preview` claimed
`speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName` sometimes has no
effect — the session keeps using a fixed voice regardless of what's
configured — reproduced there via both ephemeral tokens and a raw WebSocket
setup message. Tested directly against our own setup (plain API key, the
same setup-message shape `_setup_message()` sends): two back-to-back
sessions, `gemini_voice = "Puck"` then `gemini_voice = "Charon"`, each
speaking a distinct test phrase. A human listened to both live and confirmed
they were audibly different voices. Whatever the forum report's issue is
(ephemeral-token-specific, since fixed, or narrower than it first appeared),
it does not reproduce here — `gemini_voice` is confirmed working for a plain
API-key setup like this one.

There is no single authoritative list of valid `voiceName` values in Google's
docs — the capabilities guide points to
[AI Studio's live playground](https://aistudio.google.com/app/live) to
actually listen to the current set rather than publishing one canonical list.
Names confirmed working here: `Puck` (the default), `Charon`. Others reported
elsewhere (unverified against this codebase): `Kore`, `Leda`, `Aoede`,
`Zephyr`, `Fenrir`.

One related, unconfigurable behavior worth knowing: native-audio Live models
auto-detect and switch language based on input across 97 supported
languages rather than taking an explicit locale setting — there is no
`gemini_language`-style field to pin it, unlike voice.

Still unverified: `toolCallCancellation` and manual/off VAD.

## Known limitations / unverified

These are called out in code comments in `gemini_live.py` rather than
papered over:

- **`gemini_turn_detection = "off"` is not implemented.** Manual/no-VAD mode
  would need an explicit `activityStart`/`activityEnd` handshake on the
  client side, and that handshake has not been verified against a live
  session. Setting it logs a warning — `gemini_turn_detection = "off" is not
  implemented yet; using "high"` — and falls back to `"high"` sensitivity
  rather than sending a half-implemented manual mode that could silently
  never end a turn.

- **`toolCallCancellation` messages are logged but not acted on.** This is
  documented as a way for the server to withdraw a pending tool call, but it
  has not been exercised against a live session, so the client only logs a
  note (`toolCallCancellation: ...`). If it turns out to matter in practice,
  the fix is to drop the matching call id from whatever is queued in
  `executor.pending` before it resolves.

One more worth noting: the tool-schema conversion targets the documented
`parameters` field on a function declaration, and that field is now confirmed
to accept the converted schema (see above). A newer `parametersJsonSchema`
passthrough may still exist on the Live API as a simpler alternative, but
this conversion is verified working, not a guess needing a live check.

## Verify setup

`omarchy-voice doctor` checks for `python-websockets`, the `GEMINI_API_KEY`
environment variable (or whatever `gemini_api_key_env` names), `pw-record`/
`pw-cat`, and a usable audio input device, and prints the configured model,
voice, and sample rates.

`tests/test_gemini_live.py` covers the offline, fake-socket surface (tool
conversion, activity detection, barge-in, tool dispatch, the echo gate,
`check_ready`) the same way `tests/test_realtime.py` does for the OpenAI
engine — no API key or network access needed to run it. There is no
`tools/check_live.py`-style paid protocol probe script yet; the live
round-trip verification in the section above was done ad hoc with a real key
and billing enabled, not via a committed tool. For a manual check against
your own account: enable listening, ask a simple question, try one ordinary
window action, then mute, and confirm the indicator clears and
`omarchy-voice log -f` records a clean session close.
