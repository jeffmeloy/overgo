# Current workbench GUI: analysis record

## Structural redesign — owner direction, 2026-09-07

This direction supersedes the historical card-based front-page and phone-polish
design. Work stays exclusively in `professional_overgo_gui`. Use flat
conversation text, lists and separators; no dashboard tiles, message cards or
large model welcome panel. A compact model selector opens a list with details
on request. Navigation is a drawer on mobile. API keys, system prompts and
generation parameters live in Settings. Only active work, failures and required
decisions occupy operational space. The conversation scrolls independently and
the composer retains visible send/stop controls, including at reduced phone
viewport heights. Preserve every existing workbench capability and server route.

Acceptance measures the user's controls and conversation, not the top of an
arbitrary content container: compose, attach, read a long reply, stop, switch
conversation/model, open/close navigation and settings, and reach Library from
a phone. Inspect captures in both themes alongside behavioral assertions.

The Impeccable review uses its Operate, Distill and Craft Floor guidance from
the local checkout at `C:\Users\jeffm\impeccable` (read only). Applied here:
one sans family for interface prose, monospace for code and measurements,
restrained teal for actions and state, flat results with separators, a shared
reading measure, consistent SVG controls, explicit focus, and larger phone
touch targets. Native dialogs protect model selection and settings focus.
The mechanical scan found a thick alert edge; it was replaced with a thin
separator. The behavioral browser lane remains the acceptance authority.

## Conversation history

History uses a flat list with title search, an archived view and Load older
conversations. Actions explicitly reveal Rename and Archive/Restore. Duplicate
titles retain distinct response identities. Automatic refresh preserves the
editor, keyboard target, selected transcript and draft; Reload applies deferred
changes. Failed searches retain the previous results and an explicit retry.

The existing interaction store owns pagination and ancestry. Continuations are
bound to the search and store head; a concurrent write asks the reader to reload
instead of mixing pages. Root identity and turn count traverse the complete
immutable chain, independently of page size. Every read refreshes the existing
store handle. Labels use compare-and-swap with distinct transition keys so
repeated archive/restore operations do not replay an earlier state.

Stored transcripts are readable across models. Continuing them still requires
the matching model and recipe. A failed transcript load blocks Send and offers
Retry loading or New conversation while preserving the unsent draft.

`TestConversationHistoryPaging`, `TestConversationHistoryRefresh` and
`TestWebUIBrowserConversationNavigation` cover these contracts using a real
temporary store, synthetic model output, browser actions and injected transport
failures. The browser journey also checks both themes and phone/desktop layouts.

## Conversation actions

Edit and resend opens an inline message editor. Regenerate uses the stored
prompt with current settings. Both submit through the existing Responses path
with the original turn's parent, creating an immutable branch. The history
list distinguishes the resulting leaves by response identity. Return to
original keeps the earlier branch reachable while the new response runs.

Branch actions preserve the composer's unsent text and files. A root edit
copies that draft to the new root while keeping the original draft. Failed
submissions restore the original view with Retry branch; disconnected accepted
requests use Resume response. Stop cancels the actual branch execution.
Keyboard focus moves to Stop when a branch starts and back to Send on completion.

Stored media positions survive regeneration, including UTF-8 text offsets.
An edited prompt keeps its attachments after the edited text. Multi-message
inputs preserve the other messages. Tool-execution turns explicitly refuse
automatic replay; continuing with a new message remains available. Model and
recipe checks guard all generation actions, while stored content remains readable.

Settings exposes Copy conversation and Export Markdown. These use the selected
stored transcript, omit unsent drafts and configuration, and reference each
attachment's stored turn. Clipboard/read failures provide retry feedback. The
shared object-URL owner releases downloads when their UI owner is replaced.

`TestWebUIBrowserConversationActions` checks actual stored branches, repeat-click
admission, failed submission/retry, execution cancellation, draft preservation,
root edits, multi-message/media request fidelity, clipboard failure and export
contents. The generation model is synthetic; media request tests deliberately
refuse before execution and establish no image-model or remote-fetch evidence.

## Mobile acceptance

The conversation renderer keeps streamed text nodes stable, preserves the
reading position, and offers Jump to latest. Attachments, mode fields and
contextual notices share a bounded scrolling area above the message field;
Send/Stop remain outside it. Long code and Markdown tables scroll within the
reply. Tool disclosures use native buttons. A separate live status announces
waiting, receiving, completion and failure without reading every token.

The viewport handler distinguishes pinch zoom from keyboard resize. The
[VisualViewport API](https://developer.mozilla.org/en-US/docs/Web/API/VisualViewport)
describes their effects, and [Chrome's viewport behavior](https://developer.chrome.com/blog/viewport-resize-behavior/)
documents `interactive-widget=resizes-content`. Safe-area padding protects
controls at screen edges; zoom remains enabled.

`TestWebUIBrowserMobileInteraction` exercises reduced heights, orientation-sized
viewports, paste, safe-area overrides, pinch zoom, reading position, keyboard
controls and announcements in Chromium. It does not prove physical keyboard,
Safari or assistive-technology behavior. `TestGUIMobileDeviceEvidence` validates
an operator's records separately and is outside the default browser-test prefix.

Obtain the current embedded GUI fingerprint with:

```
go test -short ./internal/server -run '^TestGUIMobileDeviceEvidence$' -v -count=1
```

Without evidence this short run prints the asset SHA-256 and reports an explicit
integration exclusion. A signoff run without `-short` fails when evidence is
missing. Create the actual
operator record at `docs/gui/mobile-device-evidence.json` after exercising real
hardware. Do not substitute emulator results or a schema example.

- Top level: `ui_sha256` and `devices`.
- Each device: `platform` (`ios-safari` or `android-chrome`), `physical`, `device`,
  `os_version`, `browser_version`, `assistive_technology` and `cases`. Supply both
  platforms, including VoiceOver and TalkBack respectively.
- Each named case: `passed`, `observation` and `capture` containing `path` and
  `sha256`. Captures are actual PNG/JPEG files relative to the evidence record.
  Document what happened; a checkbox alone is insufficient. All cases must pass.

| Case | Exercise on each physical device |
| --- | --- |
| keyboard | Open/close the software keyboard with text, files and mode controls; reach Send/Stop. |
| orientation | Rotate while composing and while reading; preserve text and usable layout. |
| safe-areas | Reach header, drawer and composer controls around notches and the home indicator. |
| zoom | Enlarge text/page, pan and return to normal without losing the draft or controls. |
| paste-upload | Paste and pick files; resolve loading/refusal/reattachment states. |
| long-reply | Read earlier streamed text; use Jump to latest; scroll wide code and tables. |
| stop-recovery | Stop execution, disconnect/reopen and resume without duplicate output. |
| drawer-focus | Open, navigate and close the drawer; focus stays within it and returns. |
| settings-focus | Edit settings, close/reopen and remount; preserve values and relevant focus. |
| screen-reader | Verify names, focus order, disclosures and concise progress/error announcements. |

Validate the supplied record with `OVERGO_GUI_MOBILE_EVIDENCE` set to its path
and `go test ./internal/server -run '^TestGUIMobileDeviceEvidence$' -v -count=1`.
Missing/incomplete records, failed cases, stale GUI fingerprints and missing or
changed captures fail. The validator checks source and record integrity; it does
not replay or independently attest to the operator's physical observations.
The fingerprint covers GUI assets; backend acceptance remains in the server and
real-model lanes. Subsequent GUI changes require fresh device evidence.
