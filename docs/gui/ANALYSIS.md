# Current workbench GUI: analysis record

## Structural redesign — owner direction, 2026-09-07

Work stays in `professional_overgo_gui`. Flat conversation text, lists and
separators replace cards and model welcome panels. The compact model selector
discloses details; mobile navigation uses a drawer. Settings owns configuration.
The independently scrolling conversation keeps Send/Stop reachable at reduced
phone heights. Preserve existing workbench capabilities and server routes.

The read-only Impeccable Operate, Distill and Craft Floor review at
`C:\Users\jeffm\impeccable` informs typography, restrained teal, separators,
SVG controls, visible focus and phone touch targets. Functional browser journeys
and both-theme captures establish acceptance; styling does not establish it.

## Conversation history

History offers search, archived view, pagination and Rename/Archive/Restore.
Response IDs distinguish duplicate titles. Refresh preserves editing, focus,
selection and drafts; Reload applies deferred changes. Failed searches retain results.

The interaction store owns complete ancestry and head-bound pagination;
concurrent writes require Reload. Reads refresh the existing handle. Label CAS
uses distinct transition keys, including repeated archive/restore operations.
Cross-model transcripts remain readable; continuing requires the matching model
and recipe. Failed loads block Send and offer Retry/New without losing drafts.

`TestConversationHistoryPaging`, `TestConversationHistoryRefresh` and
`TestWebUIBrowserConversationNavigation` use real temporary storage, synthetic
output and transport faults, including phone/desktop layouts in both themes.

## Conversation actions

Edit/Regenerate use Responses with original ancestry and current settings.
Immutable branches retain Return to original and drafts; root edits copy drafts.
Failures offer Retry branch; accepted disconnected requests offer Resume response.
Stop cancels execution and retains focus.

Regeneration preserves UTF-8 media positions and other input messages. Editing
retains attachments after the edited text. Tool-execution replay is explicitly
refused. Model/recipe checks guard generation, independently of transcript reads.

Settings copies/exports stored transcripts, excluding drafts/configuration.
Attachments reference stored turns; failed downloads retry through shared URL ownership.

`TestWebUIBrowserConversationActions` covers stored branches, duplicate admission,
failure/retry/cancel, root drafts, media fidelity and copy/export. Output is
synthetic; refused media requests establish no image-model or remote-fetch proof.

## Attachment intake

Flat file rows expose reading, decoding, uploading, refusal, failure and ready
states, with Retry/Cancel/Remove and declared-format guidance. Stable previews
and keyboard targets survive other files' updates. The bounded composer extras
area keeps message input and Send/Stop reachable at phone keyboard heights.

Read/upload attempts own cancellation and reject late results after removal,
navigation or input changes. Native input assignment cannot overwrite a manually
edited field or a newer upload. Removing a file clears only its own assignment.
Unknown MIME uploads preserve raw bytes and let the server refuse the format.
Protected previews and stored-output reuse use shared authenticated transport.
Drafts retain immutable file references, never bytes; Reload stored file checks
current capabilities and exposes failures for retry.

`TestWebUIBrowserAttachmentWorkflow` covers native multi-file/paste, read/decode
faults, size/type refusal, upload retry/cancel, late assignments, authenticated
intake/preview/reuse and reduced phone layouts. Storage and file bytes are real;
controlled image/slot declarations establish no model or physical-device proof.

## Background work

One relevant Activity entry opens flat operation lists, receipts, results and
decisions. Idle chrome is absent. Technical details are disclosed; completed
outputs remain reachable through the Activity tab and authenticated downloads.
Live updates preserve focus and failed actions; stale receipts cannot replace
a newer selection. Cold-start refusal is idle, not an outage. Reconnecting streams reject replaced callbacks.
Wait fallback retains blocked decisions and cancels its observer on navigation.

Native Stop cancels the accepted operation, including a Stop before its ID;
failed cancellation permits retry. Reopened cancelled conversations show Stopped;
the execution context retains cancellation when executor errors lose their type.
Invalid Settings and required inputs retain
messages and editors. Unknown admission outcomes persist with explicit review
and resend acknowledgement. Library registration prevents duplicate admission;
catalog refresh preserves forms and rejects stale results. Existing store
transaction owners retain concurrent publication and conversation drafts.

Physical resource contention retains its typed cause through command exit and
model startup into a structured `resource_busy` response. The GUI explains
waiting and explicit retry, keeps the draft, and confirms the served recipe
before enabling Send. Other startup failures are not classified as contention.

`TestWebUIBrowserBackgroundWork`, `TestWorkbenchTransactionWriter`,
`TestModelSwapStartupBusy` and `TestResourceContention` cover these boundaries.
Browser output/operations/storage are real with synthetic generation; desktop,
phone and keyboard-sized Chromium captures establish no physical-device proof.

## Mobile acceptance

Stable streamed text preserves reading position with Jump to latest. Bounded
composer extras leave Send/Stop reachable. Wide code/tables scroll within replies;
native tool disclosures and concise live status support keyboard use.

The viewport handler distinguishes pinch zoom from keyboard resize. The
[VisualViewport API](https://developer.mozilla.org/en-US/docs/Web/API/VisualViewport)
describes their effects, and [Chrome's viewport behavior](https://developer.chrome.com/blog/viewport-resize-behavior/)
documents `interactive-widget=resizes-content`. Safe-area padding protects
controls at screen edges; zoom remains enabled.

`TestWebUIBrowserMobileInteraction` exercises Chromium viewport, paste, safe-area,
zoom, scrolling, keyboard and announcement behavior. Physical Safari/Android and
assistive technology remain separately validated by `TestGUIMobileDeviceEvidence`.

`go test -short ./internal/server -run '^TestGUIMobileDeviceEvidence$' -v -count=1`
prints the current GUI SHA-256 with an explicit exclusion when evidence is absent.
Signoff without `-short` fails on missing evidence. Record actual hardware cases
in `docs/gui/mobile-device-evidence.json`; emulators/schema examples do not qualify.

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
