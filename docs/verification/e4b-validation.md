# E4B validation

Source: `1b7ac58150c36e83faaaf016cdd03d2a0aefd8ab`.
The [selection](e4b-validation.json) binds 30 measured cells: 18 protocol,
six dataset-quality and six resource cells. `TestAcceptedE4BModalities`
checks their identities, denominators, bounds and promoted projection binding.

| Quality panel | Go result | Native FP32 / BF16 |
| --- | --- | --- |
| Text: MMLU abstract algebra, all test questions | 46/100 | 46/100 / 48/100 |
| Image: caption discrimination | 16/16 | 16/16 / 16/16 |
| Multiple images: caption ordering | 15/16 | 15/16 / 15/16 |
| Audio: held-out LibriSpeech | WER 2.567%; CER 0.837%; 32 clips | Frozen WER ceiling 2.953%; CER ceiling 0.908% |
| Video: caption discrimination over two ordered frames | 16/16 | 16/16 / 16/16 |
| Mixed image and speech: paired-label selection | 8/16 | 8/16 / 8/16 |

Every shared-native-correct classification remains correct in Go. The text
panel retains all questions, labels and failed captures. Native explanations
exceeded both the initial 32-token cap and the revised 256-token cap. The
final bounded classification protocol keeps 256 tokens and scores non-label
answers incorrect. Go produced 97 normal completions and three cap hits whose
full token prefixes match native captures; FP32 capped three cases, BF16 four.
Speech and protocol completion requirements remain unchanged.

Visual panels select the first 16 caption-matched local VidGen clips before
scoring, from 4,904 available videos. Source videos, frame hashes and timestamps
are pinned. These panels measure scene-caption discrimination and ordering;
they do not establish dense-caption factuality, broad temporal reasoning or
unseen-training-data performance. Mixed accuracy remains a model limitation.
The 100-question text panel does not replace full MMLU-Pro.

Held-out speech uses 32 evenly spaced original rows from 2,620 test utterances,
without replacements. One warmup and three measured rounds produced 128
complete outputs. Every case meets its independently frozen native edit bound;
there were no admission or inference failures. The separate eight-case
validation panel also matches all 32 repeated native outputs.

Protocol checks cover all six input modes through native completion, Chat and
Responses. Media recovery executes 270 requests across three loads and three
cycles: 162 complete native-matching outputs, 54 cancellations and 54 consumer
errors. Owned allocations do not accumulate; closing releases the accounted
memory. Mask-sensitive hidden-state interventions distinguish declared causal
image attention from the wrong-mask control.

Store authorities:

- Completed model and controls: `evidence:sha256:6475ce8a691922d4a2482f7dc13a0d5f83531891afca00174c81100bc8700570`.

- Selection: `evidence:sha256:f5389b9111485ba8afcea713073251fe760f4f9de63a89e6eaef809a6b1eab6f`.
- Reproduction and failed captures: `evidence:sha256:09f923f4190207c663070e8445876f102c5838b4783c1450096a850460960206`.
- Held-out native bounds: `evidence:sha256:fa1ff7ce6f8f222a1cd09c052c49d03f83cef2ab063ac3342de49d05491d0b2c`.
- Visual native bounds: `evidence:sha256:476f0890e4a9059c334865d3d3f385a7dde5b313c6d26af9b57ca18321bed9c7`.
- Text native bounds: `evidence:sha256:bd58fd0dcba826e60b1fdfccf79867aa12f600fa775c44d5b926e4522f0f1b19`.

The separate 32,768-token text recovery result retains three loads and nine
grow/shrink cycles. Its reuse review records the intervening source changes;
allocation and generation paths are unchanged. Fixed-budget regression
admission remains separate from these modality-quality measurements.

The fixed-budget guard now accepts three current-source E4B repeats and three
Qwen control repeats. Each checks the short shape and all four required rungs
against retained historical tokens, NLL, throughput and allocation limits.
E4B judged decode rates were 40.3, 39.3 and 40.7 tokens/s. The entire quiet
three-repeat experiment passes the previous pool reference as well.

The initial control `evidence:sha256:bcb2edb57f93cbdbe83ad6998ac346f76d1627d8cb11385b0fa188493c1fb39e`
missed the pool reference's short-prompt rate floor (3,249.5 versus 3,830.8
tokens/s). It remains recorded. The complete replacement experiment ran without
concurrent agent compilation; no threshold changed and no best repeat was chosen.

Comparison preserves a legacy record's device-class scope when it predates UUID
capture. Known UUIDs must still match and cannot disappear. Historical provenance
checks hash the original stored bytes instead of reserializing them with newer
optional fields. Neither repair changes inference or its measured source surface.

The workbench first-run journey also passes its E4B attached-image reply.
Attachments precede the question, and projector offload follows active inference
placement unless explicitly overridden. The gate exposed both the prior ordering
rejection and the CPU-projector timeout. These fixes change client construction
and server defaults; the measured inference implementation remains unchanged.
