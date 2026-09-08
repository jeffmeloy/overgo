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

The [resource refresh](e4b-resource-refresh.json) binds producer
`95c0ac02654d9e9034451f9ab1a4d4b2abd0646b` after the retained-output arena
repair. All 270 media requests passed in 219.6 seconds; the full 32,768-token
text recovery passed in 167.2 seconds. Text recovery retained 28,093,855,912
owned bytes across nine cycles and released them after each of three loads.
Media retained 17,442,088,104 owned bytes and 4,260 allocations without growth.
`TestAcceptedE4BResourceRefresh` checks the exact source, recipe, environment,
text-test digest/transcript and six observed media resource cells.

The repair changes arena planning only. Kernels, arithmetic, tokenizer,
projection, prompting, native oracles and dataset scoring are unchanged.
Existing executor lifetime, alias and replay checks passed in the producer
gate. The original quality, protocol and mask records retain their source;
resource acceptance uses fresh executions rather than relabeling old records.

Three new controls per model pass historical, prior-accepted and first-repeat
comparisons. E4B judged decode: 40.0, 39.7, 40.1 tokens/s; Qwen: 245.7, 245.6,
247.2 tokens/s. Every repeat includes the short shape and four context rungs,
full outputs, NLL, rates and allocation limits. Earlier cohorts and failures
remain in the store; no floor or denominator changed.

Resource selection:
`evidence:sha256:8e6d2914d494a230f55be695d076e9538d5aa4b75708e3cb742f0672e7bcba32`.
