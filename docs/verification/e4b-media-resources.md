# E4B media resource acceptance

`TestE4BMediaResourceRecovery` reuses the request preparation and native captures
from `TestE4BHTTPModalities`. It exercises text, image, audio, multiple images,
ordered video and mixed media history through Chat. The protocol test separately
checks all six inputs through native completion, Chat and Responses.

Each of three complete loads runs three six-mode cycles. Every input executes
five actions: complete generation, cancellation during token emission, complete
recovery, a token-consumer error, and complete recovery. The denominator is 270
requests: 162 complete native-matching outputs, 54 cancellations and 54 consumer
errors. Interruption credit requires the requested error from actual generation;
an admission refusal or a request canceled before generation cannot replace it.

Device accounting includes both the projector and language runner. Each request
resets their allocation high-water windows. Projection runs before language
generation, so the combined peak is the maximum of:

- projector window peak plus the language runner's retained bytes during projection;
- language window peak plus the projector's retained bytes during generation.

Independent lifetime peaks are not added. These are owned allocation counters,
not total process or board memory. The first complete six-mode cycle sets the
retained-byte and allocation ceilings. Later cycles and loads must not grow
either ceiling. Each close must return at least the accounted live bytes to the
device, and closed sessions must refuse use. Device-free-memory comparisons
require exclusive GPU ownership.

With `OVERGO_E4B_PUBLISH_RESOURCES=1`, publication requires an unchanged clean
source commit before execution and after all releases. The existing
`ServingObservation` contract stores each request's outcome, workload counts and
start, prefill-finish and generation-finish hardware samples. An existing
`GateResult`/`Run` binds the 270 observations and three releases to the exact
projection recipe, model definition, environment, native fixture digest and
request, prepared-prompt and complete-output digests. The recipe pins the model
configuration and media processor. No new controller or evidence store is used.

The compatibility acceptance reads those records without inference or writes.
It checks the complete repetition denominator, exact identities, observed
outcomes, phase-derived peak, warm-cycle retention and release. It derives
recovery counters from individual records instead of trusting a supplied total.
One identical six-mode bundle can satisfy all six resource cells.

This is a bounded media recovery claim. It does not replace the separate
32768-token text recovery guard, corpus-backed modality quality, held-out speech
evaluation or final model admission. Ordinary producer checks do not publish or
promote the projection recipe.
