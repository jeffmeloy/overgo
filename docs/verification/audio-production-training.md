# CPU audio training and measurement

`cmd/train -audio-manifest <json> -recipe <active-training-recipe> -store <store> -out <new-checkpoint-directory> [-steps <updates>] [-resume <checkpoint-directory>]` uses `trainingworkflow.Execute`.

`AudioTrainingSpec` requires unadapted `base_recipe`, `memory_bytes`, `inspection` and explicit `lowercase`; `seed`/`shuffle` declare ordering. The active recipe binds CTC, dataset/split, signature, optimizer and profiles. Omitted steps derive one update per record. Dense-model, token-length and RL options refuse.

Audio is read in place; target transformation preserves source text. Serving and training share frontend/frozen encoder, input-projection/CTC and Muon. Checkpoints preserve weights, optimizer/scheduler, stream/RNG and lineage. Resume validates authorities. Publication creates inactive candidates. Cancellation records a cancelled observation without publishing a checkpoint; record cursors are not token counts.

`TestASRProductionTrainingAcceptance` proves command/library update parity, fresh-process exact resume, authenticated HTTP reload and cancellation. `TestAudioResourceFitnessAcceptance` adds five preselected LibriSpeech clean test utterances from three speakers, source-bound WER/CER, failure denominators and silence admission. Corrupt input retains quarantine evidence without invented duration or transcript.

Component admission ceilings are not measured memory. The evaluator separates cold load, warmup and timed inference. `processmeasure.Measure` adds whole-child wall/peak working set, including source I/O, loading, scoring and publication. Target-free profiling runs separately and preserves raw outputs. Shared resource comparisons derive empirical envelopes, retain regressions and leave unknown GPU measures absent. Shared vector-policy counterexamples reject no gain, quality regression and unapproved tradeoffs.

The profile identifies shared linear algebra as the dominant CPU cost. Execution is unchanged: no runtime API, state or buffer addition, speedup, full-encoder training, training-resource, native-streaming, training-HTTP, GPU or promotion claim. Bounded read-speech evidence does not establish noisy, multilingual or full-corpus quality.
