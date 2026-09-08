# CPU audio adaptation

`cmd/train -audio-manifest <json> -recipe <active-training-recipe> -store <store> -out <new-checkpoint-directory> [-steps <updates>] [-resume <checkpoint-directory>]` executes through `trainingworkflow.Execute`.

The manifest uses `trainingworkflow.AudioTrainingSpec`: exact unadapted transcription `base_recipe`, `memory_bytes`, explicit audio `inspection` policy, required `lowercase` boolean, and dataset-order `seed`/`shuffle`. The active recipe binds a CTC objective, registered dataset version, selected split, audio-to-text signature, optimizer and exact processing profiles. Omitting steps derives one update per selected source record. Dense model/dataset, token-length and RL options refuse.

Source audio is read from its registered location, not copied into the store. Raw target text remains unchanged in the source; the declared transformation precedes tokenization. Training reuses the serving session's grouped frontend and frozen encoder, the shared input-projection/CTC adapter and Muon optimizer. The admission ceiling bounds each component's numeric storage, not measured process peak memory.

Checkpoint publication includes adapter weights, optimizer/scheduler state, stream identity/cursor, ordering declarations and exact lineage. Resume validates all authorities before restoration. A resulting serving recipe remains a candidate: training never activates it. Cancellation joins the session, records a cancelled observation and refuses subsequent checkpoint publication. CTC record cursors are not reported as token counts.

`TestASRProductionTrainingAcceptance` runs the actual command on pinned LibriSpeech audio, compares its first update with the independent library path, checks fresh-process exact resume, reloads through authenticated HTTP, and interrupts a production update. Evidence is bounded CPU adapter capability; no quality gain, full-encoder gradient, native streaming, training HTTP or GPU claim.
