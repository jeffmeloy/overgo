package composition

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

// RecordChainViability commits the Tier-0 chain experiment as a typed
// generation record: both whole models as parents, the content-addressed chain
// recipe as the training plan (no components, no bridge -- a whole-model
// child), and a decision identity carrying the SHIP or REFUSE reason. The
// probe records refusals exactly as loudly as ships.
func RecordChainViability(store *repodb.Store, config ChainConfig, result ChainResult) (runrecord.GenerationRecord, error) {
	must := func(kind artifact.Kind, payload string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte("tier0-chain/v1:"+payload))
		if err != nil {
			panic("composition: identify " + payload + ": " + err.Error())
		}
		return id
	}
	protocol := fmt.Sprintf("scorer=%s;drafter=%s;prefix=%d;draft=%d;target=%d;windows=%d",
		config.ScorerDir, config.DrafterDir, config.Prefix, config.Draft, config.Target, len(config.Windows))
	outcome := runrecord.OutcomeFailed
	if result.Ship {
		outcome = runrecord.OutcomeSucceeded
	}
	record, err := runrecord.NewGenerationRecord(runrecord.GenerationRecord{
		Parents: []artifact.ID{
			must(artifact.KindModel, "model:"+config.ScorerDir),
			must(artifact.KindModel, "model:"+config.DrafterDir),
		},
		Child:        must(artifact.KindModel, "chained:"+protocol),
		TrainingPlan: result.ChainRecipe,
		Dataset:      must(artifact.KindDataset, "windows:"+protocol),
		Split:        must(artifact.KindDatasetShard, "targets:"+protocol),
		Evaluator:    must(artifact.KindEvidence, "evaluator:displaced-heldout-ce-envelope"),
		Code:         must(artifact.KindEvidence, "code:"+protocol),
		Environment:  must(artifact.KindEvidence, "environment:host-reference"),
		Seeds:        []uint64{0},
		Budget:       must(artifact.KindEvidence, fmt.Sprintf("budget:windows=%d;draft=%d;greedy-deterministic", len(config.Windows), config.Draft)),
		Run:          must(artifact.KindRun, "run:"+protocol+";"+result.Reason),
		Outcome:      outcome,
		Decision:     must(artifact.KindEvidence, "decision:"+result.Reason),
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	// The chain and baseline recipes are NOT in this batch: the workflow
	// runtime already committed them with full content facts, and a bare
	// descriptor for the same identity would conflict.
	static := append([]artifact.ID{record.Child,
		record.Dataset, record.Split, record.Evaluator, record.Code, record.Environment,
		record.Budget, record.Run, record.Decision}, record.Parents...)
	descriptors := make([]artifact.Descriptor, len(static))
	for index, id := range static {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	ctx := context.Background()
	if _, err := store.Commit(ctx, artifact.Batch{Key: "tier0-chain/facts/" + record.ID.String(), Artifacts: descriptors}); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	batch, err := record.Batch("tier0-chain/generation/" + record.ID.String())
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	return record, nil
}

// RecordViability commits the experiment as a typed generation record: the
// depth-bearing child/parent edges plus references for the protocol facts, and
// a decision identity carrying the SHIP or REFUSE reason. Day-one identities
// are content hashes of the protocol facts themselves -- referential and
// replayable, upgraded to full artifact bindings when the experiment
// lifecycle row lands. The probe records refusals exactly as loudly as ships.
func RecordViability(store *repodb.Store, config Config, result Result) (runrecord.GenerationRecord, error) {
	fact := func(kind artifact.Kind, payload string) (artifact.ID, error) {
		return artifact.IdentifyBytes(kind, []byte("graft-probe/v1:"+payload))
	}
	must := func(kind artifact.Kind, payload string) artifact.ID {
		id, err := fact(kind, payload)
		if err != nil {
			panic("composition: identify " + payload + ": " + err.Error())
		}
		return id
	}
	protocol := fmt.Sprintf("target=%s;donor=%s;tensor=%s;layer=%d;steps=%d;seeds=%d",
		config.TargetDir, config.DonorDir, result.DonorTensor, result.GraftLayer, config.Steps, len(config.Seeds))
	seeds := make([]uint64, len(config.Seeds))
	for index, seed := range config.Seeds {
		seeds[index] = uint64(seed)
	}
	outcome := runrecord.OutcomeFailed
	if result.Ship {
		outcome = runrecord.OutcomeSucceeded
	}
	record, err := runrecord.NewGenerationRecord(runrecord.GenerationRecord{
		Parents:      []artifact.ID{must(artifact.KindModel, "target:"+config.TargetDir)},
		Child:        must(artifact.KindModel, "composed:"+protocol),
		Components:   []artifact.ID{must(artifact.KindTensorSet, "component:"+result.DonorTensor+":"+config.DonorDir)},
		Bridge:       must(artifact.KindAdapter, "bridge:"+protocol),
		TrainingPlan: must(artifact.KindRecipe, "plan:"+protocol),
		Dataset:      must(artifact.KindDataset, "tokens:"+protocol),
		Split:        must(artifact.KindDatasetShard, "heldout:"+protocol),
		Evaluator:    must(artifact.KindEvidence, "evaluator:heldout-ce-envelope"),
		Code:         must(artifact.KindEvidence, "code:"+protocol),
		Environment:  must(artifact.KindEvidence, "environment:host-reference"),
		Seeds:        seeds,
		Budget:       must(artifact.KindEvidence, fmt.Sprintf("budget:seeds=%d;steps=%d", len(seeds), config.Steps)),
		Run:          must(artifact.KindRun, "run:"+protocol+";"+result.Reason),
		Outcome:      outcome,
		Decision:     must(artifact.KindEvidence, "decision:"+result.Reason),
	})
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	static := append([]artifact.ID{record.Child, record.Bridge, record.TrainingPlan, record.Dataset,
		record.Split, record.Evaluator, record.Code, record.Environment, record.Budget,
		record.Run, record.Decision}, record.Parents...)
	static = append(static, record.Components...)
	descriptors := make([]artifact.Descriptor, len(static))
	for index, id := range static {
		descriptors[index] = artifact.Descriptor{ID: id}
	}
	ctx := context.Background()
	if _, err := store.Commit(ctx, artifact.Batch{Key: "graft-probe/facts/" + record.ID.String(), Artifacts: descriptors}); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	batch, err := record.Batch("graft-probe/generation/" + record.ID.String())
	if err != nil {
		return runrecord.GenerationRecord{}, err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return runrecord.GenerationRecord{}, err
	}
	return record, nil
}
