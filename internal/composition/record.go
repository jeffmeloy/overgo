package composition

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

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
