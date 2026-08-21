package runrecord

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

// TestGenerationRecordBindsFullProvenance proves the day-one generation
// aggregate: a valid record round-trips its content identity, its lineage
// carries the depth-bearing child->parent edges plus a dependency edge to every
// referenced identity, the batch commits to a real store and is queryable, and
// each missing or malformed reference is refused rather than defaulted.
func TestGenerationRecordBindsFullProvenance(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	valid := GenerationRecord{
		Parents:      []artifact.ID{id(artifact.KindModel, "parent-a"), id(artifact.KindModel, "parent-b")},
		Child:        id(artifact.KindModel, "child"),
		Components:   []artifact.ID{id(artifact.KindTensorSet, "organ")},
		Bridge:       id(artifact.KindAdapter, "bridge"),
		TrainingPlan: id(artifact.KindRecipe, "training-plan"),
		Dataset:      id(artifact.KindDataset, "dataset"),
		Split:        id(artifact.KindDatasetShard, "selection-split"),
		Evaluator:    id(artifact.KindEvidence, "evaluator"),
		Code:         id(artifact.KindEvidence, "code"),
		Environment:  id(artifact.KindEvidence, "environment"),
		Seeds:        []uint64{7, 11, 13},
		Budget:       id(artifact.KindEvidence, "budget"),
		Run:          id(artifact.KindRun, "run"),
		Outcome:      OutcomeSucceeded,
		Decision:     id(artifact.KindEvidence, "decision"),
	}

	record, err := generationCodec.New(withGenerationVersion(valid))
	if err != nil {
		t.Fatal(err)
	}
	if err := record.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	content, err := record.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseGenerationRecord(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.ID != record.ID || len(parsed.Parents) != 2 || parsed.Bridge != valid.Bridge ||
		parsed.Seeds[2] != 13 || parsed.Decision != valid.Decision {
		t.Fatalf("parsed record differs: %+v", parsed)
	}

	lineage := record.Lineage()
	hasEdge := func(child, parent artifact.ID, relation artifact.Relation) bool {
		for _, edge := range lineage {
			if edge.Child == child && edge.Parent == parent && edge.Relation == relation {
				return true
			}
		}
		return false
	}
	for _, parent := range valid.Parents {
		if !hasEdge(record.Child, parent, artifact.RelationTrainedFrom) {
			t.Fatalf("missing depth edge child->%s", parent)
		}
	}
	for _, dependency := range []artifact.ID{valid.TrainingPlan, valid.Dataset, valid.Split,
		valid.Evaluator, valid.Code, valid.Environment, valid.Budget, valid.Run, valid.Decision,
		valid.Bridge, valid.Components[0]} {
		if !hasEdge(record.ID, dependency, artifact.RelationDependsOn) {
			t.Fatalf("record does not depend on %s", dependency)
		}
	}

	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	static := []artifact.ID{valid.Parents[0], valid.Parents[1], valid.Child, valid.Components[0],
		valid.Bridge, valid.TrainingPlan, valid.Dataset, valid.Split, valid.Evaluator, valid.Code,
		valid.Environment, valid.Budget, valid.Run, valid.Decision}
	descriptors := make([]artifact.Descriptor, len(static))
	for index, value := range static {
		descriptors[index] = artifact.Descriptor{ID: value}
	}
	if _, err := store.Commit(context.Background(), artifact.Batch{Key: "generation/fixture/static", Artifacts: descriptors}); err != nil {
		t.Fatal(err)
	}
	batch, err := record.Batch("generation/fixture/v1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		t.Fatal(err)
	}
	parents, err := store.Parents(context.Background(), record.Child)
	if err != nil || len(parents) == 0 {
		t.Fatalf("child lineage not queryable: (%v, %v)", parents, err)
	}

	refusals := []struct {
		name   string
		mutate func(*GenerationRecord)
	}{
		{"no parents", func(r *GenerationRecord) { r.Parents = nil }},
		{"duplicate parents", func(r *GenerationRecord) { r.Parents = []artifact.ID{r.Parents[0], r.Parents[0]} }},
		{"child equals parent", func(r *GenerationRecord) { r.Child = r.Parents[0] }},
		{"components without bridge", func(r *GenerationRecord) { r.Bridge = artifact.ID{} }},
		{"bridge without components", func(r *GenerationRecord) { r.Components = nil }},
		{"bridge kind", func(r *GenerationRecord) { r.Bridge = r.Parents[0] }},
		{"no seeds", func(r *GenerationRecord) { r.Seeds = nil }},
		{"run kind", func(r *GenerationRecord) { r.Run = r.Code }},
		{"decision kind", func(r *GenerationRecord) { r.Decision = r.Run }},
		{"shared evidence identity", func(r *GenerationRecord) { r.Budget = r.Code }},
		{"outcome", func(r *GenerationRecord) { r.Outcome = Outcome("mystery") }},
	}
	for _, refusal := range refusals {
		candidate := valid
		candidate.Parents = append([]artifact.ID{}, valid.Parents...)
		candidate.Components = append([]artifact.ID{}, valid.Components...)
		candidate.Seeds = append([]uint64{}, valid.Seeds...)
		refusal.mutate(&candidate)
		if _, err := generationCodec.New(withGenerationVersion(candidate)); err == nil {
			t.Errorf("%s: accepted", refusal.name)
		}
	}
}

// withGenerationVersion stamps the codec-required version the lifecycle
// author will own once it exists.
func withGenerationVersion(record GenerationRecord) GenerationRecord {
	record.Version = artifact.InitialDocumentVersion
	return record
}

// TestGenerationRecordBindsBudget pins the typed replacement for the old
// maxGenerationSeeds code threshold: a record's seed consumption validates
// against its BOUND budget grant -- within-grant passes, over-consumption is
// refused with the grant named, and a foreign budget document is refused as
// not the record's binding.
func TestGenerationRecordBindsBudget(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	grant, err := budgetCodec.New(Budget{
		Version: artifact.InitialDocumentVersion, Unit: "seeds", Split: id(artifact.KindDatasetShard, "selection"),
		Issued: 3, Authority: id(artifact.KindEvidence, "authority"),
	})
	if err != nil {
		t.Fatal(err)
	}
	record := GenerationRecord{
		Parents:      []artifact.ID{id(artifact.KindModel, "parent")},
		Child:        id(artifact.KindModel, "child"),
		Components:   []artifact.ID{id(artifact.KindTensorSet, "organ")},
		Bridge:       id(artifact.KindAdapter, "bridge"),
		TrainingPlan: id(artifact.KindRecipe, "plan"),
		Dataset:      id(artifact.KindDataset, "dataset"),
		Split:        id(artifact.KindDatasetShard, "selection"),
		Evaluator:    id(artifact.KindEvidence, "evaluator"),
		Code:         id(artifact.KindEvidence, "code"),
		Environment:  id(artifact.KindEvidence, "environment"),
		Seeds:        []uint64{7, 11, 13},
		Budget:       grant.ID,
		Run:          id(artifact.KindRun, "run"),
		Outcome:      OutcomeSucceeded,
		Decision:     id(artifact.KindEvidence, "decision"),
	}
	bound, err := generationCodec.New(withGenerationVersion(record))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateGenerationBudget(bound, grant); err != nil {
		t.Fatalf("within-grant consumption refused: %v", err)
	}
	over := record
	over.Seeds = []uint64{7, 11, 13, 17}
	overBound, err := generationCodec.New(withGenerationVersion(over))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateGenerationBudget(overBound, grant); err == nil {
		t.Fatal("over-grant consumption accepted")
	}
	foreign, err := budgetCodec.New(Budget{
		Version: artifact.InitialDocumentVersion, Unit: "seeds", Split: record.Split,
		Issued: 100, Authority: id(artifact.KindEvidence, "other-authority"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateGenerationBudget(bound, foreign); err == nil {
		t.Fatal("foreign budget document accepted as the record's binding")
	}
}
