package runrecord

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestGenerationRecordBindsFullProvenance(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	parents := []artifact.ID{id(artifact.KindModel, "parent-b"), id(artifact.KindModel, "parent-a")}
	components := []artifact.ID{id(artifact.KindModelDefinition, "component"), id(artifact.KindProjector, "projector")}
	record, err := NewGenerationRecord(GenerationRecord{
		Parents: parents, Components: components, Bridge: id(artifact.KindAdapter, "bridge"),
		TrainingPlan: id(artifact.KindRecipe, "training-plan"), DataSplit: id(artifact.KindDatasetShard, "development"),
		Evaluator: id(artifact.KindEvidence, "evaluator"), Code: id(artifact.KindEvidence, "code"),
		Kernel: id(artifact.KindEvidence, "kernel"), Environment: id(artifact.KindEvidence, "environment"),
		Seeds: id(artifact.KindEvidence, "seeds"), Budget: id(artifact.KindEvidence, "budget"),
		Run: id(artifact.KindRun, "run"), Outcome: id(artifact.KindEvaluation, "outcome"),
		Decision: id(artifact.KindEvidence, "decision"),
	})
	if err != nil {
		t.Fatal(err)
	}
	parents[0] = parents[1]
	components[0] = components[1]
	if record.Parents[0] == record.Parents[1] || record.Components[0] == record.Components[1] {
		t.Fatal("generation record retained caller-owned slices")
	}
	lineage := artifact.DependencyLineage(record.ID, generationDependencies(record)...)
	if len(lineage) != len(record.Parents)+len(record.Components)+12 {
		t.Fatalf("generation lineage has %d edges", len(lineage))
	}
	bound := map[artifact.ID]bool{}
	for _, edge := range lineage {
		if edge.Child != record.ID || edge.Relation != artifact.RelationDependsOn {
			t.Fatalf("invalid generation lineage edge: %+v", edge)
		}
		bound[edge.Parent] = true
	}
	for _, required := range slices.Concat(record.Parents, record.Components, []artifact.ID{
		record.Bridge, record.TrainingPlan, record.DataSplit, record.Evaluator, record.Code, record.Kernel,
		record.Environment, record.Seeds, record.Budget, record.Run, record.Outcome, record.Decision,
	}) {
		if !bound[required] {
			t.Fatalf("generation lineage omits %s", required)
		}
	}
	content, err := generationCodec.Content(record)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := generationCodec.Parse(content.Data)
	if err != nil || parsed.ID != record.ID || parsed.Run != record.Run || parsed.Decision != record.Decision {
		t.Fatalf("generation round trip = (%+v, %v)", parsed, err)
	}
	if _, err := record.Batch("fixture/generation"); err != nil {
		t.Fatal(err)
	}
	invalid := record
	invalid.Outcome = id(artifact.KindOutput, "untyped-outcome")
	if _, err := NewGenerationRecord(invalid); err == nil {
		t.Fatal("untyped generation outcome accepted")
	}
}
