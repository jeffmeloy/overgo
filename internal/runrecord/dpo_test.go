package runrecord

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestDPORunEvidenceBindsAuthority(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, name)
	}
	recipe := id(artifact.KindRecipe, "dpo-run-plan")
	policy := id(artifact.KindModel, "dpo-policy")
	reference := id(artifact.KindModel, "dpo-reference")
	dataset := id(artifact.KindDataset, "dpo-dataset")
	checkpoint := id(artifact.KindCheckpoint, "dpo-checkpoint")
	run, err := NewRun(recipe, OutcomeSucceeded, []artifact.ID{policy, reference, dataset}, []artifact.ID{checkpoint}, "")
	if err != nil {
		t.Fatal(err)
	}
	evaluation, err := NewEvaluation(recipe, run.ID, dataset, []Metric{
		{Name: "dpo-loss", Value: 0.6, Direction: DirectionMinimize},
		{Name: "preference-margin", Value: 0.2, Direction: DirectionMaximize},
	})
	if err != nil {
		t.Fatal(err)
	}
	lineage := run.Lineage()
	if !slices.Contains(lineage, artifact.Lineage{Child: run.ID, Parent: reference, Relation: artifact.RelationDependsOn}) ||
		!slices.Contains(lineage, artifact.Lineage{Child: checkpoint, Parent: run.ID, Relation: artifact.RelationProducedBy}) {
		t.Fatalf("run lineage=%+v", lineage)
	}
	if evaluation.Recipe != recipe || evaluation.Run != run.ID || evaluation.Dataset != dataset || len(evaluation.Metrics) != 2 {
		t.Fatalf("DPO evaluation=%+v", evaluation)
	}
}
