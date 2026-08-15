package trainingprogram

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestRecursiveImprovementLineageAndExternalPromotion(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	base := ImprovementSpec{
		ParentModel: id(artifact.KindModel, "parent"), Dataset: id(artifact.KindDataset, "data"),
		DevelopmentSplit: id(artifact.KindDatasetShard, "development"), Recipe: id(artifact.KindRecipe, "recipe"),
		Code: id(artifact.KindEvidence, "code"), Proposer: id(artifact.KindEvidence, "proposer"),
	}
	tests := []struct {
		kind       ImprovementKind
		candidate  artifact.ID
		components []artifact.ID
	}{
		{ImprovementCorpus, id(artifact.KindDataset, "candidate-corpus"), nil},
		{ImprovementRecipe, id(artifact.KindRecipe, "candidate-recipe"), nil},
		{ImprovementEvaluator, id(artifact.KindEvidence, "candidate-evaluator"), nil},
		{ImprovementComponentComposition, id(artifact.KindModelDefinition, "candidate-composition"), []artifact.ID{
			id(artifact.KindModel, "component-b"), id(artifact.KindAdapter, "component-a"),
		}},
	}
	for _, test := range tests {
		spec := base
		spec.Kind, spec.Candidate, spec.Components = test.kind, test.candidate, test.components
		proposal, err := CompileImprovementProposal(spec)
		if err != nil || proposal.ID().Kind() != artifact.KindRecipe || proposal.Candidate() != test.candidate {
			t.Fatalf("compile %s = (%s, %s, %v)", test.kind, proposal.ID(), proposal.Candidate(), err)
		}
	}
	invalid := base
	invalid.Kind = ImprovementEvaluator
	invalid.Candidate = id(artifact.KindRecipe, "wrong-kind")
	if _, err := CompileImprovementProposal(invalid); err == nil {
		t.Fatal("candidate kind mismatch accepted")
	}
}
