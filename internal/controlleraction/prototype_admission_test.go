package controlleraction

import (
	"context"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

func prototypeImprovementAction(t *testing.T) Action {
	t.Helper()
	return Action{Version: ActionVersion, Kind: KindImprovementTrial, Improvement: &ImprovementAction{
		Proposal: trainingprogram.ImprovementSpec{
			Kind:             trainingprogram.ImprovementModelPrototype,
			ParentModel:      testutil.ArtifactID(t, artifact.KindModel, "prototype-parent-model"),
			Incumbent:        testutil.ArtifactID(t, artifact.KindProfile, "incumbent-prototype"),
			Candidate:        testutil.ArtifactID(t, artifact.KindProfile, "candidate-prototype"),
			Dataset:          testutil.ArtifactID(t, artifact.KindDataset, "prototype-dataset"),
			DevelopmentSplit: testutil.ArtifactID(t, artifact.KindDatasetShard, "prototype-development"),
			Recipe:           testutil.ArtifactID(t, artifact.KindRecipe, "prototype-execution-recipe"),
			Code:             testutil.ArtifactID(t, artifact.KindEvidence, "prototype-code"),
			Proposer:         testutil.ArtifactID(t, artifact.KindEvidence, "prototype-proposer"),
		},
		PromotionSplit: testutil.ArtifactID(t, artifact.KindDatasetShard, "prototype-promotion"),
		Evaluator:      testutil.ArtifactID(t, artifact.KindEvidence, "prototype-evaluator"),
		Authority:      testutil.ArtifactID(t, artifact.KindEvidence, "prototype-authority"),
		Objective:      testutil.ArtifactID(t, artifact.KindRecipe, "prototype-objective"),
		GPUMinutes:     45,
	}}
}

// TestModelPrototypeAdmissionContract pins the controller-side closure: a
// model-prototype proposal compiles into a durable proposal-plus-admission
// transaction only with the complete closure fields, and an action missing
// the objective or the resource ceiling — or carrying them on any other
// proposal kind — refuses before anything is committed.
func TestModelPrototypeAdmissionContract(t *testing.T) {
	action := prototypeImprovementAction(t)
	batch, err := CompileTransaction(context.Background(), nil, action)
	if err != nil || len(batch.Contents) != 2 || len(batch.Lineage) == 0 {
		t.Fatalf("prototype admission compile = (%d contents, %v)", len(batch.Contents), err)
	}

	unbudgeted := prototypeImprovementAction(t)
	unbudgeted.Improvement.GPUMinutes = 0
	if _, err := CompileTransaction(context.Background(), nil, unbudgeted); err == nil {
		t.Fatal("prototype admission without a resource ceiling compiled")
	}
	unobjective := prototypeImprovementAction(t)
	unobjective.Improvement.Objective = artifact.ID{}
	if _, err := CompileTransaction(context.Background(), nil, unobjective); err == nil {
		t.Fatal("prototype admission without an objective compiled")
	}
	misplaced := prototypeImprovementAction(t)
	misplaced.Improvement.Proposal.Kind = trainingprogram.ImprovementRecipe
	misplaced.Improvement.Proposal.Incumbent = testutil.ArtifactID(t, artifact.KindRecipe, "incumbent-recipe")
	misplaced.Improvement.Proposal.Candidate = testutil.ArtifactID(t, artifact.KindRecipe, "candidate-recipe")
	if _, err := CompileTransaction(context.Background(), nil, misplaced); err == nil ||
		!strings.Contains(err.Error(), "coherent") {
		t.Fatalf("prototype closure fields admitted on a recipe proposal: %v", err)
	}
}
