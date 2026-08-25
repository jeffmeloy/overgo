package trainingworkflow

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
)

// TestPromoteObjectiveAdaptive pins the evidence-gated ladder climb: a
// declared objective promotes to adaptive-evidence only against a
// committed SUCCEEDED training session observation whose recipe
// grounds this exact objective; a fabricated observation is refused,
// and a second promotion is refused because the objective no longer
// sits at declared.
func TestPromoteObjectiveAdaptive(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelPath := filepath.Join(root, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}

	derived := func(kind artifact.Kind, role string) artifact.ID {
		id, err := artifact.IdentifyBytes(kind, []byte("overgo/test-objective/"+role))
		if err != nil {
			t.Fatal(err)
		}
		return id
	}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: "test-pair", Kind: trainingprogram.ObjectiveFlowMatching,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{recipecontract.ModalityText},
			Outputs: []recipecontract.Modality{recipecontract.ModalityVideo},
		},
		Dataset: derived(artifact.KindDataset, "dataset"), Split: derived(artifact.KindDatasetShard, "split"),
		Processors: []artifact.ID{derived(artifact.KindProfile, "processor")},
		Loss:       derived(artifact.KindProfile, "loss"), Evaluation: derived(artifact.KindProfile, "evaluation"),
		Metric: trainingprogram.MetricVideoPSNRUnit, Evidence: []artifact.ID{derived(artifact.KindEvidence, "note")},
		Authority: trainingprogram.ObjectiveDeclared,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := objective.Content()
	if err != nil {
		t.Fatal(err)
	}
	modelFile, err := os.Open(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	modelID, _, err := artifact.Identify(artifact.KindModel, modelFile)
	modelFile.Close()
	if err != nil {
		t.Fatal(err)
	}
	const alias = "objective.registered.test-pair"
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "test-objective", Contents: []artifact.Content{content},
		Artifacts: func() []artifact.Descriptor {
			descriptors := []artifact.Descriptor{
				{ID: modelID},
				{ID: objective.Dataset}, {ID: objective.Split}, {ID: objective.Processors[0]},
				{ID: objective.Loss}, {ID: objective.Evaluation}, {ID: objective.Evidence[0]},
			}
			for _, name := range []string{"precision", "placement", "memory", "checkpoint", "promotion"} {
				id, err := artifact.IdentifyBytes(artifact.KindProfile, []byte("overgo/training-bootstrap/"+name))
				if err != nil {
					t.Fatal(err)
				}
				descriptors = append(descriptors, artifact.Descriptor{ID: id})
			}
			return descriptors
		}(),
		Aliases: []artifact.AliasBinding{{Name: alias, Target: objective.ID}},
	}); err != nil {
		t.Fatal(err)
	}

	recipeID, err := BootstrapObjectiveRecipe(ctx, store, modelPath, alias)
	if err != nil {
		t.Fatal(err)
	}
	observer, err := NewObserver(store, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := observer.Admit(ctx, modelID, recipeID); err != nil {
		t.Fatal(err)
	}
	observation, err := observer.Finish(ctx, modelID, recipeID, nil, 1)
	if err != nil {
		t.Fatal(err)
	}

	fabricated := derived(artifact.KindEvidence, "fabricated-observation")
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{fabricated}); err == nil {
		t.Fatal("a fabricated observation was accepted as promotion evidence")
	}
	promoted, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{observation})
	if err != nil {
		t.Fatal(err)
	}
	if promoted.Authority != trainingprogram.ObjectiveAdaptive {
		t.Fatalf("promoted authority = %q", promoted.Authority)
	}
	bound, found, err := store.ResolveAlias(ctx, alias)
	if err != nil || !found || bound != promoted.ID {
		t.Fatalf("alias after promotion = (%s, %t, %v), want the promoted document", bound, found, err)
	}
	reloaded, err := trainingprogram.LoadObjective(ctx, store, promoted.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !containsID(reloaded.Evidence, observation) {
		t.Fatal("promoted objective does not carry the grounding observation in its evidence")
	}
	if _, err := PromoteObjectiveAdaptive(ctx, store, alias, []artifact.ID{observation}); err == nil {
		t.Fatal("an adaptive objective was promoted a second time from declared")
	}
}

func containsID(ids []artifact.ID, id artifact.ID) bool {
	for _, candidate := range ids {
		if candidate == id {
			return true
		}
	}
	return false
}
