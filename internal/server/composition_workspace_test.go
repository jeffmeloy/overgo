package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/bridgegraph"
	"overgo/internal/composition"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

type compositionWorkflowGenerator struct {
	fakeGenerator
	inventory  CompositionInventory
	activation CompositionActivation
	activated  artifact.ID
	previous   *artifact.ID
}

func (generator *compositionWorkflowGenerator) CompositionInventory(context.Context) (CompositionInventory, error) {
	return generator.inventory, nil
}

func (generator *compositionWorkflowGenerator) ActivateComposition(
	_ context.Context,
	id artifact.ID,
	previous *artifact.ID,
) (CompositionActivation, error) {
	generator.activated, generator.previous = id, previous
	return generator.activation, nil
}

func TestCompositionAPIWorkflow(t *testing.T) {
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "server composition recipe")
	source := testutil.ArtifactID(t, artifact.KindModel, "server composition source")
	target := testutil.ArtifactID(t, artifact.KindModel, "server composition target")
	promotion := testutil.ArtifactID(t, artifact.KindEvidence, "server composition promotion")
	prior := testutil.ArtifactID(t, artifact.KindRecipe, "server prior composition")
	sourceContract := testutil.ArtifactID(t, artifact.KindProfile, "server source contract")
	targetContract := testutil.ArtifactID(t, artifact.KindProfile, "server target contract")
	bridgeDefinition := testutil.ArtifactID(t, artifact.KindProfile, "server bridge definition")
	bridgeWeights := testutil.ArtifactID(t, artifact.KindAdapter, "server bridge weights")
	execution := testutil.ArtifactID(t, artifact.KindRecipe, "server composition execution")
	trainingPolicy := testutil.ArtifactID(t, artifact.KindProfile, "server training policy")
	promotionPolicy := testutil.ArtifactID(t, artifact.KindProfile, "server promotion policy")
	heldOut := testutil.ArtifactID(t, artifact.KindDatasetShard, "server held-out split")
	regression := testutil.ArtifactID(t, artifact.KindDatasetShard, "server regression split")
	evaluator := testutil.ArtifactID(t, artifact.KindEvidence, "server evaluator")
	value := composition.CompositionRecipe{
		Version: artifact.InitialDocumentVersion, ID: recipeID,
		SourceModel: source, TargetModel: target, Task: recipe.TaskGeneration,
		SourceContract: sourceContract, TargetContract: targetContract,
		BridgeDefinition: bridgeDefinition, BridgeWeights: bridgeWeights,
		ExecutionRecipe: execution, TrainingPolicy: trainingPolicy,
		PromotionPolicy: promotionPolicy, Promotion: promotion,
	}
	generator := &compositionWorkflowGenerator{
		inventory: CompositionInventory{
			Sequence: uint64(artifact.InitialDocumentVersion),
			Candidates: []CompositionCandidate{{
				ID: recipeID, Compatible: true, Active: true, Recipe: &value,
				Graph: &bridgegraph.Definition{
					Source: sourceContract, Target: targetContract, Operator: bridgegraph.OperatorLinear,
				},
				Evidence: &CompositionEvidence{
					TrainingPolicy: trainingPolicy, PromotionPolicy: promotionPolicy,
					Promotion: promotion, HeldOutSplit: heldOut, RegressionSet: regression, Evaluator: evaluator,
				},
			}},
		},
		activation: CompositionActivation{Recipe: recipeID},
	}
	handler, err := New(Config{}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })

	response := serveTestRequest(handler, http.MethodGet, "/compositions", "")
	var inventory CompositionInventory
	if err := json.Unmarshal(response.Body.Bytes(), &inventory); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusOK || len(inventory.Candidates) != len(generator.inventory.Candidates) ||
		inventory.Candidates[0].ID != recipeID || inventory.Candidates[0].Evidence.Promotion != promotion ||
		inventory.Candidates[0].Graph.Operator != bridgegraph.OperatorLinear {
		t.Fatalf("composition inventory status=%d value=%+v", response.Code, inventory)
	}
	body, err := json.Marshal(compositionActivationRequest{Recipe: recipeID, Previous: &prior})
	if err != nil {
		t.Fatal(err)
	}
	activated := serveTestRequest(handler, http.MethodPost, "/compositions/activate", string(body))
	if activated.Code != http.StatusOK || generator.activated != recipeID ||
		generator.previous == nil || *generator.previous != prior {
		t.Fatalf("composition activation status=%d recipe=%s previous=%v", activated.Code, generator.activated, generator.previous)
	}
}

func TestCompositionGUIWorkflow(t *testing.T) {
	handler, err := New(Config{}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	index := serveTestRequest(handler, http.MethodGet, "/app.html", "").Body.String()
	module := serveTestRequest(handler, http.MethodGet, "/mod/composition.js", "").Body.String()
	if !strings.Contains(index, "/mod/composition.js") {
		t.Fatal("GUI shell does not load the composition workbench")
	}
	for _, token := range []string{
		"/compositions", "/compositions/activate", "training_policy", "held_out_split",
		"candidate.graph.operator", "candidate.runtime.components", "Activate",
	} {
		if !strings.Contains(module, token) {
			t.Errorf("composition GUI missing %q", token)
		}
	}
}
