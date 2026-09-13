package server

import (
	"encoding/json"
	"net/http"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// launchWorkspaceGenerator is the shape cmd/server builds: a runner joined
// with the workflow workspaces the launch enabled, never the evaluation
// workspace, which the launch supplies through Config.Evaluation.
func launchWorkspaceGenerator(t testing.TB, kinds ...WorkflowKind) *trainingExportGenerator {
	t.Helper()
	capabilities := make(map[WorkflowKind]WorkflowCapability, len(kinds))
	for _, kind := range kinds {
		capabilities[kind] = WorkflowCapability{
			Task:     recipe.TaskTraining,
			Recipe:   testutil.ArtifactID(t, artifact.KindRecipe, string(kind)+"-launch-recipe"),
			Stages:   []recipe.Stage{{Node: recipe.Node{ID: "execute", Module: recipe.ModuleID(kind + ".execute")}}},
			Controls: []WorkflowControl{{Name: "source", Type: WorkflowControlText, Required: true}},
		}
	}
	return &trainingExportGenerator{fakeGenerator: &fakeGenerator{}, capabilities: capabilities}
}

// launchEvaluationWorkspace is the evaluation workspace a launch opens: one
// evaluable model with one compiled plan, enough for the tab to mount.
func launchEvaluationWorkspace(t testing.TB) *evaluationWorkspaceFixture {
	t.Helper()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "launch-evaluation-model")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "launch-evaluation-recipe")
	data, err := json.Marshal(evaluation.ExactSuite{
		Schema: "fixture/v1", Source: "fixture",
		Cases: []evaluation.ExactCase{{Name: "case", Prompt: "prompt", MaxTokens: 1, Text: "A", PromptTokens: 2, GeneratedTokens: 1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := evaluation.CompileSuite(data, evaluation.ExactAuthorities{
		ModelDefinition: testutil.ArtifactID(t, artifact.KindModelDefinition, "launch-evaluation-definition"),
		RuntimeRecipe:   recipeID, CodeCommit: evaluationTestCommit,
		Environment: testutil.ArtifactID(t, artifact.KindEvidence, "launch-evaluation-environment"),
		Execution:   evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &evaluationWorkspaceFixture{
		capability: EvaluationCapability{Model: modelID, Recipe: recipeID, Suite: compiled.Descriptor()},
		run:        testutil.ArtifactID(t, artifact.KindRun, "launch-evaluation-run"),
	}
}

func workspaceTabsByID(t testing.TB, handler *Handler) map[string]workspaceTab {
	t.Helper()
	response := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	var manifest workspaceManifestResponse
	if response.Code != http.StatusOK || json.Unmarshal(response.Body.Bytes(), &manifest) != nil {
		t.Fatalf("workspace manifest status=%d body=%s", response.Code, response.Body.String())
	}
	byID := make(map[string]workspaceTab, len(manifest.Tabs))
	for _, tab := range manifest.Tabs {
		byID[tab.ID] = tab
	}
	return byID
}

// TestWorkspaceManifestShippedLaunch: the manifest built the way cmd/server
// builds its generator lists Train, Model Builder and Evaluations when the
// launch enabled them, and refuses each with its reason when it did not.
func TestWorkspaceManifestShippedLaunch(t *testing.T) {
	enabled, err := New(Config{Evaluation: launchEvaluationWorkspace(t)}, launchWorkspaceGenerator(t, WorkflowTraining, WorkflowModelBuild))
	if err != nil {
		t.Fatal(err)
	}
	defer enabled.Close()
	tabs := workspaceTabsByID(t, enabled)
	for _, id := range []string{"training-jobs", "model-builder", "evaluations"} {
		if !tabs[id].Enabled || tabs[id].Refusal != "" {
			t.Fatalf("enabled launch refuses %s: %+v", id, tabs[id])
		}
	}
	if tabs["export-jobs"].Enabled || tabs["export-jobs"].Refusal == "" {
		t.Fatalf("workspace the launch did not enable is not refused: %+v", tabs["export-jobs"])
	}
	refused, err := New(Config{}, launchWorkspaceGenerator(t))
	if err != nil {
		t.Fatal(err)
	}
	defer refused.Close()
	tabs = workspaceTabsByID(t, refused)
	for _, id := range []string{"training-jobs", "model-builder", "evaluations"} {
		if tabs[id].Enabled || tabs[id].Refusal == "" {
			t.Fatalf("plain launch does not refuse %s with a reason: %+v", id, tabs[id])
		}
	}
}
