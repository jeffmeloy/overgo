package server

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// TestAnUnvalidatedModelRunsWithItsStateShown: a media model the store holds as a registration leaves it, its candidate
// recipe published and nothing activated, is offered to run on the page with "not verified" beside it, and a run of it
// executes that candidate and publishes its output under it, as an activated model's run does.
func TestAnUnvalidatedModelRunsWithItsStateShown(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	location := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(location, []byte("oscillator-weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	model, candidate := candidateModelAt(t, store, location, oscillatorImageDefinition)
	var executed modelrecipe.CapabilityEvidenceSelection
	requestContract := artifact.JSONContract(artifact.KindFile, "overgo.test-unvalidated-input.v1")
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(_ context.Context, _ artifact.Repository, _ string, execution modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
			executed = execution
			request, err := requestContract.ContentBytes([]byte(raw))
			return capabilityruntime.Measured{Output: tinyPNG(t), Input: request}, err
		},
	}}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	handler, err := New(Config{Repository: store, Workflows: workspace}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	request := func(method, path string, body any, status int, out any) {
		t.Helper()
		var raw []byte
		if body != nil {
			if raw, err = json.Marshal(body); err != nil {
				t.Fatal(err)
			}
		}
		response := serveTestRequest(handler, method, path, string(raw))
		if response.Code != status {
			t.Fatalf("%s %s: %d %s", method, path, response.Code, response.Body)
		}
		if err := json.Unmarshal(response.Body.Bytes(), out); err != nil {
			t.Fatal(err)
		}
	}

	var listed []struct {
		Task         recipe.Task
		Recipe       artifact.ID
		Model        artifact.ID
		Refusal      string
		Verification modelrecipe.VerificationState
		Controls     []workflowControl
	}
	request(http.MethodGet, "/generation/capabilities", nil, http.StatusOK, &listed)
	if len(listed) != 1 || listed[0].Task != recipe.TaskImageGen || listed[0].Model != model || listed[0].Recipe != candidate.ID ||
		listed[0].Refusal != "" || listed[0].Verification != modelrecipe.StateNotVerified || len(listed[0].Controls) == 0 {
		t.Fatalf("listed %+v, want the candidate offered to run as not verified", listed)
	}

	var submission workflowResponse
	input := map[string]any{"class": 1, "seed": 7}
	request(http.MethodPost, "/generation/run", map[string]any{"task": recipe.TaskImageGen, "recipe": candidate.ID, "input": input}, http.StatusAccepted, &submission)
	var status operation.Status
	request(http.MethodGet, "/operations/wait?id="+submission.Operation.String(), nil, http.StatusOK, &status)
	if status.State != operation.StateCompleted || len(status.Outputs) != 1 || status.Run == nil {
		t.Fatalf("the unvalidated run ended %+v", status)
	}
	if executed.Program.Definition().ID != candidate.ID {
		t.Fatalf("the run executed %s, want the candidate %s", executed.Program.Definition().ID, candidate.ID)
	}
	run, err := runrecord.RequireRun(t.Context(), store, *status.Run)
	if err != nil || run.Recipe != candidate.ID || run.Outcome != runrecord.OutcomeSucceeded || len(run.Outputs) != 1 || run.Outputs[0] != status.Outputs[0] {
		t.Fatalf("run %+v, %v: want the candidate's output under it", run, err)
	}
	if active, err := modelrecipe.HasActiveRecipe(t.Context(), store, model, recipe.TaskImageGen); err != nil || active {
		t.Fatalf("running the candidate activated it: %v, %v", active, err)
	}
}
