package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
)

type trainingExportGenerator struct {
	*fakeGenerator
	capabilities map[WorkflowKind]WorkflowCapability
	runs         map[WorkflowKind]artifact.ID
	repository   artifact.Repository
	mu           sync.Mutex
	executed     []WorkflowKind
}

func (generator *trainingExportGenerator) WorkflowCapabilities(_ context.Context, kind WorkflowKind) ([]WorkflowCapability, error) {
	capability, ok := generator.capabilities[kind]
	if !ok {
		return nil, nil
	}
	return []WorkflowCapability{capability}, nil
}

func (generator *trainingExportGenerator) ExecuteWorkflow(
	ctx context.Context,
	kind WorkflowKind,
	_ recipe.Task,
	_ artifact.ID,
	_ json.RawMessage,
	reporter operation.Reporter,
) (operation.Completion, error) {
	generator.mu.Lock()
	generator.executed = append(generator.executed, kind)
	generator.mu.Unlock()
	if _, err := generator.repository.Commit(ctx, artifact.Batch{
		Key:       "fixture/workflow/" + generator.runs[kind].String(),
		Artifacts: []artifact.Descriptor{{ID: generator.runs[kind]}},
	}); err != nil {
		return operation.Completion{Run: generator.runs[kind]}, err
	}
	reporter.Publishing()
	return operation.Completion{Run: generator.runs[kind]}, nil
}

func TestGUITrainingAndExportShareServices(t *testing.T) {
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	capability := func(kind WorkflowKind) WorkflowCapability {
		return WorkflowCapability{
			Task:     recipe.TaskTraining,
			Recipe:   testutil.ArtifactID(t, artifact.KindRecipe, string(kind)+"-recipe"),
			Stages:   []recipe.Stage{{Node: recipe.Node{ID: "execute", Module: recipe.ModuleID(kind + ".execute")}}},
			Controls: []WorkflowControl{{Name: "source", Type: WorkflowControlText, Required: true}},
		}
	}
	generator := &trainingExportGenerator{
		fakeGenerator: &fakeGenerator{},
		repository:    store,
		capabilities: map[WorkflowKind]WorkflowCapability{
			WorkflowTraining: capability(WorkflowTraining),
			WorkflowExport:   capability(WorkflowExport),
		},
		runs: map[WorkflowKind]artifact.ID{
			WorkflowTraining: testutil.ArtifactID(t, artifact.KindRun, "training-run"),
			WorkflowExport:   testutil.ArtifactID(t, artifact.KindRun, "export-run"),
		},
	}
	handler, err := New(Config{Repository: store}, generator)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	for _, kind := range []WorkflowKind{WorkflowTraining, WorkflowExport} {
		capability := generator.capabilities[kind]
		listed := serveTestRequest(handler, http.MethodGet, "/"+string(kind)+"/capabilities", "")
		if listed.Code != http.StatusOK || !strings.Contains(listed.Body.String(), capability.Recipe.String()) {
			t.Fatalf("%s capabilities status=%d body=%s", kind, listed.Code, listed.Body.String())
		}
		body := `{"task":"training","recipe":"` + capability.Recipe.String() + `","input":{"source":"fixture"}}`
		accepted := serveTestRequest(handler, http.MethodPost, "/"+string(kind)+"/run", body)
		var submission workflowResponse
		if err := json.Unmarshal(accepted.Body.Bytes(), &submission); err != nil {
			t.Fatal(err)
		}
		waited := serveTestRequest(handler, http.MethodGet, "/operations/wait?id="+submission.Operation.String(), "")
		if waited.Code != http.StatusOK {
			t.Fatalf("%s wait status=%d body=%s", kind, waited.Code, waited.Body.String())
		}
	}
	generator.mu.Lock()
	executed := append([]WorkflowKind(nil), generator.executed...)
	generator.mu.Unlock()
	if len(executed) != 2 || executed[0] != WorkflowTraining || executed[1] != WorkflowExport {
		t.Fatalf("executed = %v", executed)
	}
	workflow := serveTestRequest(handler, http.MethodGet, "/workflow.js", "").Body.String()
	jobs := serveTestRequest(handler, http.MethodGet, "/mod/jobs.js", "").Body.String()
	if strings.Count(workflow, "capability.controls") != 1 || !strings.Contains(jobs, `scope: "training"`) ||
		!strings.Contains(jobs, `scope: "export"`) {
		t.Fatalf("workflow renderer or job registrations differ")
	}
}
