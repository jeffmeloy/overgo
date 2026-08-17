package main

import (
	"context"
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/recipe"
	llamaserver "overgo/internal/server"
)

type workflowStub struct {
	executed bool
	run      artifact.ID
}

func (stub *workflowStub) WorkflowCapabilities(context.Context, llamaserver.WorkflowKind) ([]llamaserver.WorkflowCapability, error) {
	return nil, nil
}

func (stub *workflowStub) ExecuteWorkflow(_ context.Context, _ llamaserver.WorkflowKind, _ recipe.Task, _ artifact.ID, _ json.RawMessage, _ operation.Reporter) (operation.Completion, error) {
	stub.executed = true
	return operation.Completion{Run: stub.run}, nil
}

func TestServerRuntimeExecutesDPOWorkflow(t *testing.T) {
	run, err := artifact.IdentifyBytes(artifact.KindRun, []byte("run"))
	if err != nil {
		t.Fatal(err)
	}
	stub := &workflowStub{run: run}
	runtime := &serverRuntime{WorkflowWorkspaceAPI: stub}
	var workspace llamaserver.WorkflowWorkspaceAPI = runtime
	if _, err := workspace.ExecuteWorkflow(context.Background(), llamaserver.WorkflowTraining, recipe.TaskTraining, artifact.ID{}, nil, nil); err != nil || !stub.executed {
		t.Fatalf("executed=%v err=%v", stub.executed, err)
	}
}
