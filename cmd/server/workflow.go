package main

import (
	"context"
	"encoding/json"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/recipe"
	llamaserver "overgo/internal/server"
)

type serverRuntime struct {
	*inference.Runner
	llamaserver.WorkflowWorkspaceAPI
}

// PreviewWorkflow forwards to the workspaces the runtime serves.
func (runtime serverRuntime) PreviewWorkflow(ctx context.Context, kind llamaserver.WorkflowKind, task recipe.Task, recipeID artifact.ID, raw json.RawMessage, position, limit int) (any, error) {
	return llamaserver.PreviewWorkflowIn(ctx, runtime.WorkflowWorkspaceAPI, kind, task, recipeID, raw, position, limit)
}
