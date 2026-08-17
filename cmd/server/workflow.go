package main

import (
	"overgo/internal/inference"
	llamaserver "overgo/internal/server"
)

type serverRuntime struct {
	*inference.Runner
	llamaserver.WorkflowWorkspaceAPI
}
