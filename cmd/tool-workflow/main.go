// Command tool-workflow compiles and executes one closed-world tool
// workflow from the operator's CLI: the proposal names its steps and
// exact manual identities, every manual resolves from the store, the
// compiled graph owns ordering, and the run record owns execution
// evidence -- the same governed path an agent session uses, invoked
// directly by the operator.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"strings"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/workflowruntime"
)

type workflowRequest struct {
	Proposal  agenttool.ToolWorkflowProposal    `json:"proposal"`
	Arguments map[recipe.NodeID]json.RawMessage `json:"arguments"`
}

func main() {
	clioptions.MainNamed("tool-workflow", run)
}

func run() error {
	repository := flag.String("repo", "overgodb-store", "OvergoDB store directory")
	key := flag.String("key", "", "commit key recording this workflow execution")
	flag.Parse()
	if flag.NArg() != 1 || strings.TrimSpace(*key) == "" {
		return errors.New("usage: tool-workflow -key <commit-key> [-repo <store>] <request.json>")
	}
	var request workflowRequest
	if err := jsonfile.Decode(flag.Arg(0), &request); err != nil {
		return err
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	manuals := make([]agenttool.Manual, 0, len(request.Proposal.Steps))
	seen := map[artifact.ID]bool{}
	for _, step := range request.Proposal.Steps {
		if seen[step.Manual] {
			continue
		}
		seen[step.Manual] = true
		manual, err := agenttool.LoadManual(ctx, store, step.Manual)
		if err != nil {
			return fmt.Errorf("tool-workflow: step %q: %w", step.ID, err)
		}
		manuals = append(manuals, manual)
	}
	workflow, err := agenttool.CompileToolWorkflow(request.Proposal, manuals)
	if err != nil {
		return err
	}
	result, err := workflowruntime.ExecuteToolWorkflow(
		ctx, store, workflow, agenttool.NewOperatorExecutor(), *key, request.Arguments,
	)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}
