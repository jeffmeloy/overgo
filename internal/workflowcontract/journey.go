// Package workflowcontract exercises advertised recovery paths independently
// of CLI, GUI, or HTTP rendering. It is intentionally a verifier, not another
// workflow runtime: operation remains the status owner and registered handlers
// remain the execution owners.
package workflowcontract

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/operation"
	"overgo/internal/operatoraction"
)

type State string

const (
	StateBlocked   State = "blocked"
	StateReady     State = "ready"
	StateRunning   State = "running"
	StateCompleted State = "completed"
)

type Snapshot struct {
	Workflow string
	Subject  artifact.ID
	State    State
	Block    *operatoraction.Block
}

type Handler func(context.Context, Snapshot, operatoraction.Action) (Snapshot, error)
type Registry map[string]Handler

type Transition struct {
	Action string
	From   State
	To     State
}

// FromOperation projects the shared runtime status into the verifier. A block
// for any subject other than the operation's exact recipe is malformed rather
// than actionable.
func FromOperation(workflow string, status operation.Status) (Snapshot, error) {
	if strings.TrimSpace(workflow) != workflow || workflow == "" || status.State != operation.StateBlocked ||
		status.Recovery == nil || status.Recipe != status.Recovery.Subject {
		return Snapshot{}, errors.New("workflow contract: invalid blocked operation")
	}
	if err := status.Recovery.Validate(); err != nil {
		return Snapshot{}, fmt.Errorf("workflow contract: invalid operation recovery: %w", err)
	}
	block := status.Recovery.Clone()
	return Snapshot{Workflow: workflow, Subject: status.Recipe, State: StateBlocked, Block: &block}, nil
}

// ExerciseBlocked independently runs every advertised action through its
// registered handler. Each path must preserve the immutable subject and either
// leave the blocked state or produce a materially different valid block. This
// catches unregistered commands and self-looping recovery prose before an
// interface exposes them.
func ExerciseBlocked(ctx context.Context, start Snapshot, registry Registry) ([]Transition, error) {
	if ctx == nil || start.State != StateBlocked || start.Block == nil || start.Subject != start.Block.Subject {
		return nil, errors.New("workflow contract: invalid blocked journey start")
	}
	if err := start.Block.Validate(); err != nil {
		return nil, err
	}
	transitions := make([]Transition, 0, len(start.Block.Actions))
	for _, action := range start.Block.Actions {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		handler := registry[action.Code]
		if handler == nil {
			return nil, fmt.Errorf("workflow contract: recovery action %q is not registered", action.Code)
		}
		next, err := handler(ctx, cloneSnapshot(start), action)
		if err != nil {
			return nil, fmt.Errorf("workflow contract: recovery action %q failed: %w", action.Code, err)
		}
		if next.Workflow != start.Workflow || next.Subject != start.Subject || !validState(next.State) {
			return nil, fmt.Errorf("workflow contract: recovery action %q changed journey identity", action.Code)
		}
		if next.State == StateBlocked {
			if next.Block == nil || next.Block.Subject != next.Subject || next.Block.Validate() != nil ||
				reflect.DeepEqual(*next.Block, *start.Block) {
				return nil, fmt.Errorf("workflow contract: recovery action %q is a dead end", action.Code)
			}
		} else if next.Block != nil {
			return nil, fmt.Errorf("workflow contract: recovery action %q retained a block after progress", action.Code)
		}
		transitions = append(transitions, Transition{Action: action.Code, From: start.State, To: next.State})
	}
	return transitions, nil
}

func validState(state State) bool {
	return state == StateBlocked || state == StateReady || state == StateRunning || state == StateCompleted
}

func cloneSnapshot(snapshot Snapshot) Snapshot {
	if snapshot.Block != nil {
		block := snapshot.Block.Clone()
		snapshot.Block = &block
	}
	return snapshot
}
