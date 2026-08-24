// Package agentloop coordinates agent tool steps over the existing
// authorities and owns nothing they already own: agenttool holds the
// manuals and transports, workflowruntime and the executor run them,
// and runrecord interactions persist every step. The coordinator adds
// exactly the admission discipline between them: an unregistered tool
// refuses, inspection precedes mutation, mutation requires an exact
// approval, and a session cannot step without bound.
package agentloop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// maxSessionSteps bounds one agent session's tool steps: enough for a
// long investigation, small enough that a looping agent halts at a
// typed refusal instead of consuming the store and the operator's
// budget without end.
const maxSessionSteps = 64

// Identity names the serving authorities every recorded step binds to.
type Identity struct {
	Recipe artifact.ID
	Model  artifact.ID
	Node   recipe.NodeID
}

// Session is one agent conversation's admission state: its durable
// interaction chain plus the facts the mutation gate stands on.
type Session struct {
	ID          string
	Interaction artifact.ID
	Inspected   bool
	Steps       int
}

// Coordinator admits and records agent tool steps.
type Coordinator struct {
	store    artifact.Repository
	executor *agenttool.Executor
	identity Identity
}

// New binds a coordinator to the store, the transport executor, and
// the serving identity its interactions record.
func New(store artifact.Repository, executor *agenttool.Executor, identity Identity) (*Coordinator, error) {
	if store == nil || executor == nil {
		return nil, errors.New("agent loop: nil store or executor")
	}
	if identity.Recipe.Kind() != artifact.KindRecipe || identity.Model.Kind() != artifact.KindModel || identity.Node == "" {
		return nil, errors.New("agent loop: incomplete serving identity")
	}
	return &Coordinator{store: store, executor: executor, identity: identity}, nil
}

// Propose admits one tool step: the tool must be store-registered, a
// mutation must follow at least one completed inspection and carry an
// exact approval, and the session must have steps left. An admitted
// step executes over its declared transport and persists durably
// before the result returns.
func (c *Coordinator) Propose(
	ctx context.Context,
	session *Session,
	name string,
	arguments json.RawMessage,
	approved bool,
) (json.RawMessage, error) {
	if ctx == nil || session == nil || session.ID == "" {
		return nil, errors.New("agent loop: nil context or session")
	}
	if session.Steps >= maxSessionSteps {
		return nil, fmt.Errorf("agent loop: session %q reached its step bound", session.ID)
	}
	manual, err := agenttool.ResolveRegisteredManual(ctx, c.store, name)
	if err != nil {
		return nil, fmt.Errorf("agent loop: refusing unregistered tool: %w", err)
	}
	if manual.Effect == agenttool.EffectMutation {
		if !session.Inspected {
			return nil, fmt.Errorf(
				"agent loop: mutation %q refused before any completed inspection", name)
		}
		if !approved {
			return nil, fmt.Errorf(
				"agent loop: mutation %q requires an exact approval", name)
		}
	}
	result, err := c.executor.Invoke(ctx, manual, arguments)
	if err != nil {
		return nil, err
	}
	if err := c.recordStep(ctx, session, manual, arguments, result); err != nil {
		return nil, fmt.Errorf("agent loop: step executed but did not persist: %w", err)
	}
	session.Steps++
	if manual.Effect == agenttool.EffectInspection {
		session.Inspected = true
	}
	return result, nil
}

// recordStep chains one durable interaction carrying the call and its
// result; the session advances to the new interaction so the chain
// stays walkable from the latest step back to the first.
func (c *Coordinator) recordStep(
	ctx context.Context,
	session *Session,
	manual agenttool.Manual,
	arguments, result json.RawMessage,
) error {
	callID := fmt.Sprintf("%s-step-%d", session.ID, session.Steps+1)
	published, err := runrecord.PublishInteraction(ctx, c.store, runrecord.Interaction{
		Response: callID,
		Recipe:   c.identity.Recipe, Model: c.identity.Model, Node: c.identity.Node,
		Parent: session.Interaction,
	}, []runrecord.InteractionMessage{
		{
			Role: string(inference.ChatRoleAssistant),
			ToolCalls: []runrecord.InteractionToolCall{{
				ID: callID, Type: string(inference.ChatToolTypeFunction),
				Name: manual.Name, Arguments: string(arguments),
			}},
		},
		{Role: string(inference.ChatRoleTool), ToolCallID: callID, Content: string(result)},
	})
	if err != nil {
		return err
	}
	session.Interaction = published.ID
	return nil
}
