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
	"sync"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// Identity names the serving authorities every recorded step binds to.
type Identity struct {
	Recipe artifact.ID
	Model  artifact.ID
	Node   recipe.NodeID
}

// Session is one agent conversation's admission state: its durable
// interaction chain plus the facts the mutation gate stands on. A
// session serializes its own steps -- two concurrent proposals on one
// session would race the inspection gate and fork the chain.
type Session struct {
	mu          sync.Mutex
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
	maxSteps int
}

// New binds a coordinator to the store, the transport executor, and
// the serving identity its interactions record.
func New(store artifact.Repository, executor *agenttool.Executor, identity Identity, maxSteps int) (*Coordinator, error) {
	if store == nil || executor == nil || maxSteps <= 0 {
		return nil, errors.New("agent loop: store, executor, and positive step bound required")
	}
	if identity.Recipe.Kind() != artifact.KindRecipe || identity.Model.Kind() != artifact.KindModel || identity.Node == "" {
		return nil, errors.New("agent loop: incomplete serving identity")
	}
	return &Coordinator{store: store, executor: executor, identity: identity, maxSteps: maxSteps}, nil
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
	session.mu.Lock()
	defer session.mu.Unlock()
	if session.Steps >= c.maxSteps {
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

// RestoreSession rebuilds a session from its durable interaction
// chain: steps re-resolve in order until the first absent one, the tip
// becomes the parent for the next step, and the inspection fact
// recomputes from each recorded tool's registered effect -- so a
// restarted server neither forgets a session nor forges its state.
func (c *Coordinator) RestoreSession(ctx context.Context, id string) (*Session, error) {
	if ctx == nil || id == "" {
		return nil, errors.New("agent loop: nil context or empty session id")
	}
	session := &Session{ID: id}
	for step := 1; step <= c.maxSteps; step++ {
		interaction, found, err := runrecord.ResolveInteraction(ctx, c.store, fmt.Sprintf("%s-step-%d", id, step))
		if err != nil {
			return nil, err
		}
		if !found {
			break
		}
		session.Steps, session.Interaction = step, interaction.ID
		if session.Inspected {
			continue
		}
		transcript, err := runrecord.RequireInteractionTranscript(ctx, c.store, interaction.Message)
		if err != nil {
			return nil, err
		}
		for _, message := range transcript.Messages {
			for _, call := range message.ToolCalls {
				// Resolve the EXACT tool the step ran by its recorded
				// manual identity, so a later effect-class change on the
				// name alias cannot rewrite whether this step inspected.
				// Records written before manual.ID existed fall back to the
				// name alias, the best identity they carry.
				var manual agenttool.Manual
				if call.Manual.Valid() {
					manual, err = agenttool.LoadManual(ctx, c.store, call.Manual)
				} else {
					manual, err = agenttool.ResolveRegisteredManual(ctx, c.store, call.Name)
				}
				if err == nil && manual.Effect == agenttool.EffectInspection {
					session.Inspected = true
				}
			}
		}
	}
	return session, nil
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
				Name: manual.Name, Manual: manual.ID, Arguments: string(arguments),
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
