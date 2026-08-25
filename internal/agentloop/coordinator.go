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
	"slices"
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
	return c.propose(ctx, session, name, arguments, approved, nil)
}

// ProposeWithManuals admits a step only when the active agent binds the exact manual.
func (c *Coordinator) ProposeWithManuals(
	ctx context.Context,
	session *Session,
	name string,
	arguments json.RawMessage,
	approved bool,
	manuals []artifact.ID,
) (json.RawMessage, error) {
	if len(manuals) == 0 {
		return nil, errors.New("agent loop: active agent has no tool authority")
	}
	return c.propose(ctx, session, name, arguments, approved, manuals)
}

func (c *Coordinator) propose(
	ctx context.Context,
	session *Session,
	name string,
	arguments json.RawMessage,
	approved bool,
	manuals []artifact.ID,
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
	if manuals != nil && !slices.Contains(manuals, manual.ID) {
		return nil, fmt.Errorf("agent loop: tool %q is outside active agent authority", name)
	}
	// Invocation re-checks the committed argv policy: a manual published
	// before the policy tightened, or imported from another store, still
	// cannot run a program the current policy does not name.
	if err := agenttool.CheckArgvAuthority(ctx, c.store, manual); err != nil {
		return nil, err
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
	callID := fmt.Sprintf("%s-step-%d", session.ID, session.Steps+1)
	// A mutation admits a durable receipt BEFORE it executes and closes
	// it after: if the receipt cannot persist the side effect never
	// happens, and if the process dies mid-execution the running receipt
	// is the evidence that it started -- a side effect never lacks a
	// record. Inspections are effect-free and carry no receipt.
	var receiptOperation artifact.ID
	if manual.Effect == agenttool.EffectMutation {
		if receiptOperation, err = c.admitMutationReceipt(ctx, callID, manual, arguments); err != nil {
			return nil, fmt.Errorf("agent loop: mutation %q refused without a durable receipt: %w", name, err)
		}
	}
	result, invokeErr := c.executor.Invoke(ctx, manual, arguments)
	var receipt artifact.ID
	if receiptOperation.Valid() {
		receipt, err = c.closeMutationReceipt(ctx, receiptOperation, result, invokeErr)
		if err != nil {
			return nil, errors.Join(invokeErr, fmt.Errorf("agent loop: mutation receipt did not close: %w", err))
		}
	}
	if invokeErr != nil {
		return nil, invokeErr
	}
	if err := c.recordStep(ctx, session, manual, arguments, result, receipt); err != nil {
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

// MutationReceiptOperation derives the durable operation identity one
// mutation step's receipt chain lives under, so a reader can resolve
// the chain from the same call identity the interaction records.
func MutationReceiptOperation(callID string) (artifact.ID, error) {
	return artifact.IdentifyBytes(artifact.KindEvidence, []byte("overgo/agent-mutation/"+callID))
}

// admitMutationReceipt persists the admitted and running receipts for
// one mutation step before anything executes, binding the exact manual
// identity and the committed argument bytes as the receipt's inputs.
func (c *Coordinator) admitMutationReceipt(
	ctx context.Context,
	callID string,
	manual agenttool.Manual,
	arguments json.RawMessage,
) (artifact.ID, error) {
	operation, err := MutationReceiptOperation(callID)
	if err != nil {
		return artifact.ID{}, err
	}
	argumentsID, err := artifact.IdentifyBytes(artifact.KindEvidence, arguments)
	if err != nil {
		return artifact.ID{}, err
	}
	base := runrecord.StageReceipt{
		Recipe: c.identity.Recipe, Node: c.identity.Node, Operation: operation,
		Attempt: 1, State: runrecord.StageAdmitted,
		Inputs: []runrecord.StageBinding{
			{Port: "tool", Artifacts: []artifact.ID{manual.ID}},
			{Port: "arguments", Artifacts: []artifact.ID{argumentsID}},
		},
	}
	descriptors := []artifact.Descriptor{{ID: operation}, {ID: argumentsID}}
	if _, err := runrecord.PublishStageReceipt(ctx, c.store, base, nil, descriptors); err != nil {
		return artifact.ID{}, err
	}
	base.State = runrecord.StageRunning
	if _, err := runrecord.PublishStageReceipt(ctx, c.store, base, nil, nil); err != nil {
		return artifact.ID{}, err
	}
	return operation, nil
}

// closeMutationReceipt records the terminal state of one mutation
// step: completed with the committed result bytes as its output, or
// failed with the execution error preserved as the failure fact.
func (c *Coordinator) closeMutationReceipt(
	ctx context.Context,
	operation artifact.ID,
	result json.RawMessage,
	invokeErr error,
) (artifact.ID, error) {
	base := runrecord.StageReceipt{
		Recipe: c.identity.Recipe, Node: c.identity.Node, Operation: operation, Attempt: 1,
	}
	var descriptors []artifact.Descriptor
	if invokeErr != nil {
		base.State, base.Failure = runrecord.StageFailed, invokeErr.Error()
	} else {
		resultID, err := artifact.IdentifyBytes(artifact.KindEvidence, result)
		if err != nil {
			return artifact.ID{}, err
		}
		base.State = runrecord.StageCompleted
		base.Outputs = []runrecord.StageBinding{{Port: "result", Artifacts: []artifact.ID{resultID}}}
		descriptors = []artifact.Descriptor{{ID: resultID}}
	}
	published, err := runrecord.PublishStageReceipt(ctx, c.store, base, nil, descriptors)
	if err != nil {
		return artifact.ID{}, err
	}
	return published.ID, nil
}

// recordStep chains one durable interaction carrying the call and its
// result; the session advances to the new interaction so the chain
// stays walkable from the latest step back to the first.
func (c *Coordinator) recordStep(
	ctx context.Context,
	session *Session,
	manual agenttool.Manual,
	arguments, result json.RawMessage,
	receipt artifact.ID,
) error {
	callID := fmt.Sprintf("%s-step-%d", session.ID, session.Steps+1)
	published, err := runrecord.PublishInteraction(ctx, c.store, runrecord.Interaction{
		Response: callID,
		Recipe:   c.identity.Recipe, Model: c.identity.Model, Node: c.identity.Node,
		Parent: session.Interaction, Run: receipt,
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
