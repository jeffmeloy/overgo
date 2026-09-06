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
	"overgo/internal/dataset"
	"overgo/internal/inference"
	"overgo/internal/invocation"
	"overgo/internal/operatoraction"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
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
	mu               sync.Mutex
	ID               string
	Interaction      artifact.ID
	Inspection       artifact.ID
	InspectionEffect artifact.ID
	Ceiling          artifact.ID
	Steps            int
	Contract         *ContractState
	Checkpoints      *MutationCheckpointRuntime
	// held is the last admitted stimulus boundary this session still holds;
	// nil for a stateless or resumed session, which rebuilds the bounded
	// full context instead of receiving a cursor delta.
	held *runrecord.AttemptStimulusBoundary
	// handoff is the context this session's latest attempt received: the
	// bounded full context or the verified stable cursor delta.
	handoff sessionContext
	// attempt is the durable invocation a resumed session replays under;
	// nil for a session that executes without the log.
	attempt *runrecord.DurableAttempt
}

// Coordinator admits and records agent tool steps.
type Coordinator struct {
	store    artifact.Repository
	executor *agenttool.Executor
	identity Identity
	maxSteps int
	// wakeups coalesces repeated late-stimulus notifications into one
	// pending reconciliation per consumed boundary.
	wakeups *workflowruntime.ReconcileCoalescer
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
	return &Coordinator{
		store: store, executor: executor, identity: identity, maxSteps: maxSteps,
		wakeups: workflowruntime.NewReconcileCoalescer(),
	}, nil
}

// ServingIdentity reports the authorities every recorded step binds.
func (c *Coordinator) ServingIdentity() Identity { return c.identity }

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
) (json.RawMessage, error) {
	return c.propose(ctx, session, name, arguments, nil)
}

// ProposeWithManuals admits a step only when the active agent binds the exact manual.
func (c *Coordinator) ProposeWithManuals(
	ctx context.Context,
	session *Session,
	name string,
	arguments json.RawMessage,
	manuals []artifact.ID,
) (json.RawMessage, error) {
	if len(manuals) == 0 {
		return nil, errors.New("agent loop: active agent has no tool authority")
	}
	return c.propose(ctx, session, name, arguments, manuals)
}

func (c *Coordinator) propose(
	ctx context.Context,
	session *Session,
	name string,
	arguments json.RawMessage,
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
	arguments, err = agenttool.CanonicalArguments(manual, arguments)
	if err != nil {
		return nil, err
	}
	plannedEffect, err := agenttool.DeriveInvocationEffect(manual, arguments, nil)
	if err != nil {
		return nil, err
	}
	if err := session.Contract.Admit(plannedEffect); err != nil {
		return nil, err
	}
	// A durable session reads a recorded step from its log before it
	// executes anything; the log refuses a stale invocation here.
	if replayed, found, err := c.replayChild(ctx, session, manual, arguments, plannedEffect); err != nil {
		return nil, err
	} else if found {
		return replayed, nil
	}
	callID := fmt.Sprintf("%s-step-%d", session.ID, session.Steps+1)
	var stimulus runrecord.AttemptStimulusBoundary
	var decision runrecord.HumanDecision
	if manual.Effect == agenttool.EffectMutation {
		stimulus, decision, err = c.requireMutationDecision(ctx, session, callID, manual, plannedEffect)
	} else {
		stimulus, err = c.admitAttemptStimulus(ctx, session, callID, manual, arguments, plannedEffect)
	}
	if err != nil {
		return nil, fmt.Errorf("agent loop: attempt preflight was not admitted: %w", err)
	}
	if session.handoff, err = incrementalSessionContext(session, stimulus); err != nil {
		return nil, fmt.Errorf("agent loop: attempt context handoff was not proven: %w", err)
	}
	// A mutation admits a durable receipt BEFORE it executes and closes
	// it after: if the receipt cannot persist the side effect never
	// happens, and if the process dies mid-execution the running receipt
	// is the evidence that it started -- a side effect never lacks a
	// record. Inspections are effect-free and carry no receipt. The
	// request's approval flag is intent, not authority: the gate
	// verifies the COMMITTED decision bound to this exact step, manual
	// identity, and argument bytes.
	var receiptOperation artifact.ID
	var receiptBinding *invocation.ReceiptBinding
	var checkpoint runrecord.AgentMutationCheckpoint
	if manual.Effect == agenttool.EffectMutation {
		binding := invocation.ReceiptBinding{
			Boundary: invocation.BoundaryAgent, Kind: invocation.MutationTool, Action: manual.Name,
			Subject: manual.ID, Arguments: stimulus.Arguments, Effect: stimulus.Effect,
			Preflight: stimulus.ID, Inspection: stimulus.Inspection, Ceiling: stimulus.Ceiling,
			Authority: decision.ID, CausalContext: stimulus.CausalContext, Head: stimulus.Selection.Head,
		}
		receiptBinding = &binding
		if receiptOperation, err = c.admitMutationReceipt(ctx, callID, manual, stimulus, binding); err != nil {
			return nil, fmt.Errorf("agent loop: mutation %q refused without a durable receipt: %w", name, err)
		}
		if session.Checkpoints != nil {
			checkpoint, err = session.Checkpoints.BeginCheckpoint(ctx, receiptOperation, plannedEffect, uint64(session.Steps))
			if err != nil {
				_, _ = c.closeMutationReceipt(ctx, receiptOperation, binding, nil, err)
				return nil, err
			}
		}
	}
	result, actualEffect, invokeErr := c.executor.InvokeWithEffect(ctx, manual, arguments)
	if checkpoint.ID.Valid() {
		checkpoint, err = session.Checkpoints.SealCheckpoint(context.WithoutCancel(ctx), checkpoint)
		if err != nil {
			invokeErr = errors.Join(invokeErr, err)
		}
	}
	if invokeErr == nil {
		session.Contract.ObserveMutation(actualEffect)
	}
	var receipt artifact.ID
	if receiptOperation.Valid() {
		receipt, err = c.closeMutationReceipt(ctx, receiptOperation, *receiptBinding, result, invokeErr)
		if err != nil {
			return nil, errors.Join(invokeErr, fmt.Errorf("agent loop: mutation receipt did not close: %w", err))
		}
	}
	if invokeErr != nil {
		return nil, invokeErr
	}
	interaction, err := c.recordStep(ctx, session, manual, arguments, result, receipt, checkpoint.ID, stimulus.ID)
	if err != nil {
		return nil, fmt.Errorf("agent loop: step executed but did not persist: %w", err)
	}
	if err := c.recordChild(ctx, session, interaction); err != nil {
		return nil, fmt.Errorf("agent loop: step executed but its invocation cannot complete it: %w", err)
	}
	session.Steps++
	if manual.Effect == agenttool.EffectInspection {
		session.Inspection, session.InspectionEffect = interaction, plannedEffect.ID
	} else {
		// A mutation changes the inspected state. Even an identical next action
		// needs a fresh, relevant inspection and a new bound preflight.
		session.Inspection, session.InspectionEffect = artifact.ID{}, artifact.ID{}
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
				if err == nil {
					if manual.Effect == agenttool.EffectMutation {
						session.Inspection, session.InspectionEffect = artifact.ID{}, artifact.ID{}
						continue
					}
					effect, effectErr := agenttool.DeriveInvocationEffect(manual, json.RawMessage(call.Arguments), nil)
					if effectErr != nil {
						return nil, effectErr
					}
					session.Inspection, session.InspectionEffect = interaction.ID, effect.ID
				}
			}
		}
	}
	return session, nil
}

// ApproveMutation durably records the operator's grant for the
// session's NEXT step: an approval request naming the exact manual and
// the exact argument bytes, answered granted, chained under the same
// per-step operation identity the mutation's receipt will use. The
// caller must pass the operation identity PreviewMutationDecision
// projected -- a grant that does not name what was previewed is
// refused, so the deciding action cannot mint its own approval. The
// proposal gate verifies THIS committed decision -- a request boolean
// asserts nothing on its own.
func (c *Coordinator) ApproveMutation(
	ctx context.Context,
	session *Session,
	name string,
	arguments json.RawMessage,
	approved artifact.ID,
) (artifact.ID, error) {
	if ctx == nil || session == nil || session.ID == "" {
		return artifact.ID{}, errors.New("agent loop: nil context or session")
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	manual, err := agenttool.ResolveRegisteredManual(ctx, c.store, name)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("agent loop: refusing approval of unregistered tool: %w", err)
	}
	if manual.Effect != agenttool.EffectMutation {
		return artifact.ID{}, fmt.Errorf("agent loop: %q is not a mutation; inspections need no approval", name)
	}
	arguments, err = agenttool.CanonicalArguments(manual, arguments)
	if err != nil {
		return artifact.ID{}, err
	}
	plannedEffect, err := agenttool.DeriveInvocationEffect(manual, arguments, nil)
	if err != nil {
		return artifact.ID{}, err
	}
	if err := session.Contract.Admit(plannedEffect); err != nil {
		return artifact.ID{}, err
	}
	if !session.Inspection.Valid() || !session.InspectionEffect.Valid() {
		return artifact.ID{}, fmt.Errorf("agent loop: mutation %q refused before an action-bound inspection", name)
	}
	inspectionEffect, err := invocation.RequireEffect(ctx, c.store, session.InspectionEffect)
	if err != nil {
		return artifact.ID{}, fmt.Errorf("agent loop: inspection effect is not durable: %w", err)
	}
	if !invocation.InspectionCovers(inspectionEffect, plannedEffect) {
		return artifact.ID{}, fmt.Errorf("agent loop: inspection is not relevant to mutation %q", name)
	}
	callID := fmt.Sprintf("%s-step-%d", session.ID, session.Steps+1)
	// The grant must name the previewed operation identity before anything
	// durable is admitted: approving and deciding stay two actions -- the
	// operator reads the preview, then grants exactly what it projected.
	previewed, err := MutationReceiptOperation(callID)
	if err != nil {
		return artifact.ID{}, err
	}
	if approved != previewed {
		return artifact.ID{}, fmt.Errorf(
			"agent loop: approval does not name the previewed operation for %q; preview the decision first", name,
		)
	}
	stimulus, err := c.admitAttemptStimulus(ctx, session, callID, manual, arguments, plannedEffect)
	if err != nil {
		return artifact.ID{}, err
	}
	operation := stimulus.Operation
	var prior artifact.ID
	if existing, found, err := runrecord.ResolveHumanDecision(ctx, c.store, operation); err != nil {
		return artifact.ID{}, err
	} else if found {
		prior = existing.ID
	}
	request, err := operatoraction.NewApprovalRequest(operation, c.identity.Recipe, operatoraction.Action{
		Code: manual.Name, Summary: "agent mutation " + manual.Name,
		Argv: decisionArguments(manual, stimulus),
	}, prior)
	if err != nil {
		return artifact.ID{}, err
	}
	decision, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
	if err != nil {
		return artifact.ID{}, err
	}
	if err := runrecord.PublishHumanDecision(ctx, c.store, request, decision); err != nil {
		return artifact.ID{}, err
	}
	return decision.ID, nil
}

// decisionArguments is the exact fact set a mutation decision binds: the
// immutable subject, canonical arguments, resolved effect, and preflight.
func decisionArguments(manual agenttool.Manual, stimulus runrecord.AttemptStimulusBoundary) []string {
	return invocation.ApprovalArguments(manual.ID, stimulus.Arguments, stimulus.Effect, stimulus.ID)
}

// requireMutationDecision requires one action-bound preflight and the committed
// grant for its exact subject, canonical arguments, effect, and inspection.
func (c *Coordinator) requireMutationDecision(
	ctx context.Context,
	session *Session,
	callID string,
	manual agenttool.Manual,
	plannedEffect agenttool.InvocationEffect,
) (runrecord.AttemptStimulusBoundary, runrecord.HumanDecision, error) {
	operation, err := MutationReceiptOperation(callID)
	if err != nil {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, err
	}
	stimulus, found, err := runrecord.ResolveAttemptStimulus(ctx, c.store, operation, uint32(artifact.InitialDocumentVersion))
	if err != nil {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, err
	}
	if !found {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, fmt.Errorf("agent loop: mutation %q has no action-bound preflight", manual.Name)
	}
	if stimulus.Manual != manual.ID || stimulus.Arguments != plannedEffect.Arguments || stimulus.Effect != plannedEffect.ID ||
		stimulus.Class != invocation.ClassMutation || stimulus.Inspection != session.Inspection ||
		stimulus.InspectionEffect != session.InspectionEffect || stimulus.CausalContext != session.Interaction ||
		stimulus.Ceiling != c.sessionCeiling(session) {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, fmt.Errorf("agent loop: mutation %q preflight binds different action facts", manual.Name)
	}
	inspectionEffect, err := invocation.RequireEffect(ctx, c.store, stimulus.InspectionEffect)
	if err != nil || !invocation.InspectionCovers(inspectionEffect, plannedEffect) {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, fmt.Errorf("agent loop: mutation %q preflight inspection is not relevant", manual.Name)
	}
	decision, found, err := runrecord.ResolveHumanDecision(ctx, c.store, operation)
	if err != nil {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, err
	}
	if !found {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, fmt.Errorf("agent loop: mutation %q has no committed approval decision", manual.Name)
	}
	if decision.Answer != operatoraction.AnswerGrant {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, fmt.Errorf("agent loop: mutation %q approval decision is %q", manual.Name, decision.Answer)
	}
	if decision.Tool != manual.Name || !slices.Equal(decision.Arguments, decisionArguments(manual, stimulus)) {
		return runrecord.AttemptStimulusBoundary{}, runrecord.HumanDecision{}, fmt.Errorf("agent loop: committed decision binds a different tool or arguments for %q", manual.Name)
	}
	return stimulus, decision, nil
}

// DecisionPreview reports what the mutation gate would see for the
// session's next step, so an operator decides on the exact facts: the
// step's derived operation, the manual's identity and effect, the
// argument bytes a grant would bind, any committed decision, and
// whether that decision binds these facts.
type DecisionPreview struct {
	CallID    string
	Operation artifact.ID
	Manual    artifact.ID
	Tool      string
	Effect    agenttool.Effect
	Arguments []string
	Decision  *runrecord.HumanDecision
	Binds     bool
}

// PreviewMutationDecision resolves the committed decision state for
// the session's next step without publishing or executing anything.
func (c *Coordinator) PreviewMutationDecision(
	ctx context.Context,
	session *Session,
	name string,
	arguments json.RawMessage,
) (DecisionPreview, error) {
	if ctx == nil || session == nil || session.ID == "" {
		return DecisionPreview{}, errors.New("agent loop: nil context or session")
	}
	session.mu.Lock()
	defer session.mu.Unlock()
	manual, err := agenttool.ResolveRegisteredManual(ctx, c.store, name)
	if err != nil {
		return DecisionPreview{}, fmt.Errorf("agent loop: refusing preview of unregistered tool: %w", err)
	}
	callID := fmt.Sprintf("%s-step-%d", session.ID, session.Steps+1)
	operation, err := MutationReceiptOperation(callID)
	if err != nil {
		return DecisionPreview{}, err
	}
	arguments, err = agenttool.CanonicalArguments(manual, arguments)
	if err != nil {
		return DecisionPreview{}, err
	}
	effect, err := agenttool.DeriveInvocationEffect(manual, arguments, nil)
	if err != nil {
		return DecisionPreview{}, err
	}
	preview := DecisionPreview{
		CallID: callID, Operation: operation, Manual: manual.ID,
		Tool: manual.Name, Effect: manual.Effect,
	}
	stimulus, preflightFound, err := runrecord.ResolveAttemptStimulus(ctx, c.store, operation, uint32(artifact.InitialDocumentVersion))
	if err != nil {
		return DecisionPreview{}, err
	}
	if preflightFound && stimulus.Manual == manual.ID && stimulus.Arguments == effect.Arguments && stimulus.Effect == effect.ID {
		preview.Arguments = decisionArguments(manual, stimulus)
	}
	decision, found, err := runrecord.ResolveHumanDecision(ctx, c.store, operation)
	if err != nil {
		return DecisionPreview{}, err
	}
	if found {
		preview.Decision = &decision
		preview.Binds = preflightFound && decision.Answer == operatoraction.AnswerGrant &&
			decision.Tool == manual.Name && slices.Equal(decision.Arguments, preview.Arguments)
	}
	return preview, nil
}

// MutationReceiptOperation derives the durable operation identity one
// mutation step's receipt chain lives under, so a reader can resolve
// the chain from the same call identity the interaction records.
func MutationReceiptOperation(callID string) (artifact.ID, error) {
	return artifact.IdentifyBytes(artifact.KindEvidence, []byte("overgo/agent-mutation/"+callID))
}

// admitAttemptStimulus freezes the exact pre-execution head and request bytes.
// Canonical bytes are also the conservative token ceiling when this tool
// boundary has no model tokenizer; callers can never undercount context by that
// substitution.
func (c *Coordinator) admitAttemptStimulus(
	ctx context.Context,
	session *Session,
	callID string,
	manual agenttool.Manual,
	arguments json.RawMessage,
	effect agenttool.InvocationEffect,
) (runrecord.AttemptStimulusBoundary, error) {
	operation, err := MutationReceiptOperation(callID)
	if err != nil {
		return runrecord.AttemptStimulusBoundary{}, err
	}
	effectContent, err := effect.Content()
	if err != nil {
		return runrecord.AttemptStimulusBoundary{}, err
	}
	content, err := runrecord.AttemptArgumentContent(arguments)
	if err != nil {
		return runrecord.AttemptStimulusBoundary{}, err
	}
	size := content.Descriptor.Size
	head, _ := c.store.Head()
	argumentContents := []artifact.Content{content}
	contents := slices.Concat(argumentContents, []artifact.Content{effectContent})
	documents := uint64(len(argumentContents))
	depth := uint32(len(argumentContents))
	sources := []dataset.InteractionSelectionSource{{
		Source: content.Descriptor.ID, CausalRoot: operation, Tokens: size, Bytes: size,
		Documents: documents, Depth: depth,
	}}
	selection, err := dataset.SelectInteractions(dataset.InteractionSelectionBounds{
		MaxTokens: size, MaxBytes: size, MaxDocuments: documents,
		MaxDepth: uint32(len(sources)), MaxResults: uint64(len(sources)),
	}, head, sources, nil)
	if err != nil {
		return runrecord.AttemptStimulusBoundary{}, err
	}
	var inspection, inspectionEffect artifact.ID
	if effect.Class == invocation.ClassMutation {
		inspection, inspectionEffect = session.Inspection, session.InspectionEffect
	}
	return runrecord.PublishAttemptStimulus(ctx, c.store, runrecord.AttemptStimulusBoundary{
		Operation: operation, Attempt: uint32(artifact.InitialDocumentVersion),
		Manual: manual.ID, Class: effect.Class, Arguments: effect.Arguments, Effect: effect.ID,
		Inspection: inspection, InspectionEffect: inspectionEffect,
		Ceiling: c.sessionCeiling(session), CausalContext: session.Interaction,
		Prior: session.Interaction, Selection: selection,
	}, contents)
}

func (c *Coordinator) sessionCeiling(session *Session) artifact.ID {
	if session != nil && session.Ceiling.Kind() == artifact.KindRecipe {
		return session.Ceiling
	}
	if session != nil && session.Contract != nil {
		return session.Contract.Authority()
	}
	return c.identity.Recipe
}

// admitMutationReceipt persists the admitted and running receipts for
// one mutation step before anything executes, binding the exact manual
// identity and the committed argument bytes as the receipt's inputs.
func (c *Coordinator) admitMutationReceipt(
	ctx context.Context,
	callID string,
	manual agenttool.Manual,
	stimulus runrecord.AttemptStimulusBoundary,
	binding invocation.ReceiptBinding,
) (artifact.ID, error) {
	operation, err := MutationReceiptOperation(callID)
	if err != nil {
		return artifact.ID{}, err
	}
	argumentsID := stimulus.Selection.Sources[0].Source
	base := runrecord.StageReceipt{
		Recipe: c.identity.Recipe, Node: c.identity.Node, Operation: operation,
		Attempt: uint32(artifact.InitialDocumentVersion), State: runrecord.StageAdmitted,
		Invocation: &binding,
		Inputs: []runrecord.StageBinding{
			{Port: "tool", Artifacts: []artifact.ID{manual.ID}},
			{Port: "arguments", Artifacts: []artifact.ID{argumentsID}},
			{Port: "stimulus", Artifacts: []artifact.ID{stimulus.ID}},
		},
	}
	descriptors := []artifact.Descriptor{{ID: operation}}
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
	binding invocation.ReceiptBinding,
	result json.RawMessage,
	invokeErr error,
) (artifact.ID, error) {
	base := runrecord.StageReceipt{
		Recipe: c.identity.Recipe, Node: c.identity.Node, Operation: operation,
		Attempt:    uint32(artifact.InitialDocumentVersion),
		Invocation: &binding,
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
	checkpoint artifact.ID,
	stimulus artifact.ID,
) (artifact.ID, error) {
	callID := fmt.Sprintf("%s-step-%d", session.ID, session.Steps+1)
	published, err := runrecord.PublishInteraction(ctx, c.store, runrecord.Interaction{
		Response: callID,
		Recipe:   c.identity.Recipe, Model: c.identity.Model, Node: c.identity.Node,
		Parent: session.Interaction, Run: receipt,
		Stimulus: stimulus,
		Tools:    slices.DeleteFunc([]artifact.ID{checkpoint}, func(id artifact.ID) bool { return !id.Valid() }),
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
		return artifact.ID{}, err
	}
	session.Interaction = published.ID
	return published.ID, nil
}
