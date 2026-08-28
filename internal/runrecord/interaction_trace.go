package runrecord

import (
	"context"
	"errors"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	interactionTraceMediaType        = "application/vnd.overgo.interaction-trace+json"
	interactionTraceSchema           = "overgo/interaction-trace/v1"
	traceSequenceStart        uint32 = 1
)

// InteractionEventKind defines protocol-neutral trace event classes.
type InteractionEventKind string

const (
	// InteractionEventRequest marks model input.
	InteractionEventRequest InteractionEventKind = "request"
	// InteractionEventOutput marks model output.
	InteractionEventOutput InteractionEventKind = "output"
	// InteractionEventToolCall marks emitted tool work.
	InteractionEventToolCall InteractionEventKind = "tool-call"
	// InteractionEventToolResult marks returned tool work.
	InteractionEventToolResult InteractionEventKind = "tool-result"
)

// InteractionTraceEvent records one ordered protocol-neutral event.
type InteractionTraceEvent struct {
	Sequence uint32               `json:"sequence"`
	Kind     InteractionEventKind `json:"kind"`
	Branch   string               `json:"branch,omitempty"`
	Message  InteractionMessage   `json:"message"`
}

// InteractionTrace binds request, execution events, actions, decisions, and outputs.
type InteractionTrace struct {
	Version           uint16                  `json:"version"`
	Recipe            artifact.ID             `json:"recipe"`
	Model             artifact.ID             `json:"model"`
	Operation         artifact.ID             `json:"operation,omitzero"`
	Request           artifact.ID             `json:"request"`
	Events            []InteractionTraceEvent `json:"events"`
	ToolActions       []artifact.ID           `json:"tool_actions,omitempty"`
	Decisions         []artifact.ID           `json:"decisions,omitempty"`
	FinalArtifacts    []artifact.ID           `json:"final_artifacts,omitempty"`
	TaskContract      artifact.ID             `json:"task_contract,omitzero"`
	Strategy          artifact.ID             `json:"strategy,omitzero"`
	ToolManuals       []artifact.ID           `json:"tool_manuals,omitempty"`
	InvocationEffects []artifact.ID           `json:"invocation_effects,omitempty"`
	Obligations       []artifact.ID           `json:"obligations,omitempty"`
	Resolutions       []artifact.ID           `json:"resolutions,omitempty"`
	WorkspaceClaims   []artifact.ID           `json:"workspace_claims,omitempty"`
	Attempts          []artifact.ID           `json:"attempts,omitempty"`
	BudgetCharges     []artifact.ID           `json:"budget_charges,omitempty"`
	Terminal          Outcome                 `json:"terminal,omitempty"`
	ID                artifact.ID             `json:"-"`
}

// NewAgentTrajectory extends the common interaction trace with exact agent
// execution authorities. Payloads stay in their owning artifacts.
func NewAgentTrajectory(value InteractionTrace) (InteractionTrace, error) {
	value.ID = artifact.ID{}
	if value.TaskContract.Kind() != artifact.KindRecipe || !validOutcome(value.Terminal) {
		return InteractionTrace{}, errors.New("run record: invalid agent trajectory authority")
	}
	return interactionTraceCodec.New(value)
}

// Content returns the canonical committed bytes of the trace.
func (value InteractionTrace) Content() (artifact.Content, error) {
	return interactionTraceCodec.Content(value)
}

// Lineage links the trace to every valid referenced authority.
func (value InteractionTrace) Lineage() []artifact.Lineage {
	parents := []artifact.ID{value.Recipe, value.Model, value.Operation, value.Request, value.TaskContract, value.Strategy}
	for _, group := range [][]artifact.ID{value.ToolActions, value.Decisions, value.FinalArtifacts, value.ToolManuals,
		value.InvocationEffects, value.Obligations, value.Resolutions, value.WorkspaceClaims, value.Attempts, value.BudgetCharges} {
		parents = append(parents, group...)
	}
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	return artifact.DependencyLineage(value.ID, parents...)
}

var interactionTraceCodec = artifact.JSONDocumentCodec(
	"interaction trace", artifact.KindEvidence, interactionTraceMediaType, interactionTraceSchema,
	canonicalizeInteractionTrace,
	func(value InteractionTrace) artifact.ID { return value.ID },
	func(value *InteractionTrace, id artifact.ID) { value.ID = id },
	cloneInteractionTrace,
)

// RequireInteractionTrace loads one validated trace manifest.
func RequireInteractionTrace(ctx context.Context, reader artifact.Reader, id artifact.ID) (InteractionTrace, error) {
	return interactionTraceCodec.Require(ctx, reader, id)
}

// NewInteractionTrace identifies one complete protocol-neutral execution trace.
func NewInteractionTrace(value Interaction, request artifact.ID, messages []InteractionMessage, decisions []artifact.ID) (InteractionTrace, error) {
	events := make([]InteractionTraceEvent, len(messages))
	for index, message := range messages {
		events[index] = InteractionTraceEvent{
			Sequence: traceSequenceStart + uint32(index), Kind: interactionEventKind(message), Message: message,
		}
	}
	final := slices.Clone(value.Media)
	if value.Run.Valid() {
		final = append(final, value.Run)
	}
	return interactionTraceCodec.New(InteractionTrace{
		Version: artifact.InitialDocumentVersion, Recipe: value.Recipe, Model: value.Model,
		Operation: value.Operation, Request: request, Events: events,
		ToolActions: slices.Clone(value.Tools), Decisions: slices.Clone(decisions), FinalArtifacts: final,
	})
}

// ValidateIdentity verifies trace content identity.
func (value InteractionTrace) ValidateIdentity() error {
	return interactionTraceCodec.ValidateIdentity(value)
}

func canonicalizeInteractionTrace(value *InteractionTrace) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Model.Kind() != artifact.KindModel ||
		value.Request.Kind() != artifact.KindEvidence || len(value.Events) == 0 ||
		(value.Operation.Valid() && value.Operation.Kind() != artifact.KindEvidence) {
		return errors.New("run record: invalid interaction trace")
	}
	for index, event := range value.Events {
		if event.Sequence != traceSequenceStart+uint32(index) || !event.Kind.valid() ||
			event.Branch != "" && !textcheck.Bounded(event.Branch, len(event.Branch), "\x00\r\n") {
			return errors.New("run record: invalid interaction trace event")
		}
	}
	for _, ids := range [][]artifact.ID{value.ToolActions, value.Decisions, value.FinalArtifacts} {
		if slices.ContainsFunc(ids, func(id artifact.ID) bool { return !id.Valid() }) {
			return errors.New("run record: invalid interaction trace artifact")
		}
	}
	groups := []*[]artifact.ID{&value.ToolManuals, &value.InvocationEffects, &value.Obligations, &value.Resolutions,
		&value.WorkspaceClaims, &value.Attempts, &value.BudgetCharges}
	for _, ids := range groups {
		if len(*ids) > math.MaxUint16 || slices.ContainsFunc(*ids, func(id artifact.ID) bool { return !id.Valid() }) {
			return errors.New("run record: invalid bounded trajectory artifacts")
		}
		*ids = slices.Clone(*ids)
		sort.Slice(*ids, func(i, j int) bool { return artifact.CompareID((*ids)[i], (*ids)[j]) < 0 })
		*ids = slices.Compact(*ids)
	}
	if value.TaskContract.Valid() && (value.TaskContract.Kind() != artifact.KindRecipe || !validOutcome(value.Terminal)) {
		return errors.New("run record: invalid agent trajectory")
	}
	if value.Strategy.Valid() && value.Strategy.Kind() != artifact.KindProfile {
		return errors.New("run record: invalid trajectory strategy")
	}
	return nil
}

func validOutcome(outcome Outcome) bool {
	return outcome == OutcomeSucceeded || outcome == OutcomeFailed || outcome == OutcomeCancelled
}

func (kind InteractionEventKind) valid() bool {
	return kind == InteractionEventRequest || kind == InteractionEventOutput ||
		kind == InteractionEventToolCall || kind == InteractionEventToolResult
}

func interactionEventKind(message InteractionMessage) InteractionEventKind {
	if len(message.ToolCalls) != 0 {
		return InteractionEventToolCall
	}
	if message.Role == "tool" {
		return InteractionEventToolResult
	}
	if message.Role == "assistant" {
		return InteractionEventOutput
	}
	return InteractionEventRequest
}

func cloneInteractionTrace(value InteractionTrace) InteractionTrace {
	value.Events = slices.Clone(value.Events)
	for index := range value.Events {
		value.Events[index].Message.Media = slices.Clone(value.Events[index].Message.Media)
		value.Events[index].Message.ToolCalls = slices.Clone(value.Events[index].Message.ToolCalls)
	}
	value.ToolActions = slices.Clone(value.ToolActions)
	value.Decisions = slices.Clone(value.Decisions)
	value.FinalArtifacts = slices.Clone(value.FinalArtifacts)
	value.ToolManuals = slices.Clone(value.ToolManuals)
	value.InvocationEffects = slices.Clone(value.InvocationEffects)
	value.Obligations = slices.Clone(value.Obligations)
	value.Resolutions = slices.Clone(value.Resolutions)
	value.WorkspaceClaims = slices.Clone(value.WorkspaceClaims)
	value.Attempts = slices.Clone(value.Attempts)
	value.BudgetCharges = slices.Clone(value.BudgetCharges)
	return value
}
