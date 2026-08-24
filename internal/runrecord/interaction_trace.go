package runrecord

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
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
	Message  InteractionMessage   `json:"message"`
}

// InteractionTrace binds request, execution events, actions, decisions, and outputs.
type InteractionTrace struct {
	Version        uint16                  `json:"version"`
	Recipe         artifact.ID             `json:"recipe"`
	Model          artifact.ID             `json:"model"`
	Operation      artifact.ID             `json:"operation,omitzero"`
	Request        artifact.ID             `json:"request"`
	Events         []InteractionTraceEvent `json:"events"`
	ToolActions    []artifact.ID           `json:"tool_actions,omitempty"`
	Decisions      []artifact.ID           `json:"decisions,omitempty"`
	FinalArtifacts []artifact.ID           `json:"final_artifacts,omitempty"`
	ID             artifact.ID             `json:"-"`
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
		if event.Sequence != traceSequenceStart+uint32(index) || !event.Kind.valid() {
			return errors.New("run record: invalid interaction trace event")
		}
	}
	for _, ids := range [][]artifact.ID{value.ToolActions, value.Decisions, value.FinalArtifacts} {
		if slices.ContainsFunc(ids, func(id artifact.ID) bool { return !id.Valid() }) {
			return errors.New("run record: invalid interaction trace artifact")
		}
	}
	return nil
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
	return value
}
