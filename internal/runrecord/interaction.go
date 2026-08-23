package runrecord

import (
	"context"
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	// InteractionMediaType identifies interaction documents.
	InteractionMediaType = "application/vnd.overgo.interaction+json"
	// InteractionSchema identifies the interaction contract.
	InteractionSchema = "overgo/interaction/v2"
	// InteractionTranscriptMediaType identifies transcript documents.
	InteractionTranscriptMediaType = "application/vnd.overgo.interaction-transcript+json"
	// InteractionTranscriptSchema identifies the transcript contract.
	InteractionTranscriptSchema = "overgo/interaction-transcript/v1"
	// InteractionResponseAliasRoot scopes current response interactions.
	InteractionResponseAliasRoot = "interaction/response/"
)

var interactionCodec = artifact.JSONDocumentCodec(
	"interaction", artifact.KindEvidence, InteractionMediaType, InteractionSchema,
	canonicalizeInteraction,
	func(value Interaction) artifact.ID { return value.ID },
	func(value *Interaction, id artifact.ID) { value.ID = id },
	func(value Interaction) Interaction {
		value.Tools = slices.Clone(value.Tools)
		value.Media = slices.Clone(value.Media)
		return value
	},
)

var interactionTranscriptCodec = artifact.JSONDocumentCodec(
	"interaction transcript", artifact.KindEvidence, InteractionTranscriptMediaType, InteractionTranscriptSchema,
	canonicalizeInteractionTranscript,
	func(value InteractionTranscript) artifact.ID { return value.ID },
	func(value *InteractionTranscript, id artifact.ID) { value.ID = id },
	cloneInteractionTranscript,
)

// Interaction binds one response event to durable execution and message facts.
type Interaction struct {
	Version   uint16        `json:"version"`
	Response  string        `json:"response"`
	Recipe    artifact.ID   `json:"recipe,omitzero"`
	Model     artifact.ID   `json:"model"`
	Node      recipe.NodeID `json:"node"`
	Operation artifact.ID   `json:"operation,omitzero"`
	Run       artifact.ID   `json:"run,omitzero"`
	Parent    artifact.ID   `json:"parent,omitzero"`
	Message   artifact.ID   `json:"message"`
	Trace     artifact.ID   `json:"trace"`
	Tools     []artifact.ID `json:"tools,omitempty"`
	Media     []artifact.ID `json:"media,omitempty"`
	ID        artifact.ID   `json:"-"`
}

// InteractionTranscript stores ordered protocol-neutral messages.
type InteractionTranscript struct {
	Version  uint16               `json:"version"`
	Messages []InteractionMessage `json:"messages"`
	ID       artifact.ID          `json:"-"`
}

// InteractionMessage stores one durable conversation turn.
type InteractionMessage struct {
	Role             string                `json:"role"`
	Content          string                `json:"content"`
	ReasoningContent string                `json:"reasoning_content,omitempty"`
	Name             string                `json:"name,omitempty"`
	ToolCallID       string                `json:"tool_call_id,omitempty"`
	ToolResultError  bool                  `json:"is_error,omitempty"`
	Media            []InteractionMedia    `json:"media,omitempty"`
	ToolCalls        []InteractionToolCall `json:"tool_calls,omitempty"`
}

// InteractionMedia stores one message-local media reference.
type InteractionMedia struct {
	Type       string  `json:"type"`
	Data       string  `json:"data"`
	Format     string  `json:"format,omitempty"`
	TextOffset int     `json:"text_offset,omitempty"`
	FPS        float64 `json:"fps,omitempty"`
}

// InteractionToolCall stores one function invocation.
type InteractionToolCall struct {
	ID        string `json:"id,omitempty"`
	Type      string `json:"type"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

func canonicalizeInteraction(value *Interaction) error {
	if value.Version != artifact.InitialDocumentVersion || strings.TrimSpace(value.Response) != value.Response || value.Response == "" ||
		value.Recipe.Kind() != artifact.KindRecipe || value.Model.Kind() != artifact.KindModel || value.Node == "" ||
		value.Message.Kind() != artifact.KindEvidence || value.Trace.Kind() != artifact.KindEvidence ||
		(value.Parent.Valid() && value.Parent.Kind() != artifact.KindEvidence) {
		return errors.New("run record: invalid interaction")
	}
	for _, id := range append(slices.Clone(value.Tools), value.Media...) {
		if !id.Valid() {
			return errors.New("run record: invalid interaction attachment")
		}
	}
	return nil
}

// Activation returns this event's typed graph identity.
func (value Interaction) Activation() recipe.Activation {
	return recipe.Activation{Recipe: value.Recipe, Node: value.Node, Event: value.ID}
}

func canonicalizeInteractionTranscript(value *InteractionTranscript) error {
	if value.Version != artifact.InitialDocumentVersion || len(value.Messages) == 0 {
		return errors.New("run record: invalid interaction transcript")
	}
	for _, message := range value.Messages {
		if strings.TrimSpace(message.Role) == "" {
			return errors.New("run record: invalid interaction message")
		}
	}
	return nil
}

func cloneInteractionTranscript(value InteractionTranscript) InteractionTranscript {
	value.Messages = slices.Clone(value.Messages)
	for index := range value.Messages {
		value.Messages[index].Media = slices.Clone(value.Messages[index].Media)
		value.Messages[index].ToolCalls = slices.Clone(value.Messages[index].ToolCalls)
	}
	return value
}

// NewInteractionTranscript identifies one immutable message sequence.
func NewInteractionTranscript(messages []InteractionMessage) (InteractionTranscript, error) {
	return interactionTranscriptCodec.New(InteractionTranscript{
		Version: artifact.InitialDocumentVersion, Messages: messages,
	})
}

// RequireInteractionTranscript returns one validated message sequence.
func RequireInteractionTranscript(ctx context.Context, reader artifact.Reader, id artifact.ID) (InteractionTranscript, error) {
	return interactionTranscriptCodec.Require(ctx, reader, id)
}

// NewInteraction identifies one immutable response event.
func NewInteraction(value Interaction) (Interaction, error) {
	value.Version, value.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return interactionCodec.New(value)
}

// RequireInteraction returns one validated response event.
func RequireInteraction(ctx context.Context, reader artifact.Reader, id artifact.ID) (Interaction, error) {
	return interactionCodec.Require(ctx, reader, id)
}

// ResolveInteraction resolves a response identity to its durable event.
func ResolveInteraction(ctx context.Context, reader artifact.Reader, response string) (Interaction, bool, error) {
	id, found, err := artifact.ResolveAlias(ctx, reader, InteractionResponseAliasRoot+response)
	if err != nil || !found {
		return Interaction{}, found, err
	}
	value, err := RequireInteraction(ctx, reader, id)
	return value, err == nil, err
}

// VisibleInteractionMessages materializes one branch-isolated parent chain.
func VisibleInteractionMessages(
	ctx context.Context,
	reader artifact.Reader,
	scope recipe.InteractionScope,
	leaf Interaction,
) ([]InteractionMessage, error) {
	if ctx == nil || reader == nil || !scope.Valid() || leaf.Node != scope.Node {
		return nil, errors.New("run record: invalid interaction visibility request")
	}
	current := leaf.Activation()
	seen := make(map[artifact.ID]struct{})
	var lineage []Interaction
	for {
		activation := leaf.Activation()
		if _, found := seen[activation.Event]; found {
			return nil, errors.New("run record: interaction lineage contains cycle")
		}
		if !scope.Visibility.Allows(current, activation) {
			return nil, errors.New("run record: interaction lineage is not visible")
		}
		seen[activation.Event] = struct{}{}
		lineage = append(lineage, leaf)
		if !leaf.Parent.Valid() {
			break
		}
		next, err := RequireInteraction(ctx, reader, leaf.Parent)
		if err != nil {
			return nil, err
		}
		leaf = next
	}
	slices.Reverse(lineage)
	var messages []InteractionMessage
	for _, interaction := range lineage {
		transcript, err := RequireInteractionTranscript(ctx, reader, interaction.Message)
		if err != nil {
			return nil, err
		}
		messages = append(messages, transcript.Messages...)
	}
	return messages, nil
}

// PublishInteraction commits the transcript and event atomically.
func PublishInteraction(ctx context.Context, repository artifact.Repository, value Interaction, messages []InteractionMessage) (Interaction, error) {
	if ctx == nil || repository == nil {
		return Interaction{}, errors.New("run record: interaction repository is absent")
	}
	transcript, err := NewInteractionTranscript(messages)
	if err != nil {
		return Interaction{}, err
	}
	requestMessages := messages
	if len(messages) > 1 && messages[len(messages)-1].Role == "assistant" {
		requestMessages = messages[:len(messages)-1]
	}
	requestTranscript, err := NewInteractionTranscript(requestMessages)
	if err != nil {
		return Interaction{}, err
	}
	value.Message = transcript.ID
	var decisions []artifact.ID
	if value.Operation.Valid() {
		if decision, found, resolveErr := ResolveHumanDecision(ctx, repository, value.Operation); resolveErr != nil {
			return Interaction{}, resolveErr
		} else if found {
			decisions = append(decisions, decision.ID)
		}
	}
	trace, err := NewInteractionTrace(value, requestTranscript.ID, messages, decisions)
	if err != nil {
		return Interaction{}, err
	}
	value.Trace = trace.ID
	value, err = NewInteraction(value)
	if err != nil {
		return Interaction{}, err
	}
	transcriptContent, err := interactionTranscriptCodec.Content(transcript)
	if err != nil {
		return Interaction{}, err
	}
	interactionContent, err := interactionCodec.Content(value)
	if err != nil {
		return Interaction{}, err
	}
	traceContent, err := interactionTraceCodec.Content(trace)
	if err != nil {
		return Interaction{}, err
	}
	parents := []artifact.ID{value.Trace, value.Recipe, value.Operation, value.Run, value.Parent}
	parents = append(parents, value.Tools...)
	parents = append(parents, value.Media...)
	parents = slices.DeleteFunc(parents, func(id artifact.ID) bool { return !id.Valid() })
	contents := []artifact.Content{transcriptContent}
	if requestTranscript.ID != transcript.ID {
		requestContent, err := interactionTranscriptCodec.Content(requestTranscript)
		if err != nil {
			return Interaction{}, err
		}
		contents = append(contents, requestContent)
	}
	contents = append(contents, traceContent, interactionContent)
	lineage := artifact.DependencyLineage(value.ID, parents...)
	lineage = append(lineage, artifact.DependencyLineage(trace.ID, trace.Request)...)
	batch, err := artifact.NewDocumentBatch(
		"interaction/"+value.ID.String(),
		contents,
		lineage,
		[]artifact.AliasBinding{{Name: InteractionResponseAliasRoot + value.Response, Target: value.ID}},
	)
	if err != nil {
		return Interaction{}, err
	}
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return Interaction{}, err
	}
	return value, nil
}
