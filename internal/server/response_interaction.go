package server

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const (
	responseIdentityRadix = 10
	responseIdentityBits  = 64
)

func cloneResponseMessages(messages []inference.ChatMessage) []inference.ChatMessage {
	cloned := slices.Clone(messages)
	for index := range cloned {
		cloned[index].Media = slices.Clone(messages[index].Media)
		cloned[index].ToolCalls = slices.Clone(messages[index].ToolCalls)
	}
	return cloned
}

func interactionMessages(messages []inference.ChatMessage) []runrecord.InteractionMessage {
	result := make([]runrecord.InteractionMessage, len(messages))
	for index, message := range messages {
		converted := runrecord.InteractionMessage{
			Role: string(message.Role), Content: message.Content, ReasoningContent: message.ReasoningContent,
			Name: message.Name, ToolCallID: message.ToolCallID, ToolResultError: message.ToolResultError,
		}
		if len(message.Media) != 0 {
			converted.Media = make([]runrecord.InteractionMedia, len(message.Media))
		}
		for mediaIndex, media := range message.Media {
			converted.Media[mediaIndex] = runrecord.InteractionMedia{
				Type: string(media.Type), Data: media.Data, Format: media.Format,
				TextOffset: media.TextOffset, FPS: media.FPS,
			}
		}
		if len(message.ToolCalls) != 0 {
			converted.ToolCalls = make([]runrecord.InteractionToolCall, len(message.ToolCalls))
		}
		for toolIndex, call := range message.ToolCalls {
			converted.ToolCalls[toolIndex] = runrecord.InteractionToolCall{
				ID: call.ID, Type: string(call.Type), Name: call.Function.Name, Arguments: call.Function.Arguments,
			}
		}
		result[index] = converted
	}
	return result
}

func responseMessages(messages []runrecord.InteractionMessage) []inference.ChatMessage {
	result := make([]inference.ChatMessage, len(messages))
	for index, message := range messages {
		converted := inference.ChatMessage{
			Role: inference.ChatRole(message.Role), Content: message.Content, ReasoningContent: message.ReasoningContent,
			Name: message.Name, ToolCallID: message.ToolCallID, ToolResultError: message.ToolResultError,
		}
		if len(message.Media) != 0 {
			converted.Media = make([]inference.ChatMediaPart, len(message.Media))
		}
		for mediaIndex, media := range message.Media {
			converted.Media[mediaIndex] = inference.ChatMediaPart{
				Type: inference.ChatMediaType(media.Type), Data: media.Data, Format: media.Format,
				TextOffset: media.TextOffset, FPS: media.FPS,
			}
		}
		if len(message.ToolCalls) != 0 {
			converted.ToolCalls = make([]inference.ChatToolCall, len(message.ToolCalls))
		}
		for toolIndex, call := range message.ToolCalls {
			converted.ToolCalls[toolIndex] = inference.ChatToolCall{
				ID: call.ID, Type: inference.ChatToolType(call.Type),
				Function: inference.ChatToolFunction{Name: call.Name, Arguments: call.Arguments},
			}
		}
		result[index] = converted
	}
	return result
}

// publishResponseInteraction commits the durable record behind a
// response identifier. Failure surfaces to the caller: an identifier
// whose alias still names an older conversation must not be handed to
// a client as if it were durable.
func (h *Handler) publishResponseInteraction(
	ctx context.Context,
	responseID string,
	parent artifact.ID,
	messages []inference.ChatMessage,
) error {
	if h.repository == nil {
		return nil
	}
	description, ok := h.interactionDescription()
	if !ok {
		h.observationErrors.Add(counterStep)
		return errors.New("server: interaction authority is unavailable; the response cannot be stored")
	}
	_, err := runrecord.PublishInteraction(ctx, h.repository, runrecord.Interaction{
		Response: responseID, Recipe: description.Identity.Recipe, Model: description.Identity.Model,
		Node: description.Interaction.Node, Parent: parent,
	}, interactionMessages(messages))
	if err != nil {
		h.observationErrors.Add(counterStep)
		return fmt.Errorf("server: response interaction publication failed: %w", err)
	}
	return nil
}

func (h *Handler) loadResponseInteraction(ctx context.Context, responseID string) ([]inference.ChatMessage, artifact.ID, bool) {
	if h.repository == nil {
		return nil, artifact.ID{}, false
	}
	interaction, found, err := runrecord.ResolveInteraction(ctx, h.repository, responseID)
	if err != nil || !found {
		return nil, artifact.ID{}, false
	}
	description, admitted := h.interactionDescription()
	if !admitted || interaction.Recipe != description.Identity.Recipe {
		return nil, artifact.ID{}, false
	}
	messages, err := runrecord.VisibleInteractionMessages(ctx, h.repository, description.Interaction, interaction)
	if err != nil {
		return nil, artifact.ID{}, false
	}
	h.observeResponseID(responseID)
	restored := responseMessages(messages)
	// Durably replayed tool calls were model-produced when recorded;
	// they re-enter the issued set so a resumed conversation executes.
	for _, message := range restored {
		if message.Role == inference.ChatRoleAssistant && len(message.ToolCalls) != 0 {
			h.issuedCalls.record(message.ToolCalls...)
		}
	}
	return restored, interaction.ID, true
}

func (h *Handler) interactionDescription() (modelrecipe.RuntimeDescription, bool) {
	inspector, ok := h.generator.(interface {
		RecipeRuntimeDescription(recipe.Task) (modelrecipe.RuntimeDescription, error)
	})
	if !ok {
		return modelrecipe.RuntimeDescription{}, false
	}
	description, err := inspector.RecipeRuntimeDescription(recipe.TaskInference)
	return description, err == nil && description.Interaction.Valid()
}

// seedResponseIdentifiers advances the identifier counter past every
// durably recorded response, so a restarted server never reissues an
// identifier whose interaction alias already names an older
// conversation. Truncation of the alias listing is refused: a partial
// seed would silently reopen the collision.
func (h *Handler) seedResponseIdentifiers(ctx context.Context) error {
	if h.repository == nil {
		return nil
	}
	query := overgodb.Query{
		Kind: artifact.KindEvidence, MaxResults: h.config.MaxStoredResponses,
		Projection: overgodb.ProjectAliases,
	}
	for {
		result, err := h.repository.Query(ctx, query)
		if err != nil {
			return err
		}
		for _, alias := range result.Aliases {
			if response, found := strings.CutPrefix(alias.Name, runrecord.InteractionResponseAliasRoot); found {
				h.observeResponseID(response)
			}
		}
		if result.Next == nil {
			return nil
		}
		query.Cursor = result.Next
	}
}

func (h *Handler) observeResponseID(responseID string) {
	suffix, found := strings.CutPrefix(responseID, "resp_")
	if !found {
		return
	}
	value, err := strconv.ParseUint(suffix, responseIdentityRadix, responseIdentityBits)
	if err != nil {
		return
	}
	for current := h.nextID.Load(); current < value && !h.nextID.CompareAndSwap(current, value); current = h.nextID.Load() {
	}
}
