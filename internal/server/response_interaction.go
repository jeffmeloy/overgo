package server

import (
	"context"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/inference"
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

func responseMessages(transcript runrecord.InteractionTranscript) []inference.ChatMessage {
	result := make([]inference.ChatMessage, len(transcript.Messages))
	for index, message := range transcript.Messages {
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

func (h *Handler) publishResponseInteraction(
	ctx context.Context,
	responseID string,
	parent artifact.ID,
	messages []inference.ChatMessage,
) {
	if h.repository == nil {
		return
	}
	_, recipeID, _ := h.servingIdentity(recipe.TaskInference)
	_, err := runrecord.PublishInteraction(ctx, h.repository, runrecord.Interaction{
		Response: responseID, Recipe: recipeID, Parent: parent,
	}, interactionMessages(messages))
	if err != nil {
		h.observationErrors.Add(counterStep)
	}
}

func (h *Handler) loadResponseInteraction(ctx context.Context, responseID string) ([]inference.ChatMessage, artifact.ID, bool) {
	if h.repository == nil {
		return nil, artifact.ID{}, false
	}
	interaction, found, err := runrecord.ResolveInteraction(ctx, h.repository, responseID)
	if err != nil || !found {
		return nil, artifact.ID{}, false
	}
	transcript, err := runrecord.RequireInteractionTranscript(ctx, h.repository, interaction.Message)
	if err != nil {
		return nil, artifact.ID{}, false
	}
	h.observeResponseID(responseID)
	return responseMessages(transcript), interaction.ID, true
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
