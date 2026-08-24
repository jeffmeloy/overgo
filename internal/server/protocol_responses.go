package server

import (
	"context"
	"encoding/json"
	"errors"

	"fmt"

	"overgo/internal/inference"

	"net/http"

	"strconv"

	"strings"

	"time"

	"overgo/internal/artifact"
)

type responsesTokenCountRequest struct {
	Model              string                    `json:"model"`
	Instructions       string                    `json:"instructions"`
	Input              json.RawMessage           `json:"input"`
	PreviousResponseID string                    `json:"previous_response_id"`
	Tools              json.RawMessage           `json:"tools"`
	ToolChoice         json.RawMessage           `json:"tool_choice"`
	ParallelTools      *bool                     `json:"parallel_tool_calls"`
	Reasoning          *responsesReasoningConfig `json:"reasoning"`
}

type responsesRequest struct {
	Model              string                    `json:"model"`
	Instructions       string                    `json:"instructions"`
	Input              json.RawMessage           `json:"input"`
	PreviousResponseID string                    `json:"previous_response_id"`
	MaxOutputTokens    *int                      `json:"max_output_tokens"`
	Stop               json.RawMessage           `json:"stop"`
	Stream             bool                      `json:"stream"`
	Tools              json.RawMessage           `json:"tools"`
	ToolChoice         json.RawMessage           `json:"tool_choice"`
	ParallelTools      *bool                     `json:"parallel_tool_calls"`
	Store              *bool                     `json:"store"`
	Reasoning          *responsesReasoningConfig `json:"reasoning"`
	samplingParameters
}

type responsesReasoningConfig struct {
	Effort          string `json:"effort,omitempty"`
	Summary         string `json:"summary,omitempty"`
	GenerateSummary string `json:"generate_summary,omitempty"`
}

type responseReasoningSummary struct {
	Text string `json:"text"`
	Type string `json:"type"`
}

type responseOutputText struct {
	Type        string `json:"type"`
	Annotations []any  `json:"annotations"`
	Logprobs    []any  `json:"logprobs"`
	Text        string `json:"text"`
}

type responseOutputItem struct {
	Arguments string                     `json:"arguments,omitempty"`
	CallID    string                     `json:"call_id,omitempty"`
	Content   []responseOutputText       `json:"content,omitempty"`
	ID        string                     `json:"id"`
	Name      string                     `json:"name,omitempty"`
	Role      inference.ChatRole         `json:"role,omitempty"`
	Status    string                     `json:"status,omitempty"`
	Summary   []responseReasoningSummary `json:"summary,omitempty"`
	Type      string                     `json:"type"`
}

type responseInputTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type responseUsage struct {
	InputTokens       int                       `json:"input_tokens"`
	OutputTokens      int                       `json:"output_tokens"`
	TotalTokens       int                       `json:"total_tokens"`
	InputTokenDetails responseInputTokenDetails `json:"input_tokens_details"`
}

type responsesResponse struct {
	CompletedAt int64                `json:"completed_at"`
	CreatedAt   int64                `json:"created_at"`
	ID          string               `json:"id"`
	Model       string               `json:"model"`
	Object      string               `json:"object"`
	Output      []responseOutputItem `json:"output"`
	Status      string               `json:"status"`
	Usage       responseUsage        `json:"usage"`
}

func (h *Handler) responses(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	formatter, ok := h.requireChatFormatter(response)
	if !ok {
		return
	}
	var body responsesRequest
	if !h.decodeProtocolJSON(response, request, &body, func() string { return body.Model }) {
		return
	}
	previous, parent, ok := h.previousResponseMessages(response, request.Context(), body.PreviousResponseID)
	if !ok {
		return
	}
	reasoningSummary, reasoningThinking, err := validateResponsesReasoning(body.Reasoning)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	toolSelection, err := selectResponsesTools(
		body.Tools,
		body.ToolChoice,
		body.ParallelTools,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	if reasoningSummary && len(toolSelection.active) != 0 {
		writeInvalidRequestMessage(response, "reasoning summaries cannot be combined with tools")
		return
	}
	parserFeature := ""
	if len(toolSelection.active) != 0 {
		parserFeature = "tool-call"
	} else if reasoningSummary {
		parserFeature = "reasoning"
	}
	var parser ChatOutputParser
	if parserFeature != "" {
		parser, ok = h.requireChatOutputParser(response, parserFeature)
		if !ok {
			return
		}
	}
	current, err := h.parseResponsesMessages(request.Context(), body.Input, "")
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	messages := responseRequestMessages(previous, current, body.Instructions)
	turn := cloneResponseMessages(current)
	multimodal := chatMediaCount(messages) != 0
	normalizedBody := chatCompletionRequest{Messages: messages, N: 1}
	if len(toolSelection.prompt) != 0 || body.Reasoning != nil {
		normalizedBody.TemplateKwargs = map[string]any{"enable_thinking": reasoningThinking}
	}
	if multimodal {
		normalizedBody.Tools = toolSelection.active
	}
	normalizedPrompt, err := h.normalizeChatPrompt(
		request.Context(),
		formatter,
		normalizedBody,
		toolSelection.prompt,
		true,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	maxTokens, err := boundedProtocolTokens(
		body.MaxOutputTokens, defaultProtocolMaxTokens, h.config.MaxTokens, "max_output_tokens", false,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	stops, err := parseStopSequences(body.Stop)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	samplingParams := body.samplingParameters
	if !h.configureChatToolGrammar(
		response,
		&samplingParams,
		toolSelection.active,
		toolSelection.required,
		false,
		!toolSelection.named &&
			(body.ParallelTools == nil || *body.ParallelTools),
	) {
		return
	}
	plan, ok := h.prepareProtocolGenerationPlan(
		response, request, normalizedPrompt, samplingParams, maxTokens, stops,
	)
	if !ok {
		return
	}
	defer plan.release()
	idNumber := h.nextID.Add(1)
	responseID := "resp_" + strconv.FormatUint(idNumber, 10)
	messageID := "msg_" + strconv.FormatUint(idNumber, 10)
	if body.Stream {
		h.streamResponses(
			response,
			request,
			plan,
			responseID,
			messageID,
			toolSelection.active,
			turn,
			parent,
			body.Store == nil || *body.Store,
			reasoningSummary,
			parser,
		)
		return
	}

	result, err := plan.run(nil)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	now := time.Now().Unix()
	message := inference.ChatMessage{
		Role:    inference.ChatRoleAssistant,
		Content: result.pump.text(),
	}
	if len(toolSelection.active) != 0 || reasoningSummary {
		message, err = parser.ParseChatOutput(
			result.pump.text(),
			toolSelection.active,
		)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
	}
	idSuffix := strings.TrimPrefix(responseID, "resp_")
	assignResponseCallIDs(&message, idSuffix)
	outputItems := responseItems(message, messageID, idSuffix, reasoningSummary)
	promptTokens := result.promptTokens()
	if body.Store == nil || *body.Store {
		h.publishResponseInteraction(context.WithoutCancel(request.Context()), responseID, parent, append(turn, message))
	}
	writeJSON(response, http.StatusOK, responsesResponse{
		CompletedAt: now,
		CreatedAt:   now,
		ID:          responseID,
		Model:       h.config.ModelID,
		Object:      "response",
		Output:      outputItems,
		Status:      "completed",
		Usage: responseUsage{
			InputTokens:       promptTokens,
			OutputTokens:      result.pump.generated,
			TotalTokens:       promptTokens + result.pump.generated,
			InputTokenDetails: responseInputTokenDetails{CachedTokens: 0},
		},
	})
}

func (h *Handler) streamResponses(
	response http.ResponseWriter,
	request *http.Request,
	plan *protocolGenerationPlan,
	responseID, messageID string,
	tools []inference.ChatTool,
	turn []inference.ChatMessage,
	parent artifact.ID,
	store bool,
	reasoningSummary bool,
	parser ChatOutputParser,
) {
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	writeEvent := stream.named
	inProgress := responsesProgress{ID: responseID, Object: "response", Status: "in_progress"}
	if err := writeEvent("response.created", responsesStreamEvent{
		Type: "response.created", Response: inProgress,
	}); err != nil {
		return
	}
	if err := writeEvent("response.in_progress", responsesStreamEvent{
		Type: "response.in_progress", Response: inProgress,
	}); err != nil {
		return
	}

	var output strings.Builder
	var buffered strings.Builder
	textStarted := false
	var reasoningOutput *responseOutputItem
	toolStream, streamErr := newToolDeltaStream(h.generator, tools)
	if streamErr != nil {
		_ = emitNamedGenerationError(writeEvent, "response.failed", streamErr)
		return
	}
	emitText := func(piece string) error {
		if piece == "" {
			return nil
		}
		if !textStarted {
			if err := writeEvent("response.output_item.added", responsesStreamEvent{
				Type: "response.output_item.added",
				Item: responsesMessageStart{
					Content: []any{}, ID: messageID, Role: inference.ChatRoleAssistant, Status: "in_progress", Type: "message",
				},
			}); err != nil {
				return err
			}
			if err := writeEvent("response.content_part.added", responsesStreamEvent{
				Type: "response.content_part.added", ItemID: messageID,
				Part: responsesTextStart{Type: "output_text", Text: ""},
			}); err != nil {
				return err
			}
			textStarted = true
		}
		output.WriteString(piece)
		return writeEvent("response.output_text.delta", responsesStreamEvent{
			Type: "response.output_text.delta", ItemID: messageID, Delta: piece,
		})
	}
	emitToolPiece := func(piece string) error {
		idSuffix := strings.TrimPrefix(responseID, "resp_")
		return toolStream.route(piece, toolDeltaSink{
			Content: emitText,
			Tool: func(delta inference.ChatToolCallDelta) error {
				outputIndex := delta.Index
				if textStarted {
					outputIndex++
				}
				itemID := fmt.Sprintf("fc_%s_%d", idSuffix, delta.Index)
				callID := fmt.Sprintf("call_%s_%d", idSuffix, delta.Index)
				if delta.Started {
					if err := writeEvent(
						"response.output_item.added",
						responsesStreamEvent{
							Type: "response.output_item.added", ResponseID: responseID,
							OutputIndex: eventIndex(outputIndex),
							Item: responseOutputItem{
								Arguments: "",
								CallID:    callID,
								ID:        itemID,
								Name:      delta.Name,
								Status:    "in_progress",
								Type:      "function_call",
							},
						},
					); err != nil {
						return err
					}
				}
				if delta.Arguments != "" {
					if err := writeEvent(
						"response.function_call_arguments.delta",
						responsesStreamEvent{
							Type: "response.function_call_arguments.delta", ResponseID: responseID,
							ItemID: itemID, OutputIndex: eventIndex(outputIndex), Delta: delta.Arguments,
						},
					); err != nil {
						return err
					}
				}
				return nil
			},
		})
	}
	emitReasoning := func(text string) error {
		if text == "" {
			return nil
		}
		idSuffix := strings.TrimPrefix(responseID, "resp_")
		itemID := "rs_" + idSuffix
		part := responseReasoningSummary{Type: "summary_text", Text: text}
		added := responseOutputItem{ID: itemID, Status: "in_progress", Type: "reasoning"}
		if err := writeEvent("response.output_item.added", responsesStreamEvent{
			Type: "response.output_item.added", ResponseID: responseID,
			OutputIndex: eventIndex(0), Item: added,
		}); err != nil {
			return err
		}
		if err := writeEvent("response.reasoning_summary_part.added", responsesStreamEvent{
			Type: "response.reasoning_summary_part.added", ItemID: itemID,
			OutputIndex: eventIndex(0), SummaryIndex: eventIndex(0),
			Part: responseReasoningSummary{Type: "summary_text", Text: ""},
		}); err != nil {
			return err
		}
		if err := writeEvent("response.reasoning_summary_text.delta", responsesStreamEvent{
			Type: "response.reasoning_summary_text.delta", ItemID: itemID,
			OutputIndex: eventIndex(0), SummaryIndex: eventIndex(0), Delta: text,
		}); err != nil {
			return err
		}
		if err := writeEvent("response.reasoning_summary_text.done", responsesStreamEvent{
			Type: "response.reasoning_summary_text.done", ItemID: itemID,
			OutputIndex: eventIndex(0), SummaryIndex: eventIndex(0), Text: eventString(text),
		}); err != nil {
			return err
		}
		if err := writeEvent("response.reasoning_summary_part.done", responsesStreamEvent{
			Type: "response.reasoning_summary_part.done", ItemID: itemID,
			OutputIndex: eventIndex(0), SummaryIndex: eventIndex(0), Part: part,
		}); err != nil {
			return err
		}
		completed := responseOutputItem{
			ID: itemID, Status: "completed", Summary: []responseReasoningSummary{part}, Type: "reasoning",
		}
		if err := writeEvent("response.output_item.done", responsesStreamEvent{
			Type: "response.output_item.done", ResponseID: responseID,
			OutputIndex: eventIndex(0), Item: completed,
		}); err != nil {
			return err
		}
		reasoningOutput = &completed
		return nil
	}
	result, err := plan.run(
		func(piece string) error {
			if len(tools) != 0 {
				if streamErr := emitToolPiece(piece); streamErr != nil {
					return streamErr
				}
				return request.Context().Err()
			}
			if reasoningSummary {
				buffered.WriteString(piece)
				return request.Context().Err()
			}
			return emitText(piece)
		},
	)
	if err != nil {
		_ = emitNamedGenerationError(writeEvent, "response.failed", err)
		return
	}
	var parsedMessage inference.ChatMessage
	if len(tools) != 0 {
		parsedMessage, err = toolStream.parse(parser, tools)
		if err != nil {
			_ = emitNamedGenerationError(writeEvent, "response.failed", err)
			return
		}
		if !toolStream.started {
			if err := emitText(parsedMessage.Content); err != nil {
				return
			}
		}
	} else if reasoningSummary {
		parsedMessage, err = parser.ParseChatOutput(buffered.String(), nil)
		if err != nil {
			_ = emitNamedGenerationError(writeEvent, "response.failed", err)
			return
		}
		if err := emitReasoning(parsedMessage.ReasoningContent); err != nil {
			return
		}
		if err := emitText(parsedMessage.Content); err != nil {
			return
		}
	} else {
		parsedMessage = inference.ChatMessage{
			Role:    inference.ChatRoleAssistant,
			Content: output.String(),
		}
	}
	text := output.String()
	outputItems := []responseOutputItem{}
	if reasoningOutput != nil {
		outputItems = append(outputItems, *reasoningOutput)
	}
	if textStarted {
		part := responseOutputText{
			Type:        "output_text",
			Annotations: []any{},
			Logprobs:    []any{},
			Text:        text,
		}
		item := responseOutputItem{
			Content: []responseOutputText{part},
			ID:      messageID,
			Role:    inference.ChatRoleAssistant,
			Status:  "completed",
			Type:    "message",
		}
		if err := writeEvent("response.output_text.done", responsesStreamEvent{
			Type: "response.output_text.done", ItemID: messageID, Text: eventString(text),
		}); err != nil {
			return
		}
		if err := writeEvent("response.content_part.done", responsesStreamEvent{
			Type: "response.content_part.done", ItemID: messageID, Part: part,
		}); err != nil {
			return
		}
		if err := writeEvent("response.output_item.done", responsesStreamEvent{
			Type: "response.output_item.done", Item: item,
		}); err != nil {
			return
		}
		outputItems = append(outputItems, item)
	}
	idSuffix := strings.TrimPrefix(responseID, "resp_")
	assignResponseCallIDs(&parsedMessage, idSuffix)
	callItems := responseItems(
		inference.ChatMessage{
			Role:      inference.ChatRoleAssistant,
			ToolCalls: parsedMessage.ToolCalls,
		},
		messageID,
		idSuffix,
		false,
	)
	for _, item := range callItems {
		outputIndex := len(outputItems)
		callIndex := outputIndex
		if textStarted {
			callIndex--
		}
		streamed := toolStream.streamed(callIndex)
		if streamed {
			item.Arguments = toolStream.argumentText(callIndex)
		}
		added := item
		added.Arguments = ""
		added.Status = "in_progress"
		if !streamed {
			if err := writeEvent("response.output_item.added", responsesStreamEvent{
				Type: "response.output_item.added", ResponseID: responseID,
				OutputIndex: eventIndex(outputIndex), Item: added,
			}); err != nil {
				return
			}
			if err := writeEvent(
				"response.function_call_arguments.delta",
				responsesStreamEvent{
					Type: "response.function_call_arguments.delta", ResponseID: responseID,
					ItemID: item.ID, OutputIndex: eventIndex(outputIndex), Delta: item.Arguments,
				},
			); err != nil {
				return
			}
		}
		if err := writeEvent(
			"response.function_call_arguments.done",
			responsesStreamEvent{
				Type: "response.function_call_arguments.done", ResponseID: responseID,
				ItemID: item.ID, OutputIndex: eventIndex(outputIndex), Arguments: eventString(item.Arguments),
			},
		); err != nil {
			return
		}
		if err := writeEvent("response.output_item.done", responsesStreamEvent{
			Type: "response.output_item.done", ResponseID: responseID,
			OutputIndex: eventIndex(outputIndex), Item: item,
		}); err != nil {
			return
		}
		outputItems = append(outputItems, item)
	}
	now := time.Now().Unix()
	promptTokens := result.promptTokens()
	final := responsesResponse{
		CompletedAt: now,
		CreatedAt:   now,
		ID:          responseID,
		Model:       h.config.ModelID,
		Object:      "response",
		Output:      outputItems,
		Status:      "completed",
		Usage: responseUsage{
			InputTokens:       promptTokens,
			OutputTokens:      result.pump.generated,
			TotalTokens:       promptTokens + result.pump.generated,
			InputTokenDetails: responseInputTokenDetails{CachedTokens: 0},
		},
	}
	if store {
		h.publishResponseInteraction(context.WithoutCancel(request.Context()), responseID, parent, append(turn, parsedMessage))
	}
	_ = writeEvent("response.completed", responsesStreamEvent{
		Type: "response.completed", Response: final,
	})
}

func (h *Handler) responsesInputTokens(response http.ResponseWriter, request *http.Request) {
	formatter, ok := h.requireProtocolTokenCounting(response, request)
	if !ok {
		return
	}
	var body responsesTokenCountRequest
	if !h.decodeProtocolJSON(response, request, &body, func() string { return body.Model }) {
		return
	}
	previous, _, ok := h.previousResponseMessages(response, request.Context(), body.PreviousResponseID)
	if !ok {
		return
	}
	_, reasoningThinking, err := validateResponsesReasoning(body.Reasoning)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	toolSelection, err := selectResponsesTools(
		body.Tools,
		body.ToolChoice,
		body.ParallelTools,
	)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	current, err := h.parseResponsesMessages(request.Context(), body.Input, "")
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	messages := responseRequestMessages(previous, current, body.Instructions)
	normalizedBody := chatCompletionRequest{Messages: messages, N: 1}
	if len(toolSelection.prompt) != 0 || body.Reasoning != nil {
		normalizedBody.TemplateKwargs = map[string]any{"enable_thinking": reasoningThinking}
	}
	if chatMediaCount(messages) != 0 {
		normalizedBody.Tools = toolSelection.active
	}
	normalized, err := h.normalizeChatPrompt(request.Context(), formatter, normalizedBody, toolSelection.prompt, false)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	h.writeProtocolInputTokenCount(response, request, normalized, true)
}

func (h *Handler) previousResponseMessages(
	response http.ResponseWriter,
	ctx context.Context,
	id string,
) ([]inference.ChatMessage, artifact.ID, bool) {
	if id == "" {
		return nil, artifact.ID{}, true
	}
	messages, parent, ok := h.loadResponseInteraction(ctx, id)
	if !ok {
		writeError(response, http.StatusNotFound, "not_found_error", "previous response not found")
		return nil, artifact.ID{}, false
	}
	return messages, parent, true
}

func responseRequestMessages(
	previous, current []inference.ChatMessage,
	instructions string,
) []inference.ChatMessage {
	capacity := len(previous) + len(current)
	if instructions != "" {
		capacity++
	}
	messages := make([]inference.ChatMessage, 0, capacity)
	if instructions != "" {
		messages = append(messages, inference.ChatMessage{Role: inference.ChatRoleSystem, Content: instructions})
	}
	messages = append(messages, previous...)
	messages = append(messages, current...)
	return messages
}

func validateResponsesReasoning(
	config *responsesReasoningConfig,
) (summary, thinking bool, err error) {
	if config == nil {
		return false, true, nil
	}
	switch config.Effort {
	case "", "minimal", "low", "medium", "high", "xhigh", "max":
		thinking = true
	case "none":
		thinking = false
	default:
		return false, false, fmt.Errorf("reasoning.effort %q is unsupported", config.Effort)
	}
	selected := config.Summary
	if selected == "" {
		selected = config.GenerateSummary
	} else if config.GenerateSummary != "" && config.GenerateSummary != selected {
		return false, false, errors.New("reasoning.summary conflicts with reasoning.generate_summary")
	}
	switch selected {
	case "":
		return false, thinking, nil
	case "auto", "concise", "detailed":
		return true, thinking, nil
	default:
		return false, false, fmt.Errorf("reasoning.summary %q is unsupported", selected)
	}
}
