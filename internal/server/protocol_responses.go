package server

import (
	"encoding/json"
	"errors"

	"fmt"

	"llamacpp2go/internal/inference"

	"llamacpp2go/internal/sampling"

	"llamacpp2go/internal/tokenizer"

	"net/http"

	"strconv"

	"strings"

	"time"
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
	Role      string                     `json:"role,omitempty"`
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
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	if !h.requireModel(response, body.Model) {
		return
	}
	previous, ok := h.previousResponseMessages(response, body.PreviousResponseID)
	if !ok {
		return
	}
	reasoningSummary, reasoningThinking, err := validateResponsesReasoning(body.Reasoning)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	toolSelection, err := selectResponsesTools(
		body.Tools,
		body.ToolChoice,
		body.ParallelTools,
		h.config.ResponseToolPolicy,
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if reasoningSummary && len(toolSelection.active) != 0 {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "reasoning summaries cannot be combined with tools")
		return
	}
	if len(toolSelection.active) != 0 {
		if _, ok := h.generator.(ChatOutputParser); !ok {
			writeError(
				response,
				http.StatusNotImplemented,
				"unsupported_operation",
				"tool-call output parsing is unavailable",
			)
			return
		}
	}
	if reasoningSummary {
		if _, ok := h.generator.(ChatOutputParser); !ok {
			writeError(response, http.StatusNotImplemented, "unsupported_operation", "reasoning output parsing is unavailable")
			return
		}
	}
	current, err := h.parseResponsesMessages(request.Context(), body.Input, "")
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	messages := responseRequestMessages(previous, current, body.Instructions)
	history := append(cloneResponseMessages(previous), current...)
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
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	maxTokens, err := boundedProtocolTokens(
		body.MaxOutputTokens, 16, h.config.MaxTokens, "max_output_tokens", false,
	)
	if err != nil {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			err.Error(),
		)
		return
	}
	stops, err := parseStopSequences(body.Stop)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
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
	sampler, err := h.newSampler(samplingParams)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	prepared, slotID, ok := h.prepareProtocolGeneration(response, request, normalizedPrompt)
	if !ok {
		return
	}
	defer h.releaseSlot(slotID)
	idNumber := h.nextID.Add(1)
	responseID := "resp_" + strconv.FormatUint(idNumber, 10)
	messageID := "msg_" + strconv.FormatUint(idNumber, 10)
	if body.Stream {
		h.streamResponses(
			response,
			request,
			slotID,
			prepared.Text,
			sampler,
			maxTokens,
			stops,
			responseID,
			messageID,
			toolSelection.active,
			prepared.TokenIDs,
			prepared.ProjectedInputs,
			history,
			body.Store == nil || *body.Store,
			reasoningSummary,
		)
		return
	}

	ids, pump, err := h.generateWithPump(
		request.Context(),
		slotID,
		prepared.Text,
		h.protocolGenerationOptions(maxTokens, sampler, prepared.TokenIDs, prepared.ProjectedInputs),
		stops,
		nil,
	)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	now := time.Now().Unix()
	message := inference.ChatMessage{
		Role:    "assistant",
		Content: pump.text(),
	}
	if len(toolSelection.active) != 0 || reasoningSummary {
		message, err = h.generator.(ChatOutputParser).ParseChatOutput(
			pump.text(),
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
	promptTokens := len(ids) - pump.generated
	if body.Store == nil || *body.Store {
		h.responseHistory.put(responseID, append(history, message))
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
			OutputTokens:      pump.generated,
			TotalTokens:       promptTokens + pump.generated,
			InputTokenDetails: responseInputTokenDetails{CachedTokens: 0},
		},
	})
}

func (h *Handler) streamResponses(
	response http.ResponseWriter,
	request *http.Request,
	slotID int,
	prompt string,
	sampler *sampling.Sampler,
	maxTokens int,
	stops []string,
	responseID, messageID string,
	tools []inference.ChatTool,
	promptIDs []tokenizer.TokenID,
	projectedInputs *inference.ProjectedInputs,
	history []inference.ChatMessage,
	store bool,
	reasoningSummary bool,
) {
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	writeEvent := stream.named
	inProgress := map[string]any{
		"id":     responseID,
		"object": "response",
		"status": "in_progress",
	}
	if err := writeEvent("response.created", map[string]any{
		"type":     "response.created",
		"response": inProgress,
	}); err != nil {
		return
	}
	if err := writeEvent("response.in_progress", map[string]any{
		"type":     "response.in_progress",
		"response": inProgress,
	}); err != nil {
		return
	}

	var output strings.Builder
	var buffered strings.Builder
	textStarted := false
	var reasoningOutput *responseOutputItem
	toolStream, streamErr := newToolDeltaStream(h.generator, tools)
	if streamErr != nil {
		_ = writeEvent("response.failed", map[string]any{
			"type":  "response.failed",
			"error": errorEnvelope("generation_error", streamErr.Error()).Error,
		})
		return
	}
	emitText := func(piece string) error {
		if piece == "" {
			return nil
		}
		if !textStarted {
			if err := writeEvent("response.output_item.added", map[string]any{
				"type": "response.output_item.added",
				"item": map[string]any{
					"content": []any{},
					"id":      messageID,
					"role":    "assistant",
					"status":  "in_progress",
					"type":    "message",
				},
			}); err != nil {
				return err
			}
			if err := writeEvent("response.content_part.added", map[string]any{
				"type":    "response.content_part.added",
				"item_id": messageID,
				"part": map[string]any{
					"type": "output_text",
					"text": "",
				},
			}); err != nil {
				return err
			}
			textStarted = true
		}
		output.WriteString(piece)
		return writeEvent("response.output_text.delta", map[string]any{
			"type":    "response.output_text.delta",
			"item_id": messageID,
			"delta":   piece,
		})
	}
	emitToolPiece := func(piece string) error {
		deltas, streamErr := toolStream.accept(piece)
		if streamErr != nil {
			return streamErr
		}
		idSuffix := strings.TrimPrefix(responseID, "resp_")
		for _, delta := range deltas {
			if delta.Content != "" {
				if streamErr := emitText(delta.Content); streamErr != nil {
					return streamErr
				}
			}
			outputIndex := delta.Index
			if textStarted {
				outputIndex++
			}
			itemID := fmt.Sprintf("fc_%s_%d", idSuffix, delta.Index)
			callID := fmt.Sprintf("call_%s_%d", idSuffix, delta.Index)
			if delta.Started {
				if streamErr := writeEvent(
					"response.output_item.added",
					map[string]any{
						"type":         "response.output_item.added",
						"response_id":  responseID,
						"output_index": outputIndex,
						"item": responseOutputItem{
							Arguments: "",
							CallID:    callID,
							ID:        itemID,
							Name:      delta.Name,
							Status:    "in_progress",
							Type:      "function_call",
						},
					},
				); streamErr != nil {
					return streamErr
				}
			}
			if delta.Arguments != "" {
				if streamErr := writeEvent(
					"response.function_call_arguments.delta",
					map[string]any{
						"type":         "response.function_call_arguments.delta",
						"response_id":  responseID,
						"item_id":      itemID,
						"output_index": outputIndex,
						"delta":        delta.Arguments,
					},
				); streamErr != nil {
					return streamErr
				}
			}
		}
		return nil
	}
	emitReasoning := func(text string) error {
		if text == "" {
			return nil
		}
		idSuffix := strings.TrimPrefix(responseID, "resp_")
		itemID := "rs_" + idSuffix
		part := responseReasoningSummary{Type: "summary_text", Text: text}
		added := responseOutputItem{ID: itemID, Status: "in_progress", Type: "reasoning"}
		if err := writeEvent("response.output_item.added", map[string]any{
			"type": "response.output_item.added", "response_id": responseID,
			"output_index": 0, "item": added,
		}); err != nil {
			return err
		}
		if err := writeEvent("response.reasoning_summary_part.added", map[string]any{
			"type": "response.reasoning_summary_part.added", "item_id": itemID,
			"output_index": 0, "summary_index": 0,
			"part": responseReasoningSummary{Type: "summary_text", Text: ""},
		}); err != nil {
			return err
		}
		if err := writeEvent("response.reasoning_summary_text.delta", map[string]any{
			"type": "response.reasoning_summary_text.delta", "item_id": itemID,
			"output_index": 0, "summary_index": 0, "delta": text,
		}); err != nil {
			return err
		}
		if err := writeEvent("response.reasoning_summary_text.done", map[string]any{
			"type": "response.reasoning_summary_text.done", "item_id": itemID,
			"output_index": 0, "summary_index": 0, "text": text,
		}); err != nil {
			return err
		}
		if err := writeEvent("response.reasoning_summary_part.done", map[string]any{
			"type": "response.reasoning_summary_part.done", "item_id": itemID,
			"output_index": 0, "summary_index": 0, "part": part,
		}); err != nil {
			return err
		}
		completed := responseOutputItem{
			ID: itemID, Status: "completed", Summary: []responseReasoningSummary{part}, Type: "reasoning",
		}
		if err := writeEvent("response.output_item.done", map[string]any{
			"type": "response.output_item.done", "response_id": responseID,
			"output_index": 0, "item": completed,
		}); err != nil {
			return err
		}
		reasoningOutput = &completed
		return nil
	}
	ids, pump, err := h.generateWithPump(
		request.Context(),
		slotID,
		prompt,
		h.protocolGenerationOptions(maxTokens, sampler, promptIDs, projectedInputs),
		stops,
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
		_ = writeEvent("response.failed", map[string]any{
			"type":  "response.failed",
			"error": errorEnvelope("generation_error", err.Error()).Error,
		})
		return
	}
	var parsedMessage inference.ChatMessage
	if len(tools) != 0 {
		parsedMessage, err = h.generator.(ChatOutputParser).ParseChatOutput(
			toolStream.text(),
			tools,
		)
		if err != nil {
			_ = writeEvent("response.failed", map[string]any{
				"type":  "response.failed",
				"error": errorEnvelope("generation_error", err.Error()).Error,
			})
			return
		}
		if !toolStream.started {
			if err := emitText(parsedMessage.Content); err != nil {
				return
			}
		}
	} else if reasoningSummary {
		parsedMessage, err = h.generator.(ChatOutputParser).ParseChatOutput(buffered.String(), nil)
		if err != nil {
			_ = writeEvent("response.failed", map[string]any{
				"type": "response.failed", "error": errorEnvelope("generation_error", err.Error()).Error,
			})
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
			Role:    "assistant",
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
			Role:    "assistant",
			Status:  "completed",
			Type:    "message",
		}
		if err := writeEvent("response.output_text.done", map[string]any{
			"type":    "response.output_text.done",
			"item_id": messageID,
			"text":    text,
		}); err != nil {
			return
		}
		if err := writeEvent("response.content_part.done", map[string]any{
			"type":    "response.content_part.done",
			"item_id": messageID,
			"part":    part,
		}); err != nil {
			return
		}
		if err := writeEvent("response.output_item.done", map[string]any{
			"type": "response.output_item.done",
			"item": item,
		}); err != nil {
			return
		}
		outputItems = append(outputItems, item)
	}
	idSuffix := strings.TrimPrefix(responseID, "resp_")
	assignResponseCallIDs(&parsedMessage, idSuffix)
	callItems := responseItems(
		inference.ChatMessage{
			Role:      "assistant",
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
			if err := writeEvent("response.output_item.added", map[string]any{
				"type":         "response.output_item.added",
				"response_id":  responseID,
				"output_index": outputIndex,
				"item":         added,
			}); err != nil {
				return
			}
			if err := writeEvent(
				"response.function_call_arguments.delta",
				map[string]any{
					"type":         "response.function_call_arguments.delta",
					"response_id":  responseID,
					"item_id":      item.ID,
					"output_index": outputIndex,
					"delta":        item.Arguments,
				},
			); err != nil {
				return
			}
		}
		if err := writeEvent(
			"response.function_call_arguments.done",
			map[string]any{
				"type":         "response.function_call_arguments.done",
				"response_id":  responseID,
				"item_id":      item.ID,
				"output_index": outputIndex,
				"arguments":    item.Arguments,
			},
		); err != nil {
			return
		}
		if err := writeEvent("response.output_item.done", map[string]any{
			"type":         "response.output_item.done",
			"response_id":  responseID,
			"output_index": outputIndex,
			"item":         item,
		}); err != nil {
			return
		}
		outputItems = append(outputItems, item)
	}
	now := time.Now().Unix()
	promptTokens := len(ids) - pump.generated
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
			OutputTokens:      pump.generated,
			TotalTokens:       promptTokens + pump.generated,
			InputTokenDetails: responseInputTokenDetails{CachedTokens: 0},
		},
	}
	if store {
		h.responseHistory.put(responseID, append(history, parsedMessage))
	}
	_ = writeEvent("response.completed", map[string]any{
		"type":     "response.completed",
		"response": final,
	})
}

func (h *Handler) responsesInputTokens(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	formatter, ok := h.requireChatFormatter(response)
	if !ok {
		return
	}
	if !h.requireTokenCounting(response) {
		return
	}
	var body responsesTokenCountRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	if !h.requireModel(response, body.Model) {
		return
	}
	previous, ok := h.previousResponseMessages(response, body.PreviousResponseID)
	if !ok {
		return
	}
	_, reasoningThinking, err := validateResponsesReasoning(body.Reasoning)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	toolSelection, err := selectResponsesTools(
		body.Tools,
		body.ToolChoice,
		body.ParallelTools,
		h.config.ResponseToolPolicy,
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	current, err := h.parseResponsesMessages(request.Context(), body.Input, "")
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
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
	normalized, err := h.normalizeChatPrompt(request.Context(), formatter, normalizedBody, toolSelection.prompt)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	h.writeProtocolInputTokenCount(response, request, normalized, true)
}

func (h *Handler) previousResponseMessages(
	response http.ResponseWriter,
	id string,
) ([]inference.ChatMessage, bool) {
	if id == "" {
		return nil, true
	}
	messages, ok := h.responseHistory.get(id)
	if !ok {
		writeError(response, http.StatusNotFound, "not_found_error", "previous response not found")
		return nil, false
	}
	return messages, true
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
		messages = append(messages, inference.ChatMessage{Role: "system", Content: instructions})
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
