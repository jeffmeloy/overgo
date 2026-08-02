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
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	formatter, ok := h.generator.(ChatFormatter)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "chat formatting is unavailable")
		return
	}
	var body responsesRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
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
	maxTokens := 16
	if body.MaxOutputTokens != nil {
		maxTokens = *body.MaxOutputTokens
	}
	if maxTokens < 0 || maxTokens > h.config.MaxTokens {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("max_output_tokens must be in [0,%d]", h.config.MaxTokens),
		)
		return
	}
	stops, err := parseStopSequences(body.Stop)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	samplingParams := body.samplingParameters
	if len(toolSelection.active) != 0 {
		provider, ok := h.generator.(ChatToolGrammarProvider)
		if !ok {
			writeError(
				response,
				http.StatusNotImplemented,
				"unsupported_operation",
				"tool-call grammar generation is unavailable",
			)
			return
		}
		source, root, patterns, grammarErr := provider.ChatToolGrammar(
			toolSelection.active,
			toolSelection.required,
			false,
			!toolSelection.named &&
				(body.ParallelTools == nil || *body.ParallelTools),
		)
		if grammarErr != nil {
			writeError(
				response,
				http.StatusBadRequest,
				"invalid_request_error",
				grammarErr.Error(),
			)
			return
		}
		samplingParams.Grammar = source
		samplingParams.GrammarRoot = root
		samplingParams.GrammarLazy = len(patterns) != 0
		samplingParams.GrammarTriggerPatterns = patterns
	}
	sampler, err := h.newSampler(samplingParams)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	slotID, acquired := h.acquireSlot(-1)
	if !acquired {
		response.Header().Set("Retry-After", "1")
		writeError(response, http.StatusTooManyRequests, "server_busy", "generation capacity is busy")
		return
	}
	defer h.releaseSlot(slotID)
	prepared, err := h.preparePrompt(request.Context(), normalizedPrompt, true)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
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

	var output strings.Builder
	filter := newStopFilter(stops)
	generatedTokens := 0
	ids, _, err := h.generate(
		request.Context(),
		slotID,
		prepared.Text,
		inference.GenerateOptions{
			MaxNewTokens:    maxTokens,
			Sampler:         sampler,
			ParseSpecial:    true,
			StopSequences:   stops,
			ContextShift:    h.config.ContextShift,
			PromptTokenIDs:  prepared.TokenIDs,
			CachePrompt:     prepared.ProjectedInputs != nil,
			ProjectedInputs: prepared.ProjectedInputs,
			OnToken: func(event inference.TokenEvent) error {
				generatedTokens++
				output.WriteString(filter.Accept(event.Piece))
				return nil
			},
		},
	)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	output.WriteString(filter.Flush())
	now := time.Now().Unix()
	message := inference.ChatMessage{
		Role:    "assistant",
		Content: output.String(),
	}
	if len(toolSelection.active) != 0 || reasoningSummary {
		message, err = h.generator.(ChatOutputParser).ParseChatOutput(
			output.String(),
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
	promptTokens := len(ids) - generatedTokens
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
			OutputTokens:      generatedTokens,
			TotalTokens:       promptTokens + generatedTokens,
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
	flusher, ok := response.(http.Flusher)
	if !ok {
		writeError(response, http.StatusInternalServerError, "server_error", "streaming is unavailable")
		return
	}
	response.Header().Set("Content-Type", "text/event-stream")
	response.Header().Set("Cache-Control", "no-cache")
	response.Header().Set("X-Accel-Buffering", "no")
	response.WriteHeader(http.StatusOK)
	writeEvent := func(name string, value any) error {
		if err := writeNamedSSE(response, name, value); err != nil {
			return err
		}
		flusher.Flush()
		return request.Context().Err()
	}
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
	filter := newStopFilter(stops)
	generatedTokens := 0
	textStarted := false
	var reasoningOutput *responseOutputItem
	var toolStream inference.ChatOutputStream
	streamedToolNames := make([]string, 0, 1)
	streamedToolArguments := make([]strings.Builder, 0, 1)
	if len(tools) != 0 {
		if provider, ok := h.generator.(ChatOutputStreamProvider); ok {
			var streamErr error
			toolStream, streamErr = provider.NewChatOutputStream(tools)
			if streamErr != nil {
				_ = writeEvent("response.failed", map[string]any{
					"type":  "response.failed",
					"error": errorEnvelope("generation_error", streamErr.Error()).Error,
				})
				return
			}
		}
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
		buffered.WriteString(piece)
		if toolStream == nil || piece == "" {
			return nil
		}
		deltas, streamErr := toolStream.Accept(piece)
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
			for len(streamedToolNames) <= delta.Index {
				streamedToolNames = append(streamedToolNames, "")
				streamedToolArguments = append(
					streamedToolArguments,
					strings.Builder{},
				)
			}
			outputIndex := delta.Index
			if textStarted {
				outputIndex++
			}
			itemID := fmt.Sprintf("fc_%s_%d", idSuffix, delta.Index)
			callID := fmt.Sprintf("call_%s_%d", idSuffix, delta.Index)
			if delta.Started {
				streamedToolNames[delta.Index] = delta.Name
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
				streamedToolArguments[delta.Index].WriteString(delta.Arguments)
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
	ids, _, err := h.generate(
		request.Context(),
		slotID,
		prompt,
		inference.GenerateOptions{
			MaxNewTokens:    maxTokens,
			Sampler:         sampler,
			ParseSpecial:    true,
			StopSequences:   stops,
			ContextShift:    h.config.ContextShift,
			PromptTokenIDs:  promptIDs,
			CachePrompt:     projectedInputs != nil,
			ProjectedInputs: projectedInputs,
			OnToken: func(event inference.TokenEvent) error {
				generatedTokens++
				piece := filter.Accept(event.Piece)
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
		},
	)
	if err != nil {
		_ = writeEvent("response.failed", map[string]any{
			"type":  "response.failed",
			"error": errorEnvelope("generation_error", err.Error()).Error,
		})
		return
	}
	flushed := filter.Flush()
	var parsedMessage inference.ChatMessage
	if len(tools) != 0 {
		if err := emitToolPiece(flushed); err != nil {
			_ = writeEvent("response.failed", map[string]any{
				"type":  "response.failed",
				"error": errorEnvelope("generation_error", err.Error()).Error,
			})
			return
		}
		parsedMessage, err = h.generator.(ChatOutputParser).ParseChatOutput(
			buffered.String(),
			tools,
		)
		if err != nil {
			_ = writeEvent("response.failed", map[string]any{
				"type":  "response.failed",
				"error": errorEnvelope("generation_error", err.Error()).Error,
			})
			return
		}
		if len(streamedToolNames) == 0 {
			if err := emitText(parsedMessage.Content); err != nil {
				return
			}
		}
	} else if reasoningSummary {
		buffered.WriteString(flushed)
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
		if err := emitText(flushed); err != nil {
			return
		}
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
		streamed := callIndex >= 0 && callIndex < len(streamedToolNames)
		if streamed {
			item.Arguments = streamedToolArguments[callIndex].String()
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
	promptTokens := len(ids) - generatedTokens
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
			OutputTokens:      generatedTokens,
			TotalTokens:       promptTokens + generatedTokens,
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
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		writeError(response, http.StatusMethodNotAllowed, "method_not_allowed", "POST required")
		return
	}
	formatter, ok := h.generator.(ChatFormatter)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "chat formatting is unavailable")
		return
	}
	_, ok = h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "token counting is unavailable")
		return
	}
	var body responsesTokenCountRequest
	if !h.decodeMultimodalJSON(response, request, &body) {
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
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
	prepared, err := h.preparePrompt(request.Context(), normalized, true)
	if err != nil {
		if nativePromptHasMedia(normalized) {
			writeGenerationError(response, err)
		} else {
			writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		}
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{
		"object":       "response.input_tokens",
		"input_tokens": len(prepared.TokenIDs),
	})
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
