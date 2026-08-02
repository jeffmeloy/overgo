package server

import (
	"encoding/json"

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
	Model              string          `json:"model"`
	Instructions       string          `json:"instructions"`
	Input              json.RawMessage `json:"input"`
	PreviousResponseID string          `json:"previous_response_id"`
	Tools              json.RawMessage `json:"tools"`
	ToolChoice         json.RawMessage `json:"tool_choice"`
	ParallelTools      *bool           `json:"parallel_tool_calls"`
}

type responsesRequest struct {
	Model              string          `json:"model"`
	Instructions       string          `json:"instructions"`
	Input              json.RawMessage `json:"input"`
	PreviousResponseID string          `json:"previous_response_id"`
	MaxOutputTokens    *int            `json:"max_output_tokens"`
	Stop               json.RawMessage `json:"stop"`
	Stream             bool            `json:"stream"`
	Tools              json.RawMessage `json:"tools"`
	ToolChoice         json.RawMessage `json:"tool_choice"`
	ParallelTools      *bool           `json:"parallel_tool_calls"`
	samplingParameters
}

type responseOutputText struct {
	Type        string `json:"type"`
	Annotations []any  `json:"annotations"`
	Logprobs    []any  `json:"logprobs"`
	Text        string `json:"text"`
}

type responseOutputItem struct {
	Arguments string               `json:"arguments,omitempty"`
	CallID    string               `json:"call_id,omitempty"`
	Content   []responseOutputText `json:"content,omitempty"`
	ID        string               `json:"id"`
	Name      string               `json:"name,omitempty"`
	Role      string               `json:"role,omitempty"`
	Status    string               `json:"status,omitempty"`
	Type      string               `json:"type"`
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
	if body.PreviousResponseID != "" {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "previous_response_id is not supported")
		return
	}
	toolSelection, err := selectResponsesTools(
		body.Tools,
		body.ToolChoice,
		body.ParallelTools,
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
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
	messages, err := parseResponsesMessages(body.Input, body.Instructions)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	multimodal := chatMediaCount(messages) != 0
	if multimodal {
		if rawJSONConfigured(body.Tools) || rawJSONConfigured(body.ToolChoice) || body.ParallelTools != nil {
			writeError(response, http.StatusBadRequest, "invalid_request_error", "multimodal Responses input cannot use tools")
			return
		}
	}
	normalizedBody := chatCompletionRequest{Messages: messages, N: 1}
	if len(toolSelection.prompt) != 0 {
		normalizedBody.TemplateKwargs = map[string]any{"enable_thinking": false}
	}
	normalizedPrompt, err := h.normalizeChatPrompt(
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
	if len(toolSelection.active) != 0 {
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
	outputItems := responseItems(message, messageID, idSuffix)
	promptTokens := len(ids) - generatedTokens
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
	callItems := responseItems(
		inference.ChatMessage{
			Role:      "assistant",
			ToolCalls: parsedMessage.ToolCalls,
		},
		messageID,
		idSuffix,
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
	if body.PreviousResponseID != "" {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"previous_response_id is not supported",
		)
		return
	}
	toolSelection, err := selectResponsesTools(
		body.Tools,
		body.ToolChoice,
		body.ParallelTools,
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	messages, err := parseResponsesMessages(body.Input, body.Instructions)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if chatMediaCount(messages) != 0 {
		if rawJSONConfigured(body.Tools) || rawJSONConfigured(body.ToolChoice) || body.ParallelTools != nil {
			writeError(response, http.StatusBadRequest, "invalid_request_error", "multimodal Responses input cannot use tools")
			return
		}
	}
	normalizedBody := chatCompletionRequest{Messages: messages, N: 1}
	if len(toolSelection.prompt) != 0 {
		normalizedBody.TemplateKwargs = map[string]any{"enable_thinking": false}
	}
	normalized, err := h.normalizeChatPrompt(formatter, normalizedBody, toolSelection.prompt)
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
