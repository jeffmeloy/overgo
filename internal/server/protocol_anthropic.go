package server

import (
	"encoding/json"

	"errors"

	"fmt"

	"llamacpp2go/internal/inference"

	"llamacpp2go/internal/sampling"

	"net/http"

	"strconv"

	"strings"
)

type anthropicTokenCountRequest struct {
	Model         string          `json:"model"`
	System        json.RawMessage `json:"system"`
	Messages      json.RawMessage `json:"messages"`
	MaxTokens     *int            `json:"max_tokens"`
	Temperature   *float32        `json:"temperature"`
	TopP          *float32        `json:"top_p"`
	TopK          *int            `json:"top_k"`
	StopSequences []string        `json:"stop_sequences"`
	Stream        bool            `json:"stream"`
	Tools         json.RawMessage `json:"tools"`
	ToolChoice    json.RawMessage `json:"tool_choice"`
	Thinking      json.RawMessage `json:"thinking"`
	Metadata      json.RawMessage `json:"metadata"`
}

type anthropicContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anthropicUsage struct {
	CacheReadInputTokens int `json:"cache_read_input_tokens"`
	InputTokens          int `json:"input_tokens"`
	OutputTokens         int `json:"output_tokens"`
}

type anthropicResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Content      []anthropicContentBlock `json:"content"`
	Model        string                  `json:"model"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        anthropicUsage          `json:"usage"`
}

func (h *Handler) anthropicMessages(response http.ResponseWriter, request *http.Request) {
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
	var body anthropicTokenCountRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if body.MaxTokens == nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "max_tokens is required")
		return
	}
	if *body.MaxTokens < 0 || *body.MaxTokens > h.config.MaxTokens {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			fmt.Sprintf("max_tokens must be in [0,%d]", h.config.MaxTokens),
		)
		return
	}
	if rawJSONConfigured(body.Thinking) {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"Anthropic thinking blocks are not supported",
		)
		return
	}
	toolSelection, err := selectAnthropicTools(body.Tools, body.ToolChoice)
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
	messages, err := parseAnthropicMessages(body.System, body.Messages)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	var prompt string
	if len(toolSelection.prompt) == 0 {
		prompt, err = formatter.FormatChat(messages)
	} else {
		prompt, err = formatChatRequest(
			formatter,
			messages,
			toolSelection.prompt,
			nil,
			map[string]any{"enable_thinking": false},
		)
	}
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	samplingParams := samplingParameters{
		Temperature: body.Temperature,
		TopP:        body.TopP,
		TopK:        body.TopK,
	}
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
			toolSelection.parallel,
		)
		if grammarErr != nil {
			writeError(response, http.StatusBadRequest, "invalid_request_error", grammarErr.Error())
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
	messageID := "msg_" + strconv.FormatUint(h.nextID.Add(1), 10)
	if body.Stream {
		h.streamAnthropicMessages(
			response,
			request,
			slotID,
			prompt,
			sampler,
			*body.MaxTokens,
			body.StopSequences,
			messageID,
			toolSelection.active,
		)
		return
	}
	var output strings.Builder
	filter := newStopFilter(body.StopSequences)
	generatedTokens := 0
	ids, _, err := h.generate(
		request.Context(),
		slotID,
		prompt,
		inference.GenerateOptions{
			MaxNewTokens:  *body.MaxTokens,
			Sampler:       sampler,
			ParseSpecial:  true,
			StopSequences: body.StopSequences,
			ContextShift:  h.config.ContextShift,
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
	stopReason := "end_turn"
	if !filter.Stopped() && generatedTokens >= *body.MaxTokens {
		stopReason = "max_tokens"
	}
	var stopSequence *string
	if filter.Stopped() {
		value := filter.StoppingWord()
		stopSequence = &value
	}
	message := inference.ChatMessage{Role: "assistant", Content: output.String()}
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
	content, err := anthropicBlocks(message, "toolu_"+strings.TrimPrefix(messageID, "msg_"))
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	if len(message.ToolCalls) != 0 {
		stopReason = "tool_use"
		stopSequence = nil
	}
	writeJSON(response, http.StatusOK, anthropicResponse{
		ID:           messageID,
		Type:         "message",
		Role:         "assistant",
		Content:      content,
		Model:        h.config.ModelID,
		StopReason:   stopReason,
		StopSequence: stopSequence,
		Usage: anthropicUsage{
			CacheReadInputTokens: 0,
			InputTokens:          len(ids) - generatedTokens,
			OutputTokens:         generatedTokens,
		},
	})
}

func (h *Handler) streamAnthropicMessages(
	response http.ResponseWriter,
	request *http.Request,
	slotID int,
	prompt string,
	sampler *sampling.Sampler,
	maxTokens int,
	stops []string,
	messageID string,
	tools []inference.ChatTool,
) {
	flusher, ok := response.(http.Flusher)
	if !ok {
		writeError(response, http.StatusInternalServerError, "server_error", "streaming is unavailable")
		return
	}
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "token counting is unavailable")
		return
	}
	promptIDs, err := tokenizerAPI.TokenizeText(prompt, true, true)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
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
	if err := writeEvent("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            messageID,
			"type":          "message",
			"role":          "assistant",
			"content":       []any{},
			"model":         h.config.ModelID,
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": anthropicUsage{
				CacheReadInputTokens: 0,
				InputTokens:          len(promptIDs),
				OutputTokens:         0,
			},
		},
	}); err != nil {
		return
	}
	filter := newStopFilter(stops)
	generatedTokens := 0
	textStarted := false
	var buffered strings.Builder
	emitText := func(piece string) error {
		if piece == "" {
			return nil
		}
		if !textStarted {
			if err := writeEvent("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": 0,
				"content_block": map[string]any{
					"type": "text",
					"text": "",
				},
			}); err != nil {
				return err
			}
			textStarted = true
		}
		return writeEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]any{
				"type": "text_delta",
				"text": piece,
			},
		})
	}
	_, _, err = h.generate(
		request.Context(),
		slotID,
		prompt,
		inference.GenerateOptions{
			MaxNewTokens:  maxTokens,
			Sampler:       sampler,
			ParseSpecial:  true,
			StopSequences: stops,
			ContextShift:  h.config.ContextShift,
			OnToken: func(event inference.TokenEvent) error {
				generatedTokens++
				piece := filter.Accept(event.Piece)
				if len(tools) != 0 {
					buffered.WriteString(piece)
					return request.Context().Err()
				}
				return emitText(piece)
			},
		},
	)
	if err != nil {
		_ = writeEvent("error", map[string]any{
			"type":  "error",
			"error": errorEnvelope("generation_error", err.Error()).Error,
		})
		return
	}
	flushed := filter.Flush()
	if len(tools) != 0 {
		buffered.WriteString(flushed)
	} else {
		if err := emitText(flushed); err != nil {
			return
		}
	}
	stopReason := "end_turn"
	if !filter.Stopped() && generatedTokens >= maxTokens {
		stopReason = "max_tokens"
	}
	if len(tools) != 0 {
		message, parseErr := h.generator.(ChatOutputParser).ParseChatOutput(
			buffered.String(),
			tools,
		)
		if parseErr != nil {
			_ = writeEvent("error", map[string]any{
				"type":  "error",
				"error": errorEnvelope("generation_error", parseErr.Error()).Error,
			})
			return
		}
		blocks, blockErr := anthropicBlocks(
			message,
			"toolu_"+strings.TrimPrefix(messageID, "msg_"),
		)
		if blockErr != nil {
			_ = writeEvent("error", map[string]any{
				"type":  "error",
				"error": errorEnvelope("generation_error", blockErr.Error()).Error,
			})
			return
		}
		for index, block := range blocks {
			startBlock := map[string]any{
				"type": block.Type,
			}
			if block.Type == "text" {
				startBlock["text"] = ""
			} else {
				startBlock["id"] = block.ID
				startBlock["name"] = block.Name
				startBlock["input"] = map[string]any{}
			}
			if err := writeEvent("content_block_start", map[string]any{
				"type":          "content_block_start",
				"index":         index,
				"content_block": startBlock,
			}); err != nil {
				return
			}
			var delta map[string]any
			if block.Type == "text" {
				delta = map[string]any{
					"type": "text_delta",
					"text": block.Text,
				}
			} else {
				delta = map[string]any{
					"type":         "input_json_delta",
					"partial_json": string(block.Input),
				}
			}
			if err := writeEvent("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": index,
				"delta": delta,
			}); err != nil {
				return
			}
			if err := writeEvent("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": index,
			}); err != nil {
				return
			}
		}
		if len(message.ToolCalls) != 0 {
			stopReason = "tool_use"
		}
	} else if textStarted {
		if err := writeEvent("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": 0,
		}); err != nil {
			return
		}
	}
	var stopSequence any
	if filter.Stopped() && stopReason != "tool_use" {
		stopSequence = filter.StoppingWord()
	}
	if err := writeEvent("message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": stopSequence,
		},
		"usage": map[string]int{"output_tokens": generatedTokens},
	}); err != nil {
		return
	}
	_ = writeEvent("message_stop", map[string]any{"type": "message_stop"})
}

func (h *Handler) anthropicInputTokens(response http.ResponseWriter, request *http.Request) {
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
	tokenizerAPI, ok := h.generator.(TokenizationAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "token counting is unavailable")
		return
	}
	var body anthropicTokenCountRequest
	if !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if body.Model != "" && body.Model != h.config.ModelID {
		writeError(response, http.StatusNotFound, "model_not_found", "requested model is not loaded")
		return
	}
	if rawJSONConfigured(body.Thinking) {
		writeError(
			response,
			http.StatusBadRequest,
			"invalid_request_error",
			"Anthropic thinking blocks are not supported",
		)
		return
	}
	toolSelection, err := selectAnthropicTools(body.Tools, body.ToolChoice)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	messages, err := parseAnthropicMessages(body.System, body.Messages)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	var prompt string
	if len(toolSelection.prompt) == 0 {
		prompt, err = formatter.FormatChat(messages)
	} else {
		prompt, err = formatChatRequest(
			formatter,
			messages,
			toolSelection.prompt,
			nil,
			map[string]any{"enable_thinking": false},
		)
	}
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	tokens, err := tokenizerAPI.TokenizeText(prompt, true, true)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]int{"input_tokens": len(tokens)})
}

func parseAnthropicMessages(
	rawSystem, rawMessages json.RawMessage,
) ([]inference.ChatMessage, error) {
	messages := make([]inference.ChatMessage, 0, 4)
	if len(rawSystem) != 0 && string(rawSystem) != "null" {
		system, err := parseAnthropicContent(rawSystem, "system")
		if err != nil {
			return nil, err
		}
		messages = append(messages, inference.ChatMessage{Role: "system", Content: system})
	}
	if len(rawMessages) == 0 {
		return nil, errors.New("messages is required")
	}
	var items []json.RawMessage
	if err := json.Unmarshal(rawMessages, &items); err != nil {
		return nil, errors.New("messages must be an array")
	}
	if len(items) == 0 {
		return nil, errors.New("messages must not be empty")
	}
	if len(items) > 1024 {
		return nil, errors.New("message count exceeds 1024")
	}
	for index, raw := range items {
		var item struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		decoder := json.NewDecoder(strings.NewReader(string(raw)))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&item); err != nil {
			return nil, fmt.Errorf("message %d: %w", index, err)
		}
		if err := requireEOF(decoder); err != nil {
			return nil, fmt.Errorf("message %d: %w", index, err)
		}
		if item.Role != "user" && item.Role != "assistant" {
			return nil, fmt.Errorf("message %d has unsupported role %q", index, item.Role)
		}
		parsed, err := parseAnthropicMessage(
			item.Role,
			item.Content,
			fmt.Sprintf("message %d", index),
		)
		if err != nil {
			return nil, err
		}
		messages = append(messages, parsed...)
	}
	return messages, nil
}

func parseAnthropicContent(raw json.RawMessage, label string) (string, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	var blocks []json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", fmt.Errorf("%s content must be a string or text-block array", label)
	}
	var result strings.Builder
	for index, rawBlock := range blocks {
		var block struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		decoder := json.NewDecoder(strings.NewReader(string(rawBlock)))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&block); err != nil {
			return "", fmt.Errorf("%s block %d: %w", label, index, err)
		}
		if err := requireEOF(decoder); err != nil {
			return "", fmt.Errorf("%s block %d: %w", label, index, err)
		}
		if block.Type != "text" {
			return "", fmt.Errorf("%s block %d has unsupported type %q", label, index, block.Type)
		}
		result.WriteString(block.Text)
	}
	return result.String(), nil
}
