package server

import (
	"context"
	"encoding/json"

	"errors"

	"fmt"

	"llamacpp2go/internal/inference"

	"llamacpp2go/internal/strictjson"

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
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
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
	if !requireMethod(response, request, http.MethodPost) {
		return
	}
	formatter, ok := h.requireChatFormatter(response)
	if !ok {
		return
	}
	var body anthropicTokenCountRequest
	if !h.decodeProtocolJSON(response, request, &body, func() string { return body.Model }) {
		return
	}
	maxTokens, err := boundedProtocolTokens(
		body.MaxTokens, 0, h.config.MaxTokens, "max_tokens", true,
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
	thinkingEnabled, err := validateAnthropicThinking(body.Thinking, body.MaxTokens)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	toolSelection, err := selectAnthropicTools(body.Tools, body.ToolChoice)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if thinkingEnabled && len(toolSelection.active) != 0 {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "local Anthropic thinking cannot be combined with tools; interleaved signed thinking is unavailable")
		return
	}
	if thinkingEnabled {
		if _, ok := h.generator.(ChatOutputParser); !ok {
			writeError(response, http.StatusNotImplemented, "unsupported_operation", "thinking output parsing is unavailable")
			return
		}
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
	messages, err := h.parseAnthropicMessages(request.Context(), body.System, body.Messages)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	normalizedBody := chatCompletionRequest{Messages: messages, N: 1}
	if len(toolSelection.prompt) != 0 || thinkingEnabled {
		normalizedBody.Tools = toolSelection.active
		normalizedBody.TemplateKwargs = map[string]any{"enable_thinking": thinkingEnabled}
	}
	normalized, err := h.normalizeChatPrompt(
		request.Context(), formatter, normalizedBody, toolSelection.prompt,
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	samplingParams := samplingParameters{
		Temperature: body.Temperature,
		TopP:        body.TopP,
		TopK:        body.TopK,
	}
	if !h.configureChatToolGrammar(
		response,
		&samplingParams,
		toolSelection.active,
		toolSelection.required,
		false,
		toolSelection.parallel,
	) {
		return
	}
	plan, ok := h.prepareProtocolGenerationPlan(
		response, request, normalized, samplingParams, maxTokens, body.StopSequences,
	)
	if !ok {
		return
	}
	defer plan.release()
	messageID := "msg_" + strconv.FormatUint(h.nextID.Add(1), 10)
	if body.Stream {
		h.streamAnthropicMessages(
			response,
			request,
			plan,
			messageID,
			toolSelection.active,
			thinkingEnabled,
		)
		return
	}
	result, err := plan.run(nil)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	pump := result.pump
	stopReason := pump.finishReason(maxTokens, "end_turn", "max_tokens")
	var stopSequence *string
	if pump.stopped() {
		value := pump.stoppingWord()
		stopSequence = &value
	}
	message := inference.ChatMessage{Role: "assistant", Content: pump.text()}
	if len(toolSelection.active) != 0 || thinkingEnabled {
		message, err = h.generator.(ChatOutputParser).ParseChatOutput(
			pump.text(),
			toolSelection.active,
		)
		if err != nil {
			writeGenerationError(response, err)
			return
		}
		if thinkingEnabled && message.ReasoningContent == "" {
			writeGenerationError(response, errors.New("model output omitted required thinking content"))
			return
		}
	}
	content, err := h.anthropicBlocks(message, "toolu_"+strings.TrimPrefix(messageID, "msg_"))
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
			InputTokens:          len(plan.prompt.TokenIDs),
			OutputTokens:         pump.generated,
		},
	})
}

func (h *Handler) streamAnthropicMessages(
	response http.ResponseWriter,
	request *http.Request,
	plan *protocolGenerationPlan,
	messageID string,
	tools []inference.ChatTool,
	thinkingEnabled bool,
) {
	flusher, ok := beginSSE(response)
	if !ok {
		return
	}
	stream := newSSEEmitter(request.Context(), response, flusher)
	writeEvent := stream.named
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
				InputTokens:          len(plan.prompt.TokenIDs),
				OutputTokens:         0,
			},
		},
	}); err != nil {
		return
	}
	textStarted := false
	textStopped := false
	var buffered strings.Builder
	toolStream, streamErr := newToolDeltaStream(h.generator, tools)
	if streamErr != nil {
		_ = writeEvent("error", map[string]any{
			"type":  "error",
			"error": errorEnvelope("generation_error", streamErr.Error()).Error,
		})
		return
	}
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
	emitToolPiece := func(piece string) error {
		deltas, streamErr := toolStream.accept(piece)
		if streamErr != nil {
			return streamErr
		}
		idPrefix := "toolu_" + strings.TrimPrefix(messageID, "msg_")
		for _, delta := range deltas {
			if delta.Content != "" {
				if streamErr := emitText(delta.Content); streamErr != nil {
					return streamErr
				}
			}
			if delta.Started && textStarted && !textStopped {
				if streamErr := writeEvent("content_block_stop", map[string]any{
					"type":  "content_block_stop",
					"index": 0,
				}); streamErr != nil {
					return streamErr
				}
				textStopped = true
			}
			blockIndex := delta.Index
			if textStarted {
				blockIndex++
			}
			if delta.Started {
				if streamErr := writeEvent("content_block_start", map[string]any{
					"type":  "content_block_start",
					"index": blockIndex,
					"content_block": map[string]any{
						"type":  "tool_use",
						"id":    fmt.Sprintf("%s_%d", idPrefix, delta.Index),
						"name":  delta.Name,
						"input": map[string]any{},
					},
				}); streamErr != nil {
					return streamErr
				}
			}
			if delta.Arguments != "" {
				if streamErr := writeEvent("content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": blockIndex,
					"delta": map[string]any{
						"type":         "input_json_delta",
						"partial_json": delta.Arguments,
					},
				}); streamErr != nil {
					return streamErr
				}
			}
		}
		return nil
	}
	emitCompletedBlock := func(index int, block anthropicContentBlock) error {
		switch block.Type {
		case "thinking":
			if err := writeEvent("content_block_start", map[string]any{
				"type": "content_block_start", "index": index,
				"content_block": map[string]any{"type": "thinking", "thinking": "", "signature": ""},
			}); err != nil {
				return err
			}
			if block.Thinking != "" {
				if err := writeEvent("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": index,
					"delta": map[string]any{"type": "thinking_delta", "thinking": block.Thinking},
				}); err != nil {
					return err
				}
			}
			if err := writeEvent("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": index,
				"delta": map[string]any{"type": "signature_delta", "signature": block.Signature},
			}); err != nil {
				return err
			}
		case "text":
			if err := writeEvent("content_block_start", map[string]any{
				"type": "content_block_start", "index": index,
				"content_block": map[string]any{"type": "text", "text": ""},
			}); err != nil {
				return err
			}
			if block.Text != "" {
				if err := writeEvent("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": index,
					"delta": map[string]any{"type": "text_delta", "text": block.Text},
				}); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("unsupported completed Anthropic block %q", block.Type)
		}
		return writeEvent("content_block_stop", map[string]any{
			"type": "content_block_stop", "index": index,
		})
	}
	result, err := plan.run(
		func(piece string) error {
			if thinkingEnabled {
				buffered.WriteString(piece)
				return request.Context().Err()
			}
			if len(tools) != 0 {
				if streamErr := emitToolPiece(piece); streamErr != nil {
					return streamErr
				}
				return request.Context().Err()
			}
			return emitText(piece)
		},
	)
	if err != nil {
		_ = writeEvent("error", map[string]any{
			"type":  "error",
			"error": errorEnvelope("generation_error", err.Error()).Error,
		})
		return
	}
	pump := result.pump
	stopReason := pump.finishReason(plan.maxTokens, "end_turn", "max_tokens")
	if thinkingEnabled {
		message, parseErr := h.generator.(ChatOutputParser).ParseChatOutput(buffered.String(), nil)
		if parseErr != nil || message.ReasoningContent == "" {
			if parseErr == nil {
				parseErr = errors.New("model output omitted required thinking content")
			}
			_ = writeEvent("error", map[string]any{
				"type": "error", "error": errorEnvelope("generation_error", parseErr.Error()).Error,
			})
			return
		}
		blocks, blockErr := h.anthropicBlocks(message, "")
		if blockErr != nil {
			_ = writeEvent("error", map[string]any{
				"type": "error", "error": errorEnvelope("generation_error", blockErr.Error()).Error,
			})
			return
		}
		for index, block := range blocks {
			if emitErr := emitCompletedBlock(index, block); emitErr != nil {
				return
			}
		}
		textStarted = true
		textStopped = true
	} else if len(tools) != 0 {
		message, parseErr := h.generator.(ChatOutputParser).ParseChatOutput(
			toolStream.text(),
			tools,
		)
		if parseErr != nil {
			_ = writeEvent("error", map[string]any{
				"type":  "error",
				"error": errorEnvelope("generation_error", parseErr.Error()).Error,
			})
			return
		}
		blocks, blockErr := h.anthropicBlocks(
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
			if toolStream.started {
				if block.Type == "tool_use" {
					streamIndex := index
					if textStarted {
						streamIndex--
					}
					if toolStream.streamed(streamIndex) {
						if err := writeEvent("content_block_stop", map[string]any{
							"type":  "content_block_stop",
							"index": index,
						}); err != nil {
							return
						}
					}
				}
				continue
			}
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
	} else if textStarted && !textStopped {
		if err := writeEvent("content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": 0,
		}); err != nil {
			return
		}
	}
	var stopSequence any
	if pump.stopped() && stopReason != "tool_use" {
		stopSequence = pump.stoppingWord()
	}
	if err := writeEvent("message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": stopSequence,
		},
		"usage": map[string]int{"output_tokens": pump.generated},
	}); err != nil {
		return
	}
	_ = writeEvent("message_stop", map[string]any{"type": "message_stop"})
}

func (h *Handler) anthropicInputTokens(response http.ResponseWriter, request *http.Request) {
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
	var body anthropicTokenCountRequest
	if !h.decodeProtocolJSON(response, request, &body, func() string { return body.Model }) {
		return
	}
	thinkingEnabled, err := validateAnthropicThinking(body.Thinking, body.MaxTokens)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	toolSelection, err := selectAnthropicTools(body.Tools, body.ToolChoice)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if thinkingEnabled && len(toolSelection.active) != 0 {
		writeError(response, http.StatusBadRequest, "invalid_request_error", "local Anthropic thinking cannot be combined with tools; interleaved signed thinking is unavailable")
		return
	}
	messages, err := h.parseAnthropicMessages(request.Context(), body.System, body.Messages)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	normalizedBody := chatCompletionRequest{Messages: messages, N: 1}
	if len(toolSelection.prompt) != 0 || thinkingEnabled {
		normalizedBody.Tools = toolSelection.active
		normalizedBody.TemplateKwargs = map[string]any{"enable_thinking": thinkingEnabled}
	}
	normalized, err := h.normalizeChatPrompt(
		request.Context(), formatter, normalizedBody, toolSelection.prompt,
	)
	if err != nil {
		writeError(response, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	h.writeProtocolInputTokenCount(response, request, normalized, false)
}

func (h *Handler) parseAnthropicMessages(
	ctx context.Context,
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
		if err := strictjson.DecodeBytes(raw, &item); err != nil {
			return nil, fmt.Errorf("message %d: %w", index, err)
		}
		if item.Role != "user" && item.Role != "assistant" {
			return nil, fmt.Errorf("message %d has unsupported role %q", index, item.Role)
		}
		parsed, err := h.parseAnthropicMessage(
			ctx,
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

func parseAnthropicMessages(
	rawSystem, rawMessages json.RawMessage,
) ([]inference.ChatMessage, error) {
	return (&Handler{}).parseAnthropicMessages(context.Background(), rawSystem, rawMessages)
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
		if err := strictjson.DecodeBytes(rawBlock, &block); err != nil {
			return "", fmt.Errorf("%s block %d: %w", label, index, err)
		}
		if block.Type != "text" {
			return "", fmt.Errorf("%s block %d has unsupported type %q", label, index, block.Type)
		}
		result.WriteString(block.Text)
	}
	return result.String(), nil
}
