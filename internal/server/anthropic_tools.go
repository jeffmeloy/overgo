package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/inference"
	"overgo/internal/strictjson"
)

func selectAnthropicTools(
	rawTools, rawChoice json.RawMessage,
) (chatToolSelection, error) {
	var tools []inference.ChatTool
	if strictjson.HasValue(rawTools) {
		var definitions []struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			InputSchema map[string]any `json:"input_schema"`
		}
		if err := strictjson.DecodeBytes(rawTools, &definitions); err != nil {
			return chatToolSelection{}, errors.New("tools must be an array of Anthropic tool definitions")
		}
		if err := validateProtocolToolCount(len(definitions)); err != nil {
			return chatToolSelection{}, err
		}
		tools = make([]inference.ChatTool, len(definitions))
		for index, definition := range definitions {
			if definition.Name == "" {
				return chatToolSelection{}, fmt.Errorf("tool %d name is required", index)
			}
			if definition.InputSchema == nil {
				return chatToolSelection{}, fmt.Errorf("tool %d input_schema is required", index)
			}
			tools[index] = inference.ChatTool{
				Type: inference.ChatToolTypeFunction,
				Function: inference.ChatToolDefinition{
					Name:        definition.Name,
					Description: definition.Description,
					Parameters:  definition.InputSchema,
				},
			}
		}
	}

	var openAIChoice json.RawMessage
	parallel := true
	if strictjson.HasValue(rawChoice) {
		var choice struct {
			Type               string `json:"type"`
			Name               string `json:"name"`
			DisableParallelUse *bool  `json:"disable_parallel_tool_use"`
		}
		if err := strictjson.DecodeBytes(rawChoice, &choice); err != nil {
			return chatToolSelection{}, errors.New("tool_choice must be an Anthropic tool-choice object")
		}
		switch choice.Type {
		case "auto":
			openAIChoice = json.RawMessage(`"auto"`)
		case "any":
			openAIChoice = json.RawMessage(`"required"`)
		case "tool":
			if choice.Name == "" {
				return chatToolSelection{}, errors.New("tool_choice type tool requires a name")
			}
			encoded, err := json.Marshal(map[string]any{
				"type": "function",
				"function": map[string]string{
					"name": choice.Name,
				},
			})
			if err != nil {
				return chatToolSelection{}, err
			}
			openAIChoice = encoded
		default:
			return chatToolSelection{}, fmt.Errorf("unsupported tool_choice type %q", choice.Type)
		}
		if choice.DisableParallelUse != nil {
			parallel = !*choice.DisableParallelUse
		}
	}
	selection, err := selectChatTools(chatCompletionRequest{
		Tools:      tools,
		ToolChoice: openAIChoice,
	})
	selection.parallel = parallel
	return selection, err
}

func (h *Handler) parseAnthropicMessage(
	ctx context.Context,
	role inference.ChatRole,
	raw json.RawMessage,
	label string,
) ([]inference.ChatMessage, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []inference.ChatMessage{{Role: role, Content: text}}, nil
	}
	var blocks []json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, fmt.Errorf("%s content must be a string or content-block array", label)
	}
	if role == inference.ChatRoleAssistant {
		message := inference.ChatMessage{Role: role}
		var content strings.Builder
		seenTool := false
		seenThinking := false
		seenVisible := false
		for index, rawBlock := range blocks {
			var header struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(rawBlock, &header); err != nil {
				return nil, fmt.Errorf("%s content block %d: %w", label, index, err)
			}
			switch header.Type {
			case "thinking":
				if seenThinking || seenVisible || seenTool {
					return nil, fmt.Errorf("%s thinking block %d is out of order", label, index)
				}
				var block struct {
					Type      string `json:"type"`
					Thinking  string `json:"thinking"`
					Signature string `json:"signature"`
				}
				if err := strictjson.DecodeBytes(rawBlock, &block); err != nil {
					return nil, fmt.Errorf("%s content block %d: %w", label, index, err)
				}
				if !h.thinkingSigner.verify(block.Thinking, block.Signature) {
					return nil, fmt.Errorf("%s content block %d has an invalid local thinking signature", label, index)
				}
				message.ReasoningContent = block.Thinking
				seenThinking = true
			case "text":
				if seenTool {
					return nil, fmt.Errorf("%s text after tool_use is not supported", label)
				}
				var block struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}
				if err := strictjson.DecodeBytes(rawBlock, &block); err != nil {
					return nil, fmt.Errorf("%s content block %d: %w", label, index, err)
				}
				content.WriteString(block.Text)
				seenVisible = true
			case "tool_use":
				seenTool = true
				var block struct {
					Type  string          `json:"type"`
					ID    string          `json:"id"`
					Name  string          `json:"name"`
					Input json.RawMessage `json:"input"`
				}
				if err := strictjson.DecodeBytes(rawBlock, &block); err != nil {
					return nil, fmt.Errorf("%s content block %d: %w", label, index, err)
				}
				var input map[string]any
				if block.ID == "" || block.Name == "" ||
					len(block.Input) == 0 ||
					json.Unmarshal(block.Input, &input) != nil ||
					input == nil {
					return nil, fmt.Errorf(
						"%s content block %d has invalid tool_use fields",
						label,
						index,
					)
				}
				message.ToolCalls = append(message.ToolCalls, inference.ChatToolCall{
					ID:   block.ID,
					Type: inference.ChatToolTypeFunction,
					Function: inference.ChatToolFunction{
						Name:      block.Name,
						Arguments: string(block.Input),
					},
				})
			default:
				return nil, fmt.Errorf(
					"%s content block %d has unsupported type %q",
					label,
					index,
					header.Type,
				)
			}
		}
		message.Content = content.String()
		if message.Content == "" && len(message.ToolCalls) == 0 {
			return nil, fmt.Errorf("%s content must not be empty", label)
		}
		return []inference.ChatMessage{message}, nil
	}

	_ = ctx
	messages := make([]inference.ChatMessage, 0, len(blocks))
	var content strings.Builder
	var media []inference.ChatMediaPart
	flushUser := func() {
		if content.Len() != 0 || len(media) != 0 {
			messages = append(messages, inference.ChatMessage{
				Role:    inference.ChatRoleUser,
				Content: content.String(),
				Media:   slices.Clone(media),
			})
			content.Reset()
			media = media[:0]
		}
	}
	for index, rawBlock := range blocks {
		var header struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(rawBlock, &header); err != nil {
			return nil, fmt.Errorf("%s content block %d: %w", label, index, err)
		}
		switch header.Type {
		case "text":
			var block struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if err := strictjson.DecodeBytes(rawBlock, &block); err != nil {
				return nil, fmt.Errorf("%s content block %d: %w", label, index, err)
			}
			content.WriteString(block.Text)
		case "image":
			var block struct {
				Type   string `json:"type"`
				Source struct {
					Type      string `json:"type"`
					MediaType string `json:"media_type"`
					Data      string `json:"data"`
					URL       string `json:"url"`
					FileID    string `json:"file_id"`
				} `json:"source"`
			}
			if err := strictjson.DecodeBytes(rawBlock, &block); err != nil {
				return nil, fmt.Errorf("%s content block %d: %w", label, index, err)
			}
			var source string
			switch block.Source.Type {
			case "base64":
				mediaType := strings.ToLower(strings.TrimSpace(block.Source.MediaType))
				if !strings.HasPrefix(mediaType, "image/") || block.Source.Data == "" ||
					block.Source.URL != "" || block.Source.FileID != "" {
					return nil, fmt.Errorf("%s content block %d has invalid base64 image source", label, index)
				}
				source = "data:" + mediaType + ";base64," + block.Source.Data
			case "url":
				if block.Source.URL == "" || block.Source.Data != "" || block.Source.FileID != "" || block.Source.MediaType != "" {
					return nil, fmt.Errorf("%s content block %d has invalid URL image source", label, index)
				}
				source = block.Source.URL
			case "file":
				if block.Source.FileID == "" || block.Source.Data != "" || block.Source.URL != "" || block.Source.MediaType != "" {
					return nil, fmt.Errorf("%s content block %d has invalid file image source", label, index)
				}
				file, ok := h.resolveResponseFile(block.Source.FileID)
				if !ok || !strings.HasPrefix(file.MediaType, "image/") {
					return nil, fmt.Errorf("%s content block %d image file_id is unavailable", label, index)
				}
				source = "data:" + file.MediaType + ";base64," + base64.StdEncoding.EncodeToString(file.Data)
			default:
				return nil, fmt.Errorf("%s content block %d has unsupported image source %q", label, index, block.Source.Type)
			}
			media = append(media, inference.ChatMediaPart{
				Type: "image", Data: source, TextOffset: content.Len(),
			})
		case "tool_result":
			flushUser()
			var block struct {
				Type      string          `json:"type"`
				ToolUseID string          `json:"tool_use_id"`
				Content   json.RawMessage `json:"content"`
				IsError   bool            `json:"is_error"`
			}
			if err := strictjson.DecodeBytes(rawBlock, &block); err != nil {
				return nil, fmt.Errorf("%s content block %d: %w", label, index, err)
			}
			if block.ToolUseID == "" || len(block.Content) == 0 {
				return nil, fmt.Errorf(
					"%s content block %d has invalid tool_result fields",
					label,
					index,
				)
			}
			result, err := parseAnthropicContent(
				block.Content,
				fmt.Sprintf("%s content block %d", label, index),
			)
			if err != nil {
				return nil, err
			}
			messages = append(messages, inference.ChatMessage{
				Role:            inference.ChatRoleTool,
				Content:         result,
				ToolCallID:      block.ToolUseID,
				ToolResultError: block.IsError,
			})
		default:
			return nil, fmt.Errorf(
				"%s content block %d has unsupported type %q",
				label,
				index,
				header.Type,
			)
		}
	}
	flushUser()
	if len(messages) == 0 {
		return nil, fmt.Errorf("%s content must not be empty", label)
	}
	return messages, nil
}

func (h *Handler) anthropicBlocks(message inference.ChatMessage, idPrefix string) ([]anthropicContentBlock, error) {
	blocks := make([]anthropicContentBlock, 0, 2+len(message.ToolCalls))
	if message.ReasoningContent != "" {
		if h == nil || h.thinkingSigner == nil {
			return nil, errors.New("Anthropic thinking signer is unavailable")
		}
		blocks = append(blocks, anthropicContentBlock{
			Type: "thinking", Thinking: message.ReasoningContent,
			Signature: h.thinkingSigner.sign(message.ReasoningContent),
		})
	}
	if message.Content != "" {
		blocks = append(blocks, anthropicContentBlock{
			Type: "text",
			Text: message.Content,
		})
	}
	for index, call := range message.ToolCalls {
		input := json.RawMessage(call.Function.Arguments)
		if !json.Valid(input) {
			return nil, fmt.Errorf("tool call %d arguments are invalid JSON", index)
		}
		callID := call.ID
		if callID == "" {
			callID = fmt.Sprintf("%s_%d", idPrefix, index)
		}
		blocks = append(blocks, anthropicContentBlock{
			Type:  "tool_use",
			ID:    callID,
			Name:  call.Function.Name,
			Input: input,
		})
	}
	return blocks, nil
}
