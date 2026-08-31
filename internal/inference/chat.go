package inference

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/strictjson"
)

const (
	chatMLStart       = "<|im_start|>"
	chatMLEnd         = "<|im_end|>"
	gemmaTurnStart    = "<start_of_turn>"
	gemmaTurnEnd      = "<end_of_turn>"
	llama3HeaderStart = "<|start_header_id|>"
	llama3HeaderEnd   = "<|end_header_id|>"
	llama3EndTurn     = "<|eot_id|>"
	llama3BeginText   = "<|begin_of_text|>"
)

type ChatRole string

const (
	ChatRoleSystem    ChatRole = "system"
	ChatRoleUser      ChatRole = "user"
	ChatRoleAssistant ChatRole = "assistant"
	ChatRoleTool      ChatRole = "tool"
)

func (role ChatRole) Valid() bool {
	return role == ChatRoleSystem || role == ChatRoleUser || role == ChatRoleAssistant || role == ChatRoleTool
}

type ChatMessage struct {
	Role             ChatRole        `json:"role"`
	Content          string          `json:"content"`
	Media            []ChatMediaPart `json:"-"`
	ReasoningContent string          `json:"reasoning_content,omitzero"`
	Name             string          `json:"name,omitzero"`
	ToolCallID       string          `json:"tool_call_id,omitzero"`
	ToolResultError  bool            `json:"is_error,omitzero"`
	ToolCalls        []ChatToolCall  `json:"tool_calls,omitempty"`
}

type ChatMediaType string

const (
	ChatMediaImage ChatMediaType = "image"
	ChatMediaAudio ChatMediaType = "audio"
	ChatMediaVideo ChatMediaType = "video"
)

type ChatMediaPart struct {
	Type       ChatMediaType
	Data       string
	Format     string
	TextOffset int
	FPS        float64
}

type ChatToolType string

const ChatToolTypeFunction ChatToolType = "function"

type ChatToolCall struct {
	ID       string           `json:"id,omitzero"`
	Type     ChatToolType     `json:"type"`
	Function ChatToolFunction `json:"function"`
}

type ChatToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatTool struct {
	Type     ChatToolType       `json:"type"`
	Function ChatToolDefinition `json:"function"`
}

type ChatToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitzero"`
	Parameters  map[string]any `json:"parameters"`
}

type ChatFormatOptions struct {
	Tools               []ChatTool
	AddGenerationPrompt bool
	EnableThinking      bool
	// TemplateKwargs carries declared chat-template variables beyond the
	// typed options — preserve_thinking, reasoning_effort — consumed by
	// templates that read them and inert otherwise. Reserved context keys
	// cannot be overridden.
	TemplateKwargs map[string]any
}

func (m ChatMessage) MarshalJSON() ([]byte, error) {
	content := any(m.Content)
	if len(m.ToolCalls) != 0 && m.Content == "" {
		content = nil
	}
	return json.Marshal(struct {
		Role             ChatRole       `json:"role"`
		Content          any            `json:"content"`
		ReasoningContent string         `json:"reasoning_content,omitzero"`
		Name             string         `json:"name,omitzero"`
		ToolCallID       string         `json:"tool_call_id,omitzero"`
		ToolResultError  bool           `json:"is_error,omitzero"`
		ToolCalls        []ChatToolCall `json:"tool_calls,omitempty"`
	}{
		Role:             m.Role,
		Content:          content,
		ReasoningContent: m.ReasoningContent,
		Name:             m.Name,
		ToolCallID:       m.ToolCallID,
		ToolResultError:  m.ToolResultError,
		ToolCalls:        m.ToolCalls,
	})
}

// UnmarshalJSON: string, null, supported OpenAI parts.
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	var wire struct {
		Role             ChatRole        `json:"role"`
		Content          json.RawMessage `json:"content"`
		ReasoningContent string          `json:"reasoning_content"`
		Name             string          `json:"name"`
		ToolCallID       string          `json:"tool_call_id"`
		ToolResultError  bool            `json:"is_error"`
		ToolCalls        json.RawMessage `json:"tool_calls"`
	}
	if err := strictjson.DecodeBytes(data, &wire); err != nil {
		return err
	}
	if len(wire.Content) == 0 && len(wire.ToolCalls) == 0 {
		return errors.New("inference: chat message content or tool_calls is required")
	}
	var content string
	var media []ChatMediaPart
	if strictjson.HasValue(wire.Content) {
		if err := json.Unmarshal(wire.Content, &content); err != nil {
			var rawParts []json.RawMessage
			if err := json.Unmarshal(wire.Content, &rawParts); err != nil {
				return errors.New("inference: chat message content must be a string, null, or content-part array")
			}
			var joined strings.Builder
			for index, raw := range rawParts {
				var kind struct {
					Type string `json:"type"`
				}
				if err := json.Unmarshal(raw, &kind); err != nil {
					return fmt.Errorf("inference: chat content part %d: %w", index, err)
				}
				switch kind.Type {
				case "text", "input_text", "output_text":
					var part struct {
						Type string `json:"type"`
						Text string `json:"text"`
					}
					if err := strictjson.DecodeBytes(raw, &part); err != nil {
						return fmt.Errorf("inference: chat content part %d: %w", index, err)
					}
					joined.WriteString(part.Text)
				case "image_url":
					var part struct {
						Type     string `json:"type"`
						ImageURL struct {
							URL    string `json:"url"`
							Detail string `json:"detail,omitzero"`
						} `json:"image_url"`
					}
					if err := strictjson.DecodeBytes(raw, &part); err != nil {
						return fmt.Errorf("inference: chat content part %d: %w", index, err)
					}
					if part.ImageURL.URL == "" {
						return fmt.Errorf("inference: chat content part %d image_url.url is required", index)
					}
					media = append(media, ChatMediaPart{
						Type: ChatMediaImage, Data: part.ImageURL.URL, TextOffset: joined.Len(),
					})
				case "input_audio":
					var part struct {
						Type       string `json:"type"`
						InputAudio struct {
							Data   string `json:"data"`
							URL    string `json:"url"`
							Format string `json:"format"`
						} `json:"input_audio"`
					}
					if err := strictjson.DecodeBytes(raw, &part); err != nil {
						return fmt.Errorf("inference: chat content part %d: %w", index, err)
					}
					if (part.InputAudio.Data == "") == (part.InputAudio.URL == "") {
						return fmt.Errorf("inference: chat content part %d input_audio requires exactly one of data or url", index)
					}
					if part.InputAudio.Data != "" && part.InputAudio.Format == "" {
						return fmt.Errorf("inference: chat content part %d input_audio data requires format", index)
					}
					source := part.InputAudio.Data
					if source == "" {
						source = part.InputAudio.URL
					}
					media = append(media, ChatMediaPart{
						Type: ChatMediaAudio, Data: source,
						Format: part.InputAudio.Format, TextOffset: joined.Len(),
					})
				case "input_video":
					var part struct {
						Type       string `json:"type"`
						InputVideo struct {
							Data string  `json:"data"`
							URL  string  `json:"url"`
							FPS  float64 `json:"fps,omitzero"`
						} `json:"input_video"`
					}
					if err := strictjson.DecodeBytes(raw, &part); err != nil {
						return fmt.Errorf("inference: chat content part %d: %w", index, err)
					}
					if (part.InputVideo.Data == "") == (part.InputVideo.URL == "") {
						return fmt.Errorf("inference: chat content part %d input_video requires exactly one of data or url", index)
					}
					source := part.InputVideo.Data
					if source == "" {
						source = part.InputVideo.URL
					}
					media = append(media, ChatMediaPart{
						Type: ChatMediaVideo, Data: source, FPS: part.InputVideo.FPS, TextOffset: joined.Len(),
					})
				default:
					return fmt.Errorf(
						"inference: chat content part %d has unsupported type %q",
						index,
						kind.Type,
					)
				}
			}
			content = joined.String()
		}
	}
	var toolCalls []ChatToolCall
	if strictjson.HasValue(wire.ToolCalls) {
		if err := json.Unmarshal(wire.ToolCalls, &toolCalls); err != nil {
			return fmt.Errorf("inference: invalid chat tool_calls: %w", err)
		}
		if len(toolCalls) == 0 {
			return errors.New("inference: chat tool_calls must not be empty")
		}
		for index := range toolCalls {
			if err := validateChatToolCall(toolCalls[index]); err != nil {
				return fmt.Errorf("inference: chat tool_call %d: %w", index, err)
			}
		}
	}
	m.Role = wire.Role
	m.Content = content
	m.Media = media
	m.ReasoningContent = wire.ReasoningContent
	m.Name = wire.Name
	m.ToolCallID = wire.ToolCallID
	m.ToolResultError = wire.ToolResultError
	m.ToolCalls = toolCalls
	return nil
}

func (f *ChatToolFunction) UnmarshalJSON(data []byte) error {
	var wire struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := strictjson.DecodeBytes(data, &wire); err != nil {
		return err
	}
	if wire.Name == "" {
		return errors.New("function name is required")
	}
	if len(wire.Arguments) == 0 {
		return errors.New("function arguments are required")
	}
	var arguments string
	if err := json.Unmarshal(wire.Arguments, &arguments); err != nil {
		if !json.Valid(wire.Arguments) {
			return errors.New("function arguments must be a JSON string or value")
		}
		arguments = string(wire.Arguments)
	}
	f.Name = wire.Name
	f.Arguments = arguments
	return nil
}

func validateChatToolCall(call ChatToolCall) error {
	if call.Type != ChatToolTypeFunction {
		return fmt.Errorf("unsupported type %q", call.Type)
	}
	if call.Function.Name == "" {
		return errors.New("function name is required")
	}
	if call.Function.Arguments == "" {
		return errors.New("function arguments are required")
	}
	return nil
}

// FormatChat: selects native formatter from boundary tokens in loaded
// vocabulary; Supported families currently include ChatML and Gemma turns
func (r *Runner) FormatChat(messages []ChatMessage) (string, error) {
	return r.FormatChatWithOptions(messages, ChatFormatOptions{
		AddGenerationPrompt: true,
		EnableThinking:      true,
	})
}

func (r *Runner) FormatChatWithOptions(
	messages []ChatMessage,
	options ChatFormatOptions,
) (string, error) {
	if r == nil || r.vocab == nil {
		return "", errRunnerNil
	}
	for index := range messages {
		if len(messages[index].Media) != 0 {
			return "", fmt.Errorf("inference: chat message %d requires multimodal projection", index)
		}
	}
	source := metadataString(r.file, "tokenizer.chat_template")
	if len(options.Tools) != 0 {
		if toolSource := metadataString(
			r.file,
			"tokenizer.chat_template.tool_use",
		); toolSource != "" {
			source = toolSource
		}
	}
	if source != "" {
		return r.formatJinjaChatNative(source, messages, options)
	}
	if len(options.Tools) != 0 {
		return "", errors.New("inference: tools require a GGUF Jinja chat template")
	}
	if !options.AddGenerationPrompt {
		return "", errors.New("inference: native chat formatters require add_generation_prompt")
	}
	if _, start := r.vocab.ID(chatMLStart); start {
		if _, end := r.vocab.ID(chatMLEnd); end {
			return formatChatML(messages)
		}
	}
	if _, start := r.vocab.ID(gemmaTurnStart); start {
		if _, end := r.vocab.ID(gemmaTurnEnd); end {
			return formatGemmaChat(messages)
		}
	}
	if _, start := r.vocab.ID(llama3HeaderStart); start {
		if _, end := r.vocab.ID(llama3HeaderEnd); end {
			if _, eot := r.vocab.ID(llama3EndTurn); eot {
				return formatLlama3Chat(messages)
			}
		}
	}
	return "", errors.New("inference: vocabulary has no supported chat-template boundaries")
}

// buildChatContext assembles the template variable context shared by the gonja
// and stdlib (internal/jinja) rendering paths. Messages and tools are []any so
// both engines iterate them identically.
func (r *Runner) buildChatContext(
	source string,
	messages []ChatMessage,
	options ChatFormatOptions,
) (map[string]any, error) {
	wireMessages := make([]any, len(messages))
	for index, message := range messages {
		if !message.Role.Valid() {
			return nil, fmt.Errorf(
				"inference: chat message %d has unsupported role %q",
				index,
				message.Role,
			)
		}
		wire := map[string]any{
			"role":    string(message.Role),
			"content": message.Content,
		}
		if message.ReasoningContent != "" {
			wire["reasoning_content"] = message.ReasoningContent
		}
		if message.Name != "" {
			wire["name"] = message.Name
		}
		if message.ToolCallID != "" {
			wire["tool_call_id"] = message.ToolCallID
		}
		if message.ToolResultError {
			wire["is_error"] = true
		}
		if len(message.ToolCalls) != 0 {
			calls := make([]any, len(message.ToolCalls))
			for callIndex, call := range message.ToolCalls {
				if err := validateChatToolCall(call); err != nil {
					return nil, fmt.Errorf(
						"inference: chat message %d tool_call %d: %w",
						index,
						callIndex,
						err,
					)
				}
				arguments := any(call.Function.Arguments)
				if chatTemplateUsesObjectArguments(source) {
					var object any
					if json.Unmarshal([]byte(call.Function.Arguments), &object) == nil {
						arguments = object
					}
				}
				calls[callIndex] = chatTemplateToolCall{
					call:      call,
					arguments: arguments,
				}
			}
			wire["tool_calls"] = calls
		}
		wireMessages[index] = wire
	}
	wireTools := make([]any, len(options.Tools))
	for index, tool := range options.Tools {
		if err := validateChatTool(tool); err != nil {
			return nil, fmt.Errorf("inference: chat tool %d: %w", index, err)
		}
		wireTools[index] = chatTemplateTool{tool: tool}
	}
	bos := vocabularyTokenText(r.vocab, r.vocab.BOS)
	eos := vocabularyTokenText(r.vocab, r.vocab.EOS)
	context := map[string]any{
		"messages":              wireMessages,
		"bos_token":             bos,
		"eos_token":             eos,
		"add_generation_prompt": options.AddGenerationPrompt,
		"enable_thinking":       options.EnableThinking,
		"tools":                 wireTools,
		"documents":             nil,
	}
	for name, value := range options.TemplateKwargs {
		if _, reserved := context[name]; reserved {
			return nil, fmt.Errorf("inference: chat template kwarg %q overrides a reserved context key", name)
		}
		context[name] = value
	}
	return context, nil
}

func validateChatTool(tool ChatTool) error {
	if tool.Type != ChatToolTypeFunction {
		return fmt.Errorf("unsupported type %q", tool.Type)
	}
	if tool.Function.Name == "" {
		return errors.New("function name is required")
	}
	if tool.Function.Parameters == nil {
		return errors.New("function parameters are required")
	}
	return nil
}

// chatTemplateTool, chatTemplateToolFunction, chatTemplateToolCall, and
// chatTemplateCallFunction adapt chat tools/calls for template access. Their
// JinjaGet methods (see chat_jinja.go) expose attribute/item access to the
// stdlib interpreter; their unexported fields make encoding/json emit "{}",
// matching gonja's tojson fallback for these opaque objects.
type chatTemplateTool struct {
	tool ChatTool
}

type chatTemplateToolFunction struct {
	definition ChatToolDefinition
}

type chatTemplateToolCall struct {
	call      ChatToolCall
	arguments any
}

type chatTemplateCallFunction struct {
	function  ChatToolFunction
	arguments any
}

func writeChatTemplateJSON(output *strings.Builder, value any) {
	switch typed := value.(type) {
	case chatTemplateTool:
		output.WriteString(`{"type": "function", "function": `)
		writeChatTemplateJSON(
			output,
			chatTemplateToolFunction{definition: typed.tool.Function},
		)
		output.WriteByte('}')
	case chatTemplateToolFunction:
		output.WriteString(`{"name": `)
		writeChatTemplateJSON(output, typed.definition.Name)
		output.WriteString(`, "description": `)
		writeChatTemplateJSON(output, typed.definition.Description)
		output.WriteString(`, "parameters": `)
		writeChatTemplateJSON(output, typed.definition.Parameters)
		output.WriteByte('}')
	case chatTemplateToolCall:
		output.WriteString(`{"type": "function", "function": `)
		writeChatTemplateJSON(output, chatTemplateCallFunction{
			function:  typed.call.Function,
			arguments: typed.arguments,
		})
		if typed.call.ID != "" {
			output.WriteString(`, "id": `)
			writeChatTemplateJSON(output, typed.call.ID)
		}
		output.WriteByte('}')
	case chatTemplateCallFunction:
		output.WriteString(`{"name": `)
		writeChatTemplateJSON(output, typed.function.Name)
		output.WriteString(`, "arguments": `)
		writeChatTemplateJSON(output, typed.arguments)
		output.WriteByte('}')
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		slices.Sort(keys)
		output.WriteByte('{')
		for index, key := range keys {
			if index != 0 {
				output.WriteString(", ")
			}
			writeChatTemplateJSON(output, key)
			output.WriteString(": ")
			writeChatTemplateJSON(output, typed[key])
		}
		output.WriteByte('}')
	case []any:
		output.WriteByte('[')
		for index, item := range typed {
			if index != 0 {
				output.WriteString(", ")
			}
			writeChatTemplateJSON(output, item)
		}
		output.WriteByte(']')
	case []string:
		output.WriteByte('[')
		for index, item := range typed {
			if index != 0 {
				output.WriteString(", ")
			}
			writeChatTemplateJSON(output, item)
		}
		output.WriteByte(']')
	default:
		data, err := json.Marshal(typed)
		if err != nil {
			output.WriteString("null")
			return
		}
		output.Write(data)
	}
}

func normalizeChatTemplateSource(source string) string {
	// Gemma: normalize parenthesized inline conditional.
	source = strings.ReplaceAll(
		source,
		`(first_user_prefix if loop.first else "")`,
		`((loop.first and first_user_prefix) or "")`,
	)
	// Qwen: normalize method-call precedence.
	source = strings.ReplaceAll(
		source,
		`(not loop.last and (not reasoning_content.strip() == ''))`,
		`(not loop.last and ((reasoning_content|trim|length) > 0))`,
	)
	return strings.ReplaceAll(
		source,
		`not reasoning_content.strip() == ''`,
		`(reasoning_content|trim|length) > 0`,
	)
}

func chatTemplateUsesObjectArguments(source string) bool {
	return strings.Contains(source, "tool_call.arguments") &&
		strings.Contains(source, "| tojson")
}

func formatChatML(messages []ChatMessage) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("inference: chat message list is empty")
	}
	var result strings.Builder
	for index, message := range messages {
		if !message.Role.Valid() {
			return "", fmt.Errorf("inference: chat message %d has unsupported role %q", index, message.Role)
		}
		if strings.Contains(message.Content, chatMLStart) ||
			strings.Contains(message.Content, chatMLEnd) {
			return "", fmt.Errorf("inference: chat message %d contains a ChatML boundary token", index)
		}
		result.WriteString(chatMLStart)
		result.WriteString(string(message.Role))
		result.WriteByte('\n')
		result.WriteString(message.Content)
		result.WriteString(chatMLEnd)
		result.WriteByte('\n')
	}
	result.WriteString(chatMLStart)
	result.WriteString(string(ChatRoleAssistant))
	result.WriteByte('\n')
	return result.String(), nil
}

func formatGemmaChat(messages []ChatMessage) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("inference: chat message list is empty")
	}
	firstUserPrefix := ""
	first := 0
	if messages[0].Role == ChatRoleSystem {
		if err := rejectChatBoundary(messages[0].Content, 0, gemmaTurnStart, gemmaTurnEnd); err != nil {
			return "", err
		}
		firstUserPrefix = messages[0].Content + "\n\n"
		first = 1
	}
	var result strings.Builder
	for index := first; index < len(messages); index++ {
		message := messages[index]
		wantRole := ChatRoleUser
		renderedRole := "user"
		if (index-first)%2 == 1 {
			wantRole = ChatRoleAssistant
			renderedRole = "model"
		}
		if message.Role != wantRole {
			return "", fmt.Errorf(
				"inference: Gemma chat message %d role %q; want %q",
				index,
				message.Role,
				wantRole,
			)
		}
		if err := rejectChatBoundary(message.Content, index, gemmaTurnStart, gemmaTurnEnd); err != nil {
			return "", err
		}
		result.WriteString(gemmaTurnStart)
		result.WriteString(renderedRole)
		result.WriteByte('\n')
		if index == first {
			result.WriteString(firstUserPrefix)
		}
		result.WriteString(strings.TrimSpace(message.Content))
		result.WriteString(gemmaTurnEnd)
		result.WriteByte('\n')
	}
	result.WriteString(gemmaTurnStart)
	result.WriteString("model\n")
	return result.String(), nil
}

func formatLlama3Chat(messages []ChatMessage) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("inference: chat message list is empty")
	}
	var result strings.Builder
	nextRole := ChatRoleUser
	for index, message := range messages {
		if index == 0 && message.Role == ChatRoleSystem {
			nextRole = ChatRoleUser
		} else {
			if message.Role != nextRole {
				return "", fmt.Errorf(
					"inference: Llama 3 chat message %d role %q; want %q",
					index,
					message.Role,
					nextRole,
				)
			}
			if nextRole == ChatRoleUser {
				nextRole = ChatRoleAssistant
			} else {
				nextRole = ChatRoleUser
			}
		}
		if err := rejectChatBoundary(
			message.Content,
			index,
			llama3BeginText,
			llama3HeaderStart,
			llama3HeaderEnd,
			llama3EndTurn,
		); err != nil {
			return "", err
		}
		result.WriteString(llama3HeaderStart)
		result.WriteString(string(message.Role))
		result.WriteString(llama3HeaderEnd)
		result.WriteString("\n\n")
		result.WriteString(strings.TrimSpace(message.Content))
		result.WriteString(llama3EndTurn)
	}
	result.WriteString(llama3HeaderStart)
	result.WriteString(string(ChatRoleAssistant))
	result.WriteString(llama3HeaderEnd)
	result.WriteString("\n\n")
	return result.String(), nil
}

func rejectChatBoundary(content string, index int, boundaries ...string) error {
	for _, boundary := range boundaries {
		if strings.Contains(content, boundary) {
			return fmt.Errorf(
				"inference: chat message %d contains a chat boundary token",
				index,
			)
		}
	}
	return nil
}
