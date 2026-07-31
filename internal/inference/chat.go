package inference

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/nikolalohinski/gonja/v2"
	"github.com/nikolalohinski/gonja/v2/config"
	"github.com/nikolalohinski/gonja/v2/exec"
	"github.com/nikolalohinski/gonja/v2/loaders"
)

const (
	chatMLStart           = "<|im_start|>"
	chatMLEnd             = "<|im_end|>"
	gemmaTurnStart        = "<start_of_turn>"
	gemmaTurnEnd          = "<end_of_turn>"
	llama3HeaderStart     = "<|start_header_id|>"
	llama3HeaderEnd       = "<|end_header_id|>"
	llama3EndTurn         = "<|eot_id|>"
	llama3BeginText       = "<|begin_of_text|>"
	maxChatTemplateBytes  = 1 << 20
	maxFormattedChatBytes = 16 << 20
)

type ChatMessage struct {
	Role             string         `json:"role"`
	Content          string         `json:"content"`
	ReasoningContent string         `json:"reasoning_content,omitempty"`
	Name             string         `json:"name,omitempty"`
	ToolCallID       string         `json:"tool_call_id,omitempty"`
	ToolCalls        []ChatToolCall `json:"tool_calls,omitempty"`
}

type ChatToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type"`
	Function ChatToolFunction `json:"function"`
}

type ChatToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatTool struct {
	Type     string             `json:"type"`
	Function ChatToolDefinition `json:"function"`
}

type ChatToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type ChatFormatOptions struct {
	Tools               []ChatTool
	AddGenerationPrompt bool
	EnableThinking      bool
}

func (m ChatMessage) MarshalJSON() ([]byte, error) {
	content := any(m.Content)
	if len(m.ToolCalls) != 0 && m.Content == "" {
		content = nil
	}
	return json.Marshal(struct {
		Role             string         `json:"role"`
		Content          any            `json:"content"`
		ReasoningContent string         `json:"reasoning_content,omitempty"`
		Name             string         `json:"name,omitempty"`
		ToolCallID       string         `json:"tool_call_id,omitempty"`
		ToolCalls        []ChatToolCall `json:"tool_calls,omitempty"`
	}{
		Role:             m.Role,
		Content:          content,
		ReasoningContent: m.ReasoningContent,
		Name:             m.Name,
		ToolCallID:       m.ToolCallID,
		ToolCalls:        m.ToolCalls,
	})
}

// UnmarshalJSON accepts the OpenAI text-only content-part form in addition to
// the legacy string form. Non-text parts remain explicit errors because the
// text-only Runner has no multimodal prompt encoder.
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	var wire struct {
		Role             string          `json:"role"`
		Content          json.RawMessage `json:"content"`
		ReasoningContent string          `json:"reasoning_content"`
		Name             string          `json:"name"`
		ToolCallID       string          `json:"tool_call_id"`
		ToolCalls        json.RawMessage `json:"tool_calls"`
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	if err := requireChatJSONEOF(decoder); err != nil {
		return err
	}
	if len(wire.Content) == 0 && len(wire.ToolCalls) == 0 {
		return errors.New("inference: chat message content or tool_calls is required")
	}
	var content string
	if len(wire.Content) != 0 && string(wire.Content) != "null" {
		if err := json.Unmarshal(wire.Content, &content); err != nil {
			var rawParts []json.RawMessage
			if err := json.Unmarshal(wire.Content, &rawParts); err != nil {
				return errors.New("inference: chat message content must be a string, null, or text-part array")
			}
			var joined strings.Builder
			for index, raw := range rawParts {
				var part struct {
					Type string `json:"type"`
					Text string `json:"text"`
				}
				partDecoder := json.NewDecoder(bytes.NewReader(raw))
				partDecoder.DisallowUnknownFields()
				if err := partDecoder.Decode(&part); err != nil {
					return fmt.Errorf("inference: chat content part %d: %w", index, err)
				}
				if err := requireChatJSONEOF(partDecoder); err != nil {
					return fmt.Errorf("inference: chat content part %d: %w", index, err)
				}
				switch part.Type {
				case "text", "input_text", "output_text":
				default:
					return fmt.Errorf(
						"inference: chat content part %d has unsupported type %q",
						index,
						part.Type,
					)
				}
				joined.WriteString(part.Text)
			}
			content = joined.String()
		}
	}
	var toolCalls []ChatToolCall
	if len(wire.ToolCalls) != 0 && string(wire.ToolCalls) != "null" {
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
	m.ReasoningContent = wire.ReasoningContent
	m.Name = wire.Name
	m.ToolCallID = wire.ToolCallID
	m.ToolCalls = toolCalls
	return nil
}

func (f *ChatToolFunction) UnmarshalJSON(data []byte) error {
	var wire struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	if err := requireChatJSONEOF(decoder); err != nil {
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
	if call.Type != "function" {
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

func requireChatJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("inference: unexpected JSON after message")
	}
	return err
}

// FormatChat selects a native formatter from boundary tokens in the loaded
// vocabulary. Supported families currently include ChatML and Gemma turns.
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
		return "", errors.New("inference: runner is nil")
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
		return r.formatJinjaChat(source, messages, options)
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

func (r *Runner) formatJinjaChat(
	source string,
	messages []ChatMessage,
	options ChatFormatOptions,
) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("inference: chat message list is empty")
	}
	if len(source) > maxChatTemplateBytes {
		return "", errors.New("inference: chat template exceeds 1 MiB")
	}
	source = normalizeChatTemplateSource(source)
	wireMessages := make([]map[string]any, len(messages))
	for index, message := range messages {
		switch message.Role {
		case "system", "user", "assistant", "tool":
		default:
			return "", fmt.Errorf(
				"inference: chat message %d has unsupported role %q",
				index,
				message.Role,
			)
		}
		wireMessages[index] = map[string]any{
			"role":    message.Role,
			"content": message.Content,
		}
		if message.ReasoningContent != "" {
			wireMessages[index]["reasoning_content"] = message.ReasoningContent
		}
		if message.Name != "" {
			wireMessages[index]["name"] = message.Name
		}
		if message.ToolCallID != "" {
			wireMessages[index]["tool_call_id"] = message.ToolCallID
		}
		if len(message.ToolCalls) != 0 {
			calls := make([]chatTemplateToolCall, len(message.ToolCalls))
			for callIndex, call := range message.ToolCalls {
				if err := validateChatToolCall(call); err != nil {
					return "", fmt.Errorf(
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
			wireMessages[index]["tool_calls"] = calls
		}
	}
	wireTools := make([]chatTemplateTool, len(options.Tools))
	for index, tool := range options.Tools {
		if err := validateChatTool(tool); err != nil {
			return "", fmt.Errorf("inference: chat tool %d: %w", index, err)
		}
		wireTools[index] = chatTemplateTool{tool: tool}
	}
	const identifier = "/tokenizer.chat_template"
	loader, err := loaders.NewMemoryLoader(map[string]string{
		identifier: source,
	})
	if err != nil {
		return "", fmt.Errorf("inference: initialize chat-template loader: %w", err)
	}
	configuration := config.New()
	configuration.AutoEscape = false
	template, err := exec.NewTemplate(
		identifier,
		configuration,
		loader,
		newChatTemplateEnvironment(),
	)
	if err != nil {
		return "", fmt.Errorf("inference: parse GGUF chat template: %w", err)
	}
	bos := vocabularyTokenText(r.vocab, r.vocab.BOS)
	eos := vocabularyTokenText(r.vocab, r.vocab.EOS)
	context := exec.NewContext(map[string]any{
		"messages":              wireMessages,
		"bos_token":             bos,
		"eos_token":             eos,
		"add_generation_prompt": options.AddGenerationPrompt,
		"enable_thinking":       options.EnableThinking,
		"tools":                 wireTools,
		"documents":             nil,
	})
	writer := &boundedStringWriter{limit: maxFormattedChatBytes}
	if err := template.Execute(writer, context); err != nil {
		return "", fmt.Errorf("inference: execute GGUF chat template: %w", err)
	}
	result := writer.String()
	if r.vocab.AddBOS && bos != "" {
		result = strings.TrimPrefix(result, bos)
	}
	if result == "" {
		return "", errors.New("inference: chat template produced an empty prompt")
	}
	return result, nil
}

func validateChatTool(tool ChatTool) error {
	if tool.Type != "function" {
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

type chatTemplateTool struct {
	tool ChatTool
}

func (t chatTemplateTool) GetAttribute(name string) (*exec.Value, bool) {
	return t.GetItem(name)
}

func (t chatTemplateTool) GetItem(key any) (*exec.Value, bool) {
	switch key {
	case "type":
		return exec.AsValue("function"), true
	case "function":
		return exec.AsValue(chatTemplateToolFunction{definition: t.tool.Function}), true
	default:
		return exec.AsValue(nil), false
	}
}

type chatTemplateToolFunction struct {
	definition ChatToolDefinition
}

type chatTemplateToolCall struct {
	call      ChatToolCall
	arguments any
}

func (c chatTemplateToolCall) GetAttribute(name string) (*exec.Value, bool) {
	return c.GetItem(name)
}

func (c chatTemplateToolCall) GetItem(key any) (*exec.Value, bool) {
	switch key {
	case "id":
		if c.call.ID == "" {
			return exec.AsValue(nil), false
		}
		return exec.AsValue(c.call.ID), true
	case "type":
		return exec.AsValue("function"), true
	case "function":
		return exec.AsValue(chatTemplateCallFunction{
			function:  c.call.Function,
			arguments: c.arguments,
		}), true
	default:
		return exec.AsValue(nil), false
	}
}

type chatTemplateCallFunction struct {
	function  ChatToolFunction
	arguments any
}

func (f chatTemplateCallFunction) GetAttribute(name string) (*exec.Value, bool) {
	return f.GetItem(name)
}

func (f chatTemplateCallFunction) GetItem(key any) (*exec.Value, bool) {
	switch key {
	case "name":
		return exec.AsValue(f.function.Name), true
	case "arguments":
		return exec.AsValue(f.arguments), true
	default:
		return exec.AsValue(nil), false
	}
}

func (f chatTemplateToolFunction) GetAttribute(name string) (*exec.Value, bool) {
	return f.GetItem(name)
}

func (f chatTemplateToolFunction) GetItem(key any) (*exec.Value, bool) {
	switch key {
	case "name":
		return exec.AsValue(f.definition.Name), true
	case "description":
		return exec.AsValue(f.definition.Description), true
	case "parameters":
		return exec.AsValue(f.definition.Parameters), true
	default:
		return exec.AsValue(nil), false
	}
}

func newChatTemplateEnvironment() *exec.Environment {
	filters := exec.NewFilterSet(map[string]exec.FilterFunction{}).
		Update(gonja.DefaultEnvironment.Filters)
	fallback, _ := filters.Get("tojson")
	_ = filters.Replace(
		"tojson",
		func(
			evaluator *exec.Evaluator,
			input *exec.Value,
			parameters *exec.VarArgs,
		) *exec.Value {
			if len(parameters.Args) != 0 || len(parameters.KwArgs) != 0 {
				return fallback(evaluator, input, parameters)
			}
			var output strings.Builder
			writeChatTemplateJSON(&output, input.Interface())
			return exec.AsSafeValue(output.String())
		},
	)
	return &exec.Environment{
		Context:           gonja.DefaultEnvironment.Context,
		Filters:           filters,
		Tests:             gonja.DefaultEnvironment.Tests,
		ControlStructures: gonja.DefaultEnvironment.ControlStructures,
		Methods:           gonja.DefaultEnvironment.Methods,
	}
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
		sort.Strings(keys)
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
	// gonja v2.9 supports Jinja inline conditionals but currently rejects one
	// parenthesized form used by the pinned Gemma template when it appears as
	// the right operand of string concatenation. This equivalent boolean form
	// is safe here because both false and empty-prefix outcomes are empty.
	source = strings.ReplaceAll(
		source,
		`(first_user_prefix if loop.first else "")`,
		`((loop.first and first_user_prefix) or "")`,
	)
	// gonja's precedence for this Qwen expression differs from Jinja's.
	// Express the non-blank test through filters so an empty reasoning block
	// does not get emitted for historical assistant tool calls.
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

type boundedStringWriter struct {
	builder strings.Builder
	limit   int
}

func (w *boundedStringWriter) Write(data []byte) (int, error) {
	if len(data) > w.limit-w.builder.Len() {
		return 0, errors.New("formatted chat prompt exceeds 16 MiB")
	}
	return w.builder.Write(data)
}

func (w *boundedStringWriter) String() string {
	return w.builder.String()
}

func formatChatML(messages []ChatMessage) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("inference: chat message list is empty")
	}
	var result strings.Builder
	for index, message := range messages {
		switch message.Role {
		case "system", "user", "assistant", "tool":
		default:
			return "", fmt.Errorf("inference: chat message %d has unsupported role %q", index, message.Role)
		}
		if strings.Contains(message.Content, chatMLStart) ||
			strings.Contains(message.Content, chatMLEnd) {
			return "", fmt.Errorf("inference: chat message %d contains a ChatML boundary token", index)
		}
		result.WriteString(chatMLStart)
		result.WriteString(message.Role)
		result.WriteByte('\n')
		result.WriteString(message.Content)
		result.WriteString(chatMLEnd)
		result.WriteByte('\n')
	}
	result.WriteString(chatMLStart)
	result.WriteString("assistant\n")
	return result.String(), nil
}

func formatGemmaChat(messages []ChatMessage) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("inference: chat message list is empty")
	}
	firstUserPrefix := ""
	first := 0
	if messages[0].Role == "system" {
		if err := rejectChatBoundary(messages[0].Content, 0, gemmaTurnStart, gemmaTurnEnd); err != nil {
			return "", err
		}
		firstUserPrefix = messages[0].Content + "\n\n"
		first = 1
	}
	var result strings.Builder
	for index := first; index < len(messages); index++ {
		message := messages[index]
		wantRole := "user"
		renderedRole := "user"
		if (index-first)%2 == 1 {
			wantRole = "assistant"
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
	nextRole := "user"
	for index, message := range messages {
		if index == 0 && message.Role == "system" {
			nextRole = "user"
		} else {
			if message.Role != nextRole {
				return "", fmt.Errorf(
					"inference: Llama 3 chat message %d role %q; want %q",
					index,
					message.Role,
					nextRole,
				)
			}
			if nextRole == "user" {
				nextRole = "assistant"
			} else {
				nextRole = "user"
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
		result.WriteString(message.Role)
		result.WriteString(llama3HeaderEnd)
		result.WriteString("\n\n")
		result.WriteString(strings.TrimSpace(message.Content))
		result.WriteString(llama3EndTurn)
	}
	result.WriteString(llama3HeaderStart)
	result.WriteString("assistant")
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
