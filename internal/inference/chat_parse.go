package inference

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"overgo/internal/strictjson"
)

const (
	toolCallOpen  = "<tool_call>"
	toolCallClose = "</tool_call>"
)

// ParseChatOutput separates reasoning, visible content, and tool calls using
// syntax advertised by loaded GGUF chat template
func (r *Runner) ParseChatOutput(
	output string,
	tools []ChatTool,
) (ChatMessage, error) {
	if r == nil {
		return ChatMessage{}, errRunnerNil
	}
	source := metadataString(r.file, "tokenizer.chat_template")
	if len(tools) != 0 {
		if toolSource := metadataString(
			r.file,
			"tokenizer.chat_template.tool_use",
		); toolSource != "" {
			source = toolSource
		}
	}
	return parseChatOutput(source, output, tools)
}

func parseChatOutput(
	template, output string,
	tools []ChatTool,
) (ChatMessage, error) {
	message := ChatMessage{Role: ChatRoleAssistant}
	reasoning, body, err := splitChatReasoning(output)
	if err != nil {
		return ChatMessage{}, err
	}
	message.ReasoningContent = reasoning
	firstCall := strings.Index(body, toolCallOpen)
	if firstCall < 0 {
		if strings.Contains(body, toolCallClose) {
			return ChatMessage{}, errors.New(
				"inference: tool-call output has an unmatched closing tag",
			)
		}
		message.Content = body
		return message, nil
	}
	message.Content = strings.TrimSpace(body[:firstCall])
	remainder := body[firstCall:]
	hermes := strings.Contains(template, "<function=example_function_name>")
	for len(remainder) != 0 {
		remainder = strings.TrimLeft(remainder, " \t\r\n")
		if remainder == "" {
			break
		}
		if !strings.HasPrefix(remainder, toolCallOpen) {
			return ChatMessage{}, errors.New(
				"inference: text after a tool call is not supported",
			)
		}
		end := strings.Index(remainder, toolCallClose)
		if end < 0 {
			return ChatMessage{}, errors.New(
				"inference: tool-call output has an unterminated block",
			)
		}
		payload := strings.TrimSpace(remainder[len(toolCallOpen):end])
		var call ChatToolCall
		if hermes {
			call, err = parseHermesToolCall(payload, tools)
		} else {
			call, err = parseJSONToolCall(payload, tools)
		}
		if err != nil {
			return ChatMessage{}, fmt.Errorf(
				"inference: parse tool call %d: %w",
				len(message.ToolCalls),
				err,
			)
		}
		message.ToolCalls = append(message.ToolCalls, call)
		remainder = remainder[end+len(toolCallClose):]
	}
	return message, nil
}

func splitChatReasoning(output string) (reasoning, content string, err error) {
	const (
		open  = "<think>"
		close = "</think>"
	)
	openIndex := strings.Index(output, open)
	closeIndex := strings.Index(output, close)
	switch {
	case openIndex >= 0:
		if closeIndex < openIndex {
			return "", "", errors.New(
				"inference: reasoning output has an unterminated think block",
			)
		}
		prefix := output[:openIndex]
		if strings.TrimSpace(prefix) != "" {
			return "", "", errors.New(
				"inference: text before a reasoning block is not supported",
			)
		}
		reasoning = strings.Trim(output[openIndex+len(open):closeIndex], "\r\n")
		content = strings.TrimLeft(output[closeIndex+len(close):], "\r\n")
	case closeIndex >= 0:
		// Some templates put opening tag in generation prompt;
		// decoder output therefore starts inside reasoning block
		reasoning = strings.Trim(output[:closeIndex], "\r\n")
		content = strings.TrimLeft(output[closeIndex+len(close):], "\r\n")
	default:
		content = output
	}
	if strings.Contains(content, open) || strings.Contains(content, close) {
		return "", "", errors.New(
			"inference: reasoning output contains unmatched think tags",
		)
	}
	return reasoning, content, nil
}

func parseJSONToolCall(
	payload string,
	tools []ChatTool,
) (ChatToolCall, error) {
	var wire struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := strictjson.Decode(strings.NewReader(payload), &wire); err != nil {
		return ChatToolCall{}, err
	}
	if wire.Name == "" {
		return ChatToolCall{}, errors.New("function name is required")
	}
	if len(tools) != 0 && !chatToolExists(tools, wire.Name) {
		return ChatToolCall{}, fmt.Errorf(
			"function %q is not present in tools",
			wire.Name,
		)
	}
	if len(wire.Arguments) == 0 {
		return ChatToolCall{}, errors.New("function arguments are required")
	}
	arguments, err := compactToolArguments(wire.Arguments)
	if err != nil {
		return ChatToolCall{}, err
	}
	var parsedArguments any
	if err := json.Unmarshal(wire.Arguments, &parsedArguments); err != nil {
		return ChatToolCall{}, err
	}
	if err := validateParsedToolArguments(
		wire.Name,
		parsedArguments,
		tools,
	); err != nil {
		return ChatToolCall{}, err
	}
	return ChatToolCall{
		Type: ChatToolTypeFunction,
		Function: ChatToolFunction{
			Name:      wire.Name,
			Arguments: arguments,
		},
	}, nil
}

func parseHermesToolCall(
	payload string,
	tools []ChatTool,
) (ChatToolCall, error) {
	const functionPrefix = "<function="
	if !strings.HasPrefix(payload, functionPrefix) {
		return ChatToolCall{}, errors.New("function opening tag is required")
	}
	nameEnd := strings.IndexByte(payload[len(functionPrefix):], '>')
	if nameEnd < 0 {
		return ChatToolCall{}, errors.New("function opening tag is unterminated")
	}
	nameEnd += len(functionPrefix)
	name := payload[len(functionPrefix):nameEnd]
	if name == "" || strings.ContainsAny(name, "<>\r\n") {
		return ChatToolCall{}, errors.New("function name is invalid")
	}
	if len(tools) != 0 && !chatToolExists(tools, name) {
		return ChatToolCall{}, fmt.Errorf(
			"function %q is not present in tools",
			name,
		)
	}
	const functionClose = "</function>"
	bodyStart := nameEnd + 1
	functionEnd := strings.LastIndex(payload, functionClose)
	if functionEnd < bodyStart ||
		strings.TrimSpace(payload[functionEnd+len(functionClose):]) != "" {
		return ChatToolCall{}, errors.New("function closing tag is required")
	}
	parameters, err := parseHermesParameters(
		name,
		payload[bodyStart:functionEnd],
		tools,
	)
	if err != nil {
		return ChatToolCall{}, err
	}
	if err := validateParsedToolArguments(name, parameters, tools); err != nil {
		return ChatToolCall{}, err
	}
	arguments, err := json.Marshal(parameters)
	if err != nil {
		return ChatToolCall{}, err
	}
	return ChatToolCall{
		Type: ChatToolTypeFunction,
		Function: ChatToolFunction{
			Name:      name,
			Arguments: string(arguments),
		},
	}, nil
}

func chatToolExists(tools []ChatTool, name string) bool {
	for _, tool := range tools {
		if tool.Function.Name == name {
			return true
		}
	}
	return false
}

func validateParsedToolArguments(
	toolName string,
	arguments any,
	tools []ChatTool,
) error {
	if len(tools) == 0 {
		return nil
	}
	var definition *ChatToolDefinition
	for index := range tools {
		if tools[index].Function.Name == toolName {
			definition = &tools[index].Function
			break
		}
	}
	if definition == nil {
		return fmt.Errorf("function %q is not present in tools", toolName)
	}
	kind, _ := definition.Parameters["type"].(string)
	if kind != "" && kind != "object" {
		return nil
	}
	object, ok := arguments.(map[string]any)
	if !ok {
		return errors.New("function arguments must be an object")
	}
	properties, _ := definition.Parameters["properties"].(map[string]any)
	if additional, exists := definition.Parameters["additionalProperties"]; exists {
		if allowed, ok := additional.(bool); ok && !allowed {
			for name := range object {
				if _, known := properties[name]; !known {
					return fmt.Errorf("unknown function argument %q", name)
				}
			}
		}
	}
	for _, required := range schemaRequiredNames(
		definition.Parameters["required"],
	) {
		if _, exists := object[required]; !exists {
			return fmt.Errorf(
				"required function argument %q is missing",
				required,
			)
		}
	}
	for name, value := range object {
		property, _ := properties[name].(map[string]any)
		expected, _ := property["type"].(string)
		if !toolArgumentTypeMatches(expected, value) {
			return fmt.Errorf(
				"function argument %q does not match type %q",
				name,
				expected,
			)
		}
	}
	return nil
}

func schemaRequiredNames(value any) []string {
	switch typed := value.(type) {
	case []string:
		return typed
	case []any:
		result := make([]string, 0, len(typed))
		for _, item := range typed {
			if name, ok := item.(string); ok {
				result = append(result, name)
			}
		}
		return result
	default:
		return nil
	}
}

func toolArgumentTypeMatches(expected string, value any) bool {
	switch expected {
	case "", "null":
		return expected == "" || value == nil
	case "string":
		_, ok := value.(string)
		return ok
	case "integer":
		switch number := value.(type) {
		case int, int8, int16, int32, int64,
			uint, uint8, uint16, uint32, uint64:
			return true
		case float64:
			return number == float64(int64(number))
		default:
			return false
		}
	case "number":
		switch value.(type) {
		case int, int8, int16, int32, int64,
			uint, uint8, uint16, uint32, uint64,
			float32, float64:
			return true
		default:
			return false
		}
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "object":
		_, ok := value.(map[string]any)
		return ok
	case "array":
		_, ok := value.([]any)
		return ok
	default:
		return true
	}
}

func parseHermesParameters(
	toolName, body string,
	tools []ChatTool,
) (map[string]any, error) {
	parameters := make(map[string]any)
	remainder := strings.TrimSpace(body)
	const parameterPrefix = "<parameter="
	const parameterClose = "</parameter>"
	for remainder != "" {
		if !strings.HasPrefix(remainder, parameterPrefix) {
			return nil, errors.New("parameter opening tag is required")
		}
		nameEnd := strings.IndexByte(remainder[len(parameterPrefix):], '>')
		if nameEnd < 0 {
			return nil, errors.New("parameter opening tag is unterminated")
		}
		nameEnd += len(parameterPrefix)
		name := remainder[len(parameterPrefix):nameEnd]
		if name == "" || strings.ContainsAny(name, "<>\r\n") {
			return nil, errors.New("parameter name is invalid")
		}
		if _, duplicate := parameters[name]; duplicate {
			return nil, fmt.Errorf("parameter %q is duplicated", name)
		}
		end := strings.Index(remainder[nameEnd+1:], parameterClose)
		if end < 0 {
			return nil, fmt.Errorf("parameter %q is unterminated", name)
		}
		end += nameEnd + 1
		value := strings.Trim(remainder[nameEnd+1:end], "\r\n")
		parameters[name] = coerceToolParameter(toolName, name, value, tools)
		remainder = strings.TrimSpace(
			remainder[end+len(parameterClose):],
		)
	}
	return parameters, nil
}

func coerceToolParameter(
	toolName, parameter, value string,
	tools []ChatTool,
) any {
	for _, tool := range tools {
		if tool.Function.Name != toolName {
			continue
		}
		properties, _ := tool.Function.Parameters["properties"].(map[string]any)
		property, _ := properties[parameter].(map[string]any)
		kind, _ := property["type"].(string)
		switch kind {
		case "integer":
			if parsed, err := strconv.ParseInt(value, 10, 64); err == nil {
				return parsed
			}
		case "number":
			if parsed, err := strconv.ParseFloat(value, 64); err == nil {
				return parsed
			}
		case "boolean":
			if parsed, err := strconv.ParseBool(value); err == nil {
				return parsed
			}
		case "array", "object":
			var parsed any
			if json.Unmarshal([]byte(value), &parsed) == nil {
				return parsed
			}
		}
		break
	}
	return value
}

func compactToolArguments(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || !json.Valid(raw) {
		return "", errors.New("function arguments must be valid JSON")
	}
	var output bytes.Buffer
	if err := json.Compact(&output, raw); err != nil {
		return "", err
	}
	return output.String(), nil
}
