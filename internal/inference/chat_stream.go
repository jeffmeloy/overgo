package inference

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// ChatToolCallDelta: incremental function-call update.
type ChatToolCallDelta struct {
	Index            int
	Name             string
	Arguments        string
	Content          string
	ReasoningContent string
	Started          bool
}

// ChatOutputStream: structured tool-output stream.
type ChatOutputStream interface {
	Accept(string) ([]ChatToolCallDelta, error)
	Finish() (ChatMessage, error)
}

type chatOutputStream struct {
	template   string
	tools      []ChatTool
	output     strings.Builder
	snapshots  []chatToolSnapshot
	prefixSent bool
}

type chatToolSnapshot struct {
	name      string
	arguments string
}

// NewChatOutputStream: template-aware stream parser.
func NewChatOutputStream(
	template string,
	tools []ChatTool,
) ChatOutputStream {
	return &chatOutputStream{
		template: template,
		tools:    slices.Clone(tools),
	}
}

// NewChatOutputStream: loaded-template stream parser.
func (r *Runner) NewChatOutputStream(
	tools []ChatTool,
) (ChatOutputStream, error) {
	if r == nil {
		return nil, errRunnerNil
	}
	template := metadataString(r.file, "tokenizer.chat_template")
	if len(tools) != 0 {
		if toolTemplate := metadataString(
			r.file,
			"tokenizer.chat_template.tool_use",
		); toolTemplate != "" {
			template = toolTemplate
		}
	}
	return NewChatOutputStream(template, tools), nil
}

func (s *chatOutputStream) Accept(
	piece string,
) ([]ChatToolCallDelta, error) {
	s.output.WriteString(piece)
	snapshots, err := chatToolSnapshots(
		s.template,
		s.output.String(),
		s.tools,
	)
	if err != nil {
		return nil, err
	}
	deltas := make([]ChatToolCallDelta, 0, len(snapshots))
	for index, snapshot := range snapshots {
		var previous chatToolSnapshot
		if index < len(s.snapshots) {
			previous = s.snapshots[index]
		}
		if previous.name != "" && snapshot.name != previous.name {
			return nil, fmt.Errorf(
				"inference: tool call %d name changed while streaming",
				index,
			)
		}
		if !strings.HasPrefix(snapshot.arguments, previous.arguments) {
			return nil, fmt.Errorf(
				"inference: tool call %d arguments changed while streaming",
				index,
			)
		}
		started := previous.name == "" && snapshot.name != ""
		argumentDelta := snapshot.arguments[len(previous.arguments):]
		if started || argumentDelta != "" {
			delta := ChatToolCallDelta{
				Index:     index,
				Name:      snapshot.name,
				Arguments: argumentDelta,
				Started:   started,
			}
			if started && !s.prefixSent {
				prefix := s.output.String()
				if toolStart := strings.Index(prefix, toolCallOpen); toolStart >= 0 {
					var prefixErr error
					delta.ReasoningContent, delta.Content, prefixErr =
						splitChatReasoning(prefix[:toolStart])
					if prefixErr != nil {
						return nil, prefixErr
					}
				}
				s.prefixSent = true
			}
			deltas = append(deltas, delta)
		}
	}
	s.snapshots = snapshots
	return deltas, nil
}

func (s *chatOutputStream) Finish() (ChatMessage, error) {
	return parseChatOutput(s.template, s.output.String(), s.tools)
}

func chatToolSnapshots(
	template, output string,
	tools []ChatTool,
) ([]chatToolSnapshot, error) {
	hermes := strings.Contains(template, "<function=example_function_name>")
	snapshots := make([]chatToolSnapshot, 0, 1)
	remainder := output
	for {
		start := strings.Index(remainder, toolCallOpen)
		if start < 0 {
			break
		}
		payload := remainder[start+len(toolCallOpen):]
		if end := strings.Index(payload, toolCallClose); end >= 0 {
			payload = payload[:end]
			remainder = remainder[start+len(toolCallOpen)+end+len(toolCallClose):]
		} else {
			remainder = ""
		}
		var snapshot chatToolSnapshot
		var err error
		if hermes {
			snapshot, err = hermesToolSnapshot(payload, tools)
		} else {
			snapshot, err = jsonToolSnapshot(payload)
		}
		if err != nil {
			return nil, err
		}
		if snapshot.name != "" {
			snapshots = append(snapshots, snapshot)
		}
		if remainder == "" {
			break
		}
	}
	return snapshots, nil
}

func jsonToolSnapshot(payload string) (chatToolSnapshot, error) {
	nameStart, ok, err := jsonObjectFieldStart(payload, "name")
	if err != nil || !ok {
		return chatToolSnapshot{}, err
	}
	nameEnd, complete, err := jsonValuePrefixEnd(payload, nameStart)
	if err != nil || !complete {
		return chatToolSnapshot{}, err
	}
	var name string
	if err := json.Unmarshal([]byte(payload[nameStart:nameEnd]), &name); err != nil {
		return chatToolSnapshot{}, nil
	}
	argumentsStart, ok, err := jsonObjectFieldStart(payload, "arguments")
	if err != nil || !ok {
		return chatToolSnapshot{name: name}, err
	}
	argumentsEnd, _, err := jsonValuePrefixEnd(payload, argumentsStart)
	if err != nil {
		return chatToolSnapshot{}, err
	}
	return chatToolSnapshot{
		name:      name,
		arguments: payload[argumentsStart:argumentsEnd],
	}, nil
}

func jsonObjectFieldStart(
	input, wanted string,
) (int, bool, error) {
	index := skipJSONSpace(input, 0)
	if index >= len(input) {
		return 0, false, nil
	}
	if input[index] != '{' {
		return 0, false, nil
	}
	index++
	for {
		index = skipJSONSpace(input, index)
		if index >= len(input) || input[index] == '}' {
			return 0, false, nil
		}
		keyEnd, complete, err := jsonValuePrefixEnd(input, index)
		if err != nil || !complete {
			return 0, false, err
		}
		var key string
		if err := json.Unmarshal([]byte(input[index:keyEnd]), &key); err != nil {
			return 0, false, nil
		}
		index = skipJSONSpace(input, keyEnd)
		if index >= len(input) || input[index] != ':' {
			return 0, false, nil
		}
		index = skipJSONSpace(input, index+1)
		if index >= len(input) {
			return 0, false, nil
		}
		if key == wanted {
			return index, true, nil
		}
		valueEnd, complete, err := jsonValuePrefixEnd(input, index)
		if err != nil || !complete {
			return 0, false, err
		}
		index = skipJSONSpace(input, valueEnd)
		if index >= len(input) || input[index] != ',' {
			return 0, false, nil
		}
		index++
	}
}

func jsonValuePrefixEnd(
	input string,
	start int,
) (int, bool, error) {
	if start >= len(input) {
		return start, false, nil
	}
	switch input[start] {
	case '"':
		escaped := false
		for index := start + 1; index < len(input); index++ {
			switch {
			case escaped:
				escaped = false
			case input[index] == '\\':
				escaped = true
			case input[index] == '"':
				return index + 1, true, nil
			}
		}
		return len(input), false, nil
	case '{', '[':
		open := input[start]
		close := byte('}')
		if open == '[' {
			close = ']'
		}
		depth := 0
		inString := false
		escaped := false
		for index := start; index < len(input); index++ {
			char := input[index]
			if inString {
				switch {
				case escaped:
					escaped = false
				case char == '\\':
					escaped = true
				case char == '"':
					inString = false
				}
				continue
			}
			switch char {
			case '"':
				inString = true
			case open:
				depth++
			case close:
				depth--
				if depth == 0 {
					return index + 1, true, nil
				}
			}
		}
		return len(input), false, nil
	default:
		for index := start; index < len(input); index++ {
			if strings.ContainsRune(",}] \t\r\n", rune(input[index])) {
				return index, true, nil
			}
		}
		return len(input), false, nil
	}
}

func skipJSONSpace(input string, index int) int {
	for index < len(input) && strings.ContainsRune(" \t\r\n", rune(input[index])) {
		index++
	}
	return index
}

func hermesToolSnapshot(
	payload string,
	tools []ChatTool,
) (chatToolSnapshot, error) {
	const functionPrefix = "<function="
	start := strings.Index(payload, functionPrefix)
	if start < 0 {
		return chatToolSnapshot{}, nil
	}
	nameStart := start + len(functionPrefix)
	nameEnd := strings.IndexByte(payload[nameStart:], '>')
	if nameEnd < 0 {
		return chatToolSnapshot{}, nil
	}
	nameEnd += nameStart
	name := payload[nameStart:nameEnd]
	if name == "" {
		return chatToolSnapshot{}, nil
	}
	arguments := strings.Builder{}
	arguments.WriteByte('{')
	remainder := payload[nameEnd+1:]
	parameterCount := 0
	for {
		const parameterPrefix = "<parameter="
		parameterStart := strings.Index(remainder, parameterPrefix)
		functionEnd := strings.Index(remainder, "</function>")
		if functionEnd >= 0 && (parameterStart < 0 || functionEnd < parameterStart) {
			arguments.WriteByte('}')
			break
		}
		if parameterStart < 0 {
			break
		}
		keyStart := parameterStart + len(parameterPrefix)
		keyEnd := strings.IndexByte(remainder[keyStart:], '>')
		if keyEnd < 0 {
			break
		}
		keyEnd += keyStart
		key := remainder[keyStart:keyEnd]
		if parameterCount != 0 {
			arguments.WriteByte(',')
		}
		encodedKey, _ := json.Marshal(key)
		arguments.Write(encodedKey)
		arguments.WriteByte(':')
		valueStart := keyEnd + 1
		valueEnd := strings.Index(remainder[valueStart:], "</parameter>")
		complete := valueEnd >= 0
		if complete {
			valueEnd += valueStart
		} else {
			valueEnd = len(remainder)
		}
		value := strings.Trim(remainder[valueStart:valueEnd], "\r\n")
		kind := chatToolParameterType(tools, name, key)
		if kind == "" || kind == "string" {
			encodedValue, _ := json.Marshal(value)
			if complete {
				arguments.Write(encodedValue)
			} else {
				arguments.Write(encodedValue[:len(encodedValue)-1])
			}
		} else {
			arguments.WriteString(value)
		}
		parameterCount++
		if !complete {
			break
		}
		remainder = remainder[valueEnd+len("</parameter>"):]
	}
	return chatToolSnapshot{name: name, arguments: arguments.String()}, nil
}

func chatToolParameterType(
	tools []ChatTool,
	toolName, parameterName string,
) string {
	for _, tool := range tools {
		if tool.Function.Name != toolName {
			continue
		}
		properties, _ := tool.Function.Parameters["properties"].(map[string]any)
		property, _ := properties[parameterName].(map[string]any)
		kind, _ := property["type"].(string)
		return kind
	}
	return ""
}
