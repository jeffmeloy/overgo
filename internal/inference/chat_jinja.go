package inference

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"overgo/internal/jinja"
)

// JinjaGet implements jinja.Getter for the chat tool adapters, mirroring the
// gonja GetAttribute/GetItem contract (both resolve identically).

func (t chatTemplateTool) JinjaGet(key string) (any, bool) {
	switch key {
	case "type":
		return string(t.tool.Type), true
	case "function":
		return chatTemplateToolFunction{definition: t.tool.Function}, true
	}
	return nil, false
}

func (f chatTemplateToolFunction) JinjaGet(key string) (any, bool) {
	switch key {
	case "name":
		return f.definition.Name, true
	case "description":
		return f.definition.Description, true
	case "parameters":
		return f.definition.Parameters, true
	}
	return nil, false
}

func (c chatTemplateToolCall) JinjaGet(key string) (any, bool) {
	switch key {
	case "id":
		if c.call.ID == "" {
			return nil, false
		}
		return c.call.ID, true
	case "type":
		return string(c.call.Type), true
	case "function":
		return chatTemplateCallFunction{function: c.call.Function, arguments: c.arguments}, true
	}
	return nil, false
}

func (f chatTemplateCallFunction) JinjaGet(key string) (any, bool) {
	switch key {
	case "name":
		return f.function.Name, true
	case "arguments":
		return f.arguments, true
	}
	return nil, false
}

// newChatJinjaEnv builds the stdlib-only interpreter environment with the same
// runtime extensions the gonja environment provided: a tojson filter matching
// writeChatTemplateJSON (no args) / gonja's fallback (with args), strftime_now,
// and raise_exception.
func newChatJinjaEnv() *jinja.Env {
	env := jinja.New()
	env.SetFilter("tojson", func(in any, args []any, kwargs map[string]any) (any, error) {
		if len(args) == 0 && len(kwargs) == 0 {
			var output strings.Builder
			writeChatTemplateJSON(&output, in)
			return output.String(), nil
		}
		data, err := json.Marshal(normalizeJinjaJSON(in))
		if err != nil {
			return "null", nil
		}
		return strings.ReplaceAll(string(data), "'", "\\u0027"), nil
	})
	strftime := newChatTemplateStrftime(time.Now())
	env.SetGlobal("strftime_now", func(args []any, _ map[string]any) (any, error) {
		format := ""
		if len(args) >= 1 {
			if s, ok := args[0].(string); ok {
				format = s
			}
		}
		return strftime(format)
	})
	env.SetGlobal("raise_exception", func(args []any, _ map[string]any) (any, error) {
		message := ""
		if len(args) >= 1 {
			if s, ok := args[0].(string); ok {
				message = s
			}
		}
		_, err := chatTemplateRaiseException(message)
		return nil, err
	})
	return env
}

// normalizeJinjaJSON mirrors gonja's normalizeJSONValue for the tojson fallback
// path: adapters marshal to "{}" via their (unexported) fields, maps/slices
// pass through to encoding/json.
func normalizeJinjaJSON(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = normalizeJinjaJSON(val)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = normalizeJinjaJSON(val)
		}
		return out
	}
	return v
}

// formatJinjaChatNative renders the chat template with the stdlib-only
// interpreter (internal/jinja) instead of gonja.
func (r *Runner) formatJinjaChatNative(
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
	context, err := r.buildChatContext(source, messages, options)
	if err != nil {
		return "", err
	}
	env := newChatJinjaEnv()
	result, err := env.Render(source, context)
	if err != nil {
		return "", fmt.Errorf("inference: execute GGUF chat template: %w", err)
	}
	if len(result) > maxFormattedChatBytes {
		return "", errors.New("formatted chat prompt exceeds 16 MiB")
	}
	bos := context["bos_token"].(string)
	if r.vocab.AddBOS && bos != "" {
		result = strings.TrimPrefix(result, bos)
	}
	if result == "" {
		return "", errors.New("inference: chat template produced an empty prompt")
	}
	return result, nil
}
