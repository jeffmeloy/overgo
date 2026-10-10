package inference

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"overgo/internal/jinja"
)

// What one render of a chat template may take. The template comes out of a
// model file, so nothing here wrote it, and the engine refuses one that goes
// past either limit. Measured 2026-10-09 on the served models' templates: for
// a conversation of the test battery, a few messages long, at most 1,295
// steps and 9,435 bytes. For two thousand messages, half a megabyte of text
// and a full context of the models served here, at most 160 thousand steps
// and 5 megabytes for every template but Gemma's two, which look ahead from
// each message to the end and take 12 million steps and 37 megabytes; at four
// thousand messages they take 49 million and 138. The limits are the next
// powers of two above that, so they refuse no conversation a served model's
// context can hold, and end a template that runs away within seconds and a
// few hundred megabytes.
const (
	chatTemplateSteps = 1 << 26
	chatTemplateBytes = 1 << 28
)

// newChatJinjaEnv builds the template environment with the two functions
// transformers gives a chat template beside Jinja2's own: strftime_now and
// raise_exception. tojson, which transformers also defines its own way, is
// the engine's.
func newChatJinjaEnv() *jinja.Env {
	env := jinja.New()
	env.SetLimits(jinja.Limits{Steps: chatTemplateSteps, Bytes: chatTemplateBytes})
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

// formatJinjaChatNative renders the chat template with overgo's own engine
// (internal/jinja).
func (r *Runner) formatJinjaChatNative(
	source string,
	messages []ChatMessage,
	options ChatFormatOptions,
) (string, error) {
	if len(messages) == 0 {
		return "", errors.New("inference: chat message list is empty")
	}
	context, err := r.buildChatContext(messages, options)
	if err != nil {
		return "", err
	}
	env := newChatJinjaEnv()
	result, err := env.Render(source, context)
	if err != nil {
		return "", fmt.Errorf("inference: execute GGUF chat template: %w", err)
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
