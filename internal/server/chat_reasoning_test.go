package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

type declaredReasoningGenerator struct {
	*fakeGenerator
	template, prompt string
	preparedIDs      []tokenizer.TokenID
	afterToken       func()
	lastError        error
	controlPieces    map[tokenizer.TokenID]string
}

type decodedReasoningStream struct {
	inference.ChatOutputStream
	decoder inference.TokenEventDecoder
}

func (s decodedReasoningStream) TokenEventDecoder() inference.TokenEventDecoder { return s.decoder }

func (g *declaredReasoningGenerator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	if g.afterToken != nil || options.TokenEventDecoder != nil {
		emit := options.OnToken
		options.OnToken = func(event inference.TokenEvent) error {
			if options.TokenEventDecoder != nil {
				var err error
				event.Piece, err = options.TokenEventDecoder(event.ID)
				if err != nil {
					return err
				}
			}
			if emit != nil {
				if err := emit(event); err != nil {
					return err
				}
			}
			if g.afterToken != nil {
				g.afterToken()
			}
			return nil
		}
	}
	ids, text, err := g.fakeGenerator.Generate(ctx, prompt, options)
	g.lastError = err
	return ids, text, err
}

func (g *declaredReasoningGenerator) FormatChat([]inference.ChatMessage) (string, error) {
	return g.prompt, nil
}
func (g *declaredReasoningGenerator) FormatChatWithOptions(messages []inference.ChatMessage, options inference.ChatFormatOptions) (string, error) {
	if _, err := g.fakeGenerator.FormatChatWithOptions(messages, options); err != nil {
		return "", err
	}
	return g.prompt, nil
}
func (g *declaredReasoningGenerator) NewChatOutputStreamForPrompt(prompt string, ids []tokenizer.TokenID, tools []inference.ChatTool) (inference.ChatOutputStream, error) {
	g.preparedIDs = slices.Clone(ids)
	stream, err := inference.NewChatOutputStreamForPrompt(g.template, prompt, tools)
	if err != nil {
		return nil, err
	}
	if g.controlPieces != nil {
		return decodedReasoningStream{ChatOutputStream: stream, decoder: func(id tokenizer.TokenID) (string, error) { return g.controlPieces[id], nil }}, nil
	}
	return stream, nil
}

func TestChatReasoningDeclaredChannels(t *testing.T) {
	const gemma = "<|channel>thought\n<channel|>"
	t.Run("token_event_decoder_reaches_generation", func(t *testing.T) {
		for _, streaming := range []bool{false, true} {
			g := &declaredReasoningGenerator{fakeGenerator: &fakeGenerator{pieces: []string{"", "", "", ""}}, template: "<think></think>", prompt: "assistant", controlPieces: map[tokenizer.TokenID]string{30: "<think>", 31: "secret", 32: "</think>", 33: "answer"}}
			message, _, _ := runDeclaredReasoningRequest(t, newTestHandler(t, g), streaming, len(g.pieces)+1, "")
			if message.ReasoningContent != "secret" || message.Content != "answer" {
				t.Fatalf("token decoder not applied: %+v", message)
			}
		}
	})
	t.Run("progress_and_cancellation", func(t *testing.T) {
		for _, reasoning := range []bool{false, true} {
			for _, cancelEarly := range []bool{false, true} {
				g := &declaredReasoningGenerator{fakeGenerator: &fakeGenerator{pieces: []string{"first", " later"}}, template: gemma, prompt: "assistant"}
				field := `"content":"first"`
				if reasoning {
					g.prompt += "<|channel>thought\n"
					field = `"reasoning_content":"first"`
				}
				response := httptest.NewRecorder()
				ctx, cancel := context.WithCancelCause(t.Context())
				observed := false
				g.afterToken = func() {
					if observed {
						return
					}
					observed = true
					if !strings.Contains(response.Body.String(), field) {
						t.Errorf("output was withheld until completion: %s", response.Body.String())
					}
					if cancelEarly {
						cancel(context.Canceled)
					}
				}
				request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"question"}],"max_tokens":3,"stream":true}`)).WithContext(ctx)
				newTestHandler(t, g).ServeHTTP(response, request)
				cancel(nil)
				if !observed {
					t.Fatal("no progressive output")
				}
				if cancelEarly && (!errors.Is(g.lastError, context.Canceled) || strings.Contains(response.Body.String(), " later")) {
					t.Fatalf("cancellation: error=%v body=%s", g.lastError, response.Body.String())
				}
			}
		}
	})
	t.Run("legacy_provider_fallback", func(t *testing.T) {
		for _, streaming := range []bool{false, true} {
			message, _, _ := runDeclaredReasoningRequest(t, newTestHandler(t, &fakeGenerator{pieces: []string{"<think>literal</think>"}}), streaming, 2, "")
			if message.Content != "<think>literal</think>" || message.ReasoningContent != "" {
				t.Fatalf("provider fallback changed: %+v", message)
			}
		}
	})
	for _, test := range []struct{ name, template, prompt, output, reasoning, content, fields string }{
		{"declared", gemma, "assistant", "<|channel>thought\ncheck facts<channel|>answer", "check facts", "answer", ""},
		{"prompt_open", gemma, "assistant<|channel>thought\n", "check facts<channel|>answer", "check facts", "answer", ""},
		{"prompt_closed", "<think></think>", "assistant<think>\n\n</think>\n", "\nanswer", "", "\nanswer", ""},
		{"inkling", "<|content_thinking|><|content_text|>", "assistant", "<|content_thinking|>check<|content_text|>answer", "check", "answer", ""},
		{"plain", gemma, "assistant", "  answer 🙂\n", "", "  answer 🙂\n", ""},
		{"literal", gemma, "assistant", "guide: <|channel>thought\nquoted<channel|>text", "", "guide: <|channel>thought\nquoted<channel|>text", ""},
		{"undeclared", "", "assistant", "<think>literal</think>", "", "<think>literal</think>", ""},
		{"no_tools_literal", gemma, "assistant", "<tool_call>example</tool_call>", "", "<tool_call>example</tool_call>", ""},
		{"tool_choice_none", gemma, "assistant", "<|channel>thought\nr<channel|><tool_call>example</tool_call>", "r", "<tool_call>example</tool_call>", `,"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}],"tool_choice":"none"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			for split := range len(test.output) + 1 {
				for _, streaming := range []bool{false, true} {
					g := &declaredReasoningGenerator{fakeGenerator: &fakeGenerator{pieces: []string{test.output[:split], test.output[split:]}}, template: test.template, prompt: test.prompt}
					h := newTestHandler(t, g)
					message, _, _ := runDeclaredReasoningRequest(t, h, streaming, 3, test.fields)
					if message.ReasoningContent != test.reasoning || message.Content != test.content || len(message.ToolCalls) != 0 {
						t.Fatalf("split=%d stream=%v got=%+v want=%q / %q", split, streaming, message, test.reasoning, test.content)
					}
					if !slices.Equal(g.preparedIDs, g.fakeGenerator.promptIDs) || len(g.preparedIDs) == 0 {
						t.Fatal("parser did not receive actual prepared IDs")
					}
				}
			}
		})
	}
	t.Run("bytewise_unicode", func(t *testing.T) {
		output := "<think>🙂</think>π"
		var pieces []string
		for index := range len(output) {
			pieces = append(pieces, output[index:index+1])
		}
		g := &declaredReasoningGenerator{fakeGenerator: &fakeGenerator{pieces: pieces}, template: "<think></think>", prompt: "assistant"}
		h, err := New(Config{ModelID: testModelID, MaxTokens: len(pieces) + 1, DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP, Analysis: testAnalysisPolicy}, g)
		if err != nil {
			t.Fatal(err)
		}
		message, _, _ := runDeclaredReasoningRequest(t, h, true, len(pieces)+1, "")
		if message.ReasoningContent != "🙂" || message.Content != "π" || len(message.ToolCalls) != 0 {
			t.Fatalf("message=%+v", message)
		}
	})
	t.Run("budgets_and_stops", func(t *testing.T) {
		for _, streaming := range []bool{false, true} {
			g := &declaredReasoningGenerator{fakeGenerator: &fakeGenerator{pieces: []string{"unfinished", " ignored"}}, template: gemma, prompt: "assistant<|channel>thought\n"}
			message, finish, usage := runDeclaredReasoningRequest(t, newTestHandler(t, g), streaming, 1, "")
			if message.ReasoningContent != "unfinished" || message.Content != "" || finish != "length" || usage.CompletionTokens != 1 {
				t.Fatalf("budget: %+v %s %+v", message, finish, usage)
			}
			g = &declaredReasoningGenerator{fakeGenerator: &fakeGenerator{pieces: []string{"checkHALT", "ignored"}}, template: gemma, prompt: "assistant<|channel>thought\n"}
			message, finish, usage = runDeclaredReasoningRequest(t, newTestHandler(t, g), streaming, 3, `,"stop":["HALT"]`)
			if message.ReasoningContent != "check" || message.Content != "" || finish != "stop" || usage.CompletionTokens != 1 {
				t.Fatalf("stop: %+v %s %+v", message, finish, usage)
			}
		}
	})
	t.Run("eager_constraint_content", func(t *testing.T) {
		for _, streaming := range []bool{false, true} {
			g := &declaredReasoningGenerator{fakeGenerator: &fakeGenerator{pieces: []string{`{"ok":true}`}}, template: "<think></think>", prompt: "assistant<think>\n"}
			message, _, _ := runDeclaredReasoningRequest(t, newTestHandler(t, g), streaming, 2, `,"response_format":{"type":"json_object"}`)
			if message.Content != `{"ok":true}` || message.ReasoningContent != "" {
				t.Fatalf("constrained answer hidden: %+v", message)
			}
		}
	})
	t.Run("reasoning_tool_example", func(t *testing.T) {
		thought := `example <tool_call>{"name":"example","arguments":{}}</tool_call>`
		call := `<tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>`
		for _, streaming := range []bool{false, true} {
			g := &declaredReasoningGenerator{fakeGenerator: &fakeGenerator{pieces: []string{thought, "</think> Hello " + call}}, template: "<think></think>", prompt: "assistant<think>\n"}
			message, finish, _ := runDeclaredReasoningRequest(t, newTestHandler(t, g), streaming, 3, `,"tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]`)
			if message.ReasoningContent != thought || message.Content != " Hello " || finish != "tool_calls" || len(message.ToolCalls) != 1 || message.ToolCalls[0].Function.Name != "weather" || message.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` || message.ToolCalls[0].ID == "" {
				t.Fatalf("tool separation: %+v finish=%s", message, finish)
			}
		}
	})
}

func runDeclaredReasoningRequest(t *testing.T, handler *Handler, streaming bool, maxTokens int, fields string) (inference.ChatMessage, string, completionUsage) {
	t.Helper()
	response := httptest.NewRecorder()
	body := fmt.Sprintf(`{"messages":[{"role":"user","content":"question"}],"max_tokens":%d,"stream":%v%s}`, maxTokens, streaming, fields)
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if !streaming {
		var result chatResponse
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if len(result.Choices) != 1 {
			t.Fatalf("choices=%+v", result.Choices)
		}
		return result.Choices[0].Message, result.Choices[0].FinishReason, result.Usage
	}
	message := inference.ChatMessage{Role: inference.ChatRoleAssistant}
	var finish string
	var usage completionUsage
	for line := range strings.SplitSeq(response.Body.String(), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var wire map[string]json.RawMessage
		if err := json.Unmarshal([]byte(data), &wire); err != nil {
			t.Fatal(err)
		}
		if wire["error"] != nil {
			t.Fatalf("stream error: %s", data)
		}
		var chunk chatStreamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatal(err)
		}
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
		for _, choice := range chunk.Choices {
			message.Content += choice.Delta.Content
			message.ReasoningContent += choice.Delta.ReasoningContent
			if choice.FinishReason != nil {
				finish = *choice.FinishReason
			}
			for _, call := range choice.Delta.ToolCalls {
				if call.ID == "" && call.Function.Name == "" && call.Function.Arguments == "" {
					t.Fatalf("dummy tool delta: %s", data)
				}
				for len(message.ToolCalls) <= call.Index {
					message.ToolCalls = append(message.ToolCalls, inference.ChatToolCall{})
				}
				target := &message.ToolCalls[call.Index]
				if call.ID != "" {
					target.ID = call.ID
					target.Type = call.Type
					target.Function.Name = call.Function.Name
				}
				target.Function.Arguments += call.Function.Arguments
			}
		}
	}
	if finish == "" {
		t.Fatalf("missing terminal chunk: %s", response.Body.String())
	}
	return message, finish, usage
}
