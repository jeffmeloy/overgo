package inference

import (
	"os"
	"strings"
	"testing"

	"overgo/internal/gguf"
)

func TestChatReasoningDeclaredChannels(t *testing.T) {
	gemma, err := os.ReadFile("testdata/chat_templates/gemma_e4b.jinja")
	if err != nil {
		t.Fatal(err)
	}
	const inkling = "{{ '<|content_thinking|>' + message.reasoning_content + '<|content_text|>' + message.content }}"
	t.Run("tool_use_template_selection", func(t *testing.T) {
		runner := &Runner{preparedModel: preparedModel{file: &gguf.File{Metadata: []gguf.Metadata{
			gguf.StringMetadata("tokenizer.chat_template", string(gemma)),
			gguf.StringMetadata("tokenizer.chat_template.tool_use", "<think></think>"),
		}}}}
		tools := []ChatTool{toolWeatherDefinition()}
		output := `<think>check facts</think><tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>`
		message, err := runner.ParseChatOutput(output, tools)
		if err != nil {
			t.Fatal(err)
		}
		stream, err := runner.NewChatOutputStream(tools)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := stream.Accept(output); err != nil {
			t.Fatal(err)
		}
		streamed, err := stream.Finish()
		if err != nil {
			t.Fatal(err)
		}
		for _, value := range []ChatMessage{message, streamed} {
			if value.ReasoningContent != "check facts" || value.Content != "" || len(value.ToolCalls) != 1 || value.ToolCalls[0].Function.Name != "weather" {
				t.Fatalf("tool-use template was not selected: %+v", value)
			}
		}
	})
	t.Run("unfinished_stream", func(t *testing.T) {
		stream := NewChatOutputStream(string(gemma), nil)
		deltas, err := stream.Accept("<|channel>thought\nunfinished")
		if err != nil || len(deltas) != 0 {
			t.Fatalf("incomplete reasoning: deltas=%v err=%v", deltas, err)
		}
		if _, err := stream.Finish(); err == nil {
			t.Fatal("unfinished reasoning was accepted as complete")
		}
	})
	for _, test := range []struct {
		name, template, output, reasoning, content string
		invalid                                    bool
	}{
		{"declared_thought", string(gemma), "<|channel>thought\ncheck facts\n<channel|>answer", "check facts", "answer", false},
		{"unicode_bytes", string(gemma), "<|channel>thought\n確認 🙂\n<channel|>réponse π", "確認 🙂", "réponse π", false},
		{"thought_prompt_open", string(gemma), "check facts\n<channel|>answer", "check facts", "answer", false},
		{"declared_content_channel", inkling, "<|content_thinking|>check facts<|content_text|>answer", "check facts", "answer", false},
		{"content_channel_prompt_open", inkling, "check facts<|content_text|>answer", "check facts", "answer", false},
		{"legacy_think", "", "<think>\ncheck facts\n</think>\nanswer", "check facts", "answer", false},
		{"legacy_prompt_open", "", "check facts</think>answer", "check facts", "answer", false},
		{"undeclared_channel", "", "<|channel>thought\ntext<channel|>answer", "", "<|channel>thought\ntext<channel|>answer", false},
		{"plain_bytes", string(gemma), "  ordinary answer\n", "", "  ordinary answer\n", false},
		{"foreign_marker_literal", string(gemma), "<think>ordinary literal</think>", "", "<think>ordinary literal</think>", false},
		{"empty_thought", string(gemma), "<|channel>thought\n<channel|>answer", "", "answer", false},
		{"unterminated", string(gemma), "<|channel>thought\nunfinished", "", "", true},
		{"text_before_thought", string(gemma), "prefix<|channel>thought\ntext<channel|>answer", "", "", true},
		{"repeated_channel", string(gemma), "<|channel>thought\ntext<channel|>answer<channel|>", "", "", true},
		{"ambiguous_declaration", string(gemma) + inkling, "ordinary", "", "", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			runner := &Runner{preparedModel: preparedModel{file: &gguf.File{Metadata: []gguf.Metadata{
				gguf.StringMetadata("tokenizer.chat_template", test.template),
			}}}}
			message, err := runner.ParseChatOutput(test.output, nil)
			if test.invalid {
				if err == nil {
					t.Fatalf("accepted invalid reasoning output: %+v", message)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if message.ReasoningContent != test.reasoning || message.Content != test.content || len(message.ToolCalls) != 0 {
				t.Fatalf("reasoning=%q content=%q calls=%v; want %q / %q", message.ReasoningContent, message.Content, message.ToolCalls, test.reasoning, test.content)
			}
			for split := range len(test.output) + 1 {
				stream, err := runner.NewChatOutputStream(nil)
				if err != nil {
					t.Fatal(err)
				}
				for _, piece := range []string{test.output[:split], test.output[split:]} {
					if _, err := stream.Accept(piece); err != nil {
						t.Fatalf("split %d: %v", split, err)
					}
				}
				got, err := stream.Finish()
				if err != nil {
					t.Fatal(err)
				}
				if got.ReasoningContent != test.reasoning || got.Content != test.content || len(got.ToolCalls) != 0 {
					t.Fatalf("split %d disagrees with buffered result: %+v", split, got)
				}
			}
		})
	}
	t.Run("reasoning_before_tool_scanning", func(t *testing.T) {
		for _, syntax := range []struct{ template, open, close string }{
			{"<think></think>", "<think>", "</think>"},
			{string(gemma), "<|channel>thought\n", "<channel|>"},
			{inkling, "<|content_thinking|>", "<|content_text|>"},
		} {
			stream := NewChatOutputStream(syntax.template, nil)
			thought := `example: <tool_call>{"name":"example","arguments":{}}</tool_call>`
			prefix := syntax.open + thought
			for index := range len(prefix) {
				deltas, err := stream.Accept(prefix[index : index+1])
				if err != nil {
					t.Fatal(err)
				}
				for _, delta := range deltas {
					if delta.Started || delta.Name != "" || delta.Arguments != "" {
						t.Fatalf("reasoning emitted a tool delta: %+v", delta)
					}
				}
			}
			actualCall := `<tool_call>{"name":"actual","arguments":{"city":"Paris"}}</tool_call>`
			var started int
			var arguments strings.Builder
			for _, piece := range []string{syntax.close, actualCall} {
				deltas, err := stream.Accept(piece)
				if err != nil {
					t.Fatal(err)
				}
				for _, delta := range deltas {
					if delta.Started {
						started++
						if delta.Name != "actual" {
							t.Fatalf("started %q", delta.Name)
						}
					}
					arguments.WriteString(delta.Arguments)
				}
			}
			message, err := stream.Finish()
			if err != nil {
				t.Fatal(err)
			}
			if started != 1 || arguments.String() != `{"city":"Paris"}` || message.ReasoningContent != thought || len(message.ToolCalls) != 1 || message.ToolCalls[0].Function.Name != "actual" {
				t.Fatalf("reasoning/tool separation: starts=%d arguments=%q message=%+v", started, arguments.String(), message)
			}
		}
	})
}
