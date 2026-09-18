package inference

import (
	"strings"
	"testing"
	"unicode/utf8"

	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
)

func TestChatReasoningStreamPromptState(t *testing.T) {
	const template = "<|channel>thought\n<channel|>"
	t.Run("native_control_token_delivery", func(t *testing.T) {
		runner := &Runner{preparedModel: preparedModel{
			file: &gguf.File{Metadata: []gguf.Metadata{gguf.StringMetadata("tokenizer.chat_template", template)}},
			vocab: &tokenizer.Vocab{Model: "llama", EOS: 6, EOT: tokenizer.NullToken, EOM: tokenizer.NullToken, Tokens: []tokenizer.Token{
				{Text: "<unrelated>", Type: tokenizer.TokenControl},
				{Text: "<|channel>", Type: tokenizer.TokenControl},
				{Text: "thought\n", Type: tokenizer.TokenNormal},
				{Text: "secret", Type: tokenizer.TokenNormal},
				{Text: "<channel|>", Type: tokenizer.TokenControl},
				{Text: "answer", Type: tokenizer.TokenNormal},
				{Text: "</s>", Type: tokenizer.TokenControl},
			}},
		}}
		stream, err := runner.NewChatOutputStreamForPrompt("assistant", nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		options := GenerateOptions{OnToken: func(event TokenEvent) error { _, err := stream.Accept(event.Piece); return err }}
		options.TokenEventDecoder = stream.(ChatOutputTokenDecoder).TokenEventDecoder()
		var generated strings.Builder
		for index := range len(runner.vocab.Tokens) {
			event := TokenEvent{ID: tokenizer.TokenID(index)}
			stopped, err := runner.deliverGenerationToken(&event, options, &generated)
			if err != nil {
				t.Fatal(err)
			}
			if stopped != (index == len(runner.vocab.Tokens)-1) {
				t.Fatalf("stop changed at %d", index)
			}
		}
		message, err := stream.Finish()
		if err != nil || message.ReasoningContent != "secret" || message.Content != "answer" {
			t.Fatalf("native token delivery: %+v %v", message, err)
		}
	})
	t.Run("public_tool_stream_byte_boundaries", func(t *testing.T) {
		stream, err := NewChatOutputStreamForPrompt("<think></think>", "assistant<think>\n", []ChatTool{toolWeatherDefinition()})
		if err != nil {
			t.Fatal(err)
		}
		thought := `example <tool_call>{"name":"example","arguments":{}}</tool_call>`
		output := thought + `</think> Hello <tool_call>{"name":"weather","arguments":{"city":"東京"}}</tool_call>`
		var reasoning, content, arguments strings.Builder
		starts := 0
		for index := range len(output) {
			deltas, err := stream.Accept(output[index : index+1])
			if err != nil {
				t.Fatal(err)
			}
			for _, delta := range deltas {
				reasoning.WriteString(delta.ReasoningContent)
				content.WriteString(delta.Content)
				arguments.WriteString(delta.Arguments)
				if delta.Started {
					starts++
					if delta.Name != "weather" {
						t.Fatalf("wrong tool: %+v", delta)
					}
				}
			}
		}
		message, err := stream.Finish()
		if err != nil {
			t.Fatal(err)
		}
		if starts != 1 || reasoning.String() != thought || content.String() != " Hello " || arguments.String() != `{"city":"東京"}` || message.ReasoningContent != reasoning.String() || message.Content != content.String() || len(message.ToolCalls) != 1 || message.ToolCalls[0].Function.Arguments != arguments.String() {
			t.Fatalf("stream/final mismatch: starts=%d reasoning=%q content=%q args=%q final=%+v", starts, reasoning.String(), content.String(), arguments.String(), message)
		}
	})
	t.Run("runner_prepared_tokens_and_tool_template", func(t *testing.T) {
		runner := &Runner{preparedModel: preparedModel{
			file: &gguf.File{Metadata: []gguf.Metadata{
				gguf.StringMetadata("tokenizer.chat_template", template),
				gguf.StringMetadata("tokenizer.chat_template.tool_use", "<think></think>"),
			}},
			vocab: &tokenizer.Vocab{Model: "llama", Tokens: []tokenizer.Token{
				{Text: "<|channel>thought", Type: tokenizer.TokenControl},
				{Text: "<think>", Type: tokenizer.TokenControl},
			}},
		}}
		stream, err := runner.NewChatOutputStreamForPrompt("original unprojected text", []tokenizer.TokenID{0}, nil)
		if err != nil {
			t.Fatal(err)
		}
		deltas, err := stream.Accept("確認<channel|>answer")
		if err != nil {
			t.Fatal(err)
		}
		if len(deltas) != 1 || deltas[0].ReasoningContent != "確認" || deltas[0].Content != "answer" || deltas[0].Started {
			t.Fatalf("actual token state: %+v", deltas)
		}
		message, err := stream.Finish()
		if err != nil || message.ReasoningContent != "確認" || message.Content != "answer" {
			t.Fatalf("finish: %+v %v", message, err)
		}
		stream, err = runner.NewChatOutputStreamForPrompt("original text", []tokenizer.TokenID{1}, []ChatTool{toolWeatherDefinition()})
		if err != nil {
			t.Fatal(err)
		}
		thought := `example <tool_call>{"name":"example","arguments":{}}</tool_call>`
		deltas, err = stream.Accept(thought)
		if err != nil {
			t.Fatal(err)
		}
		for _, delta := range deltas {
			if delta.Started || delta.Name != "" || delta.Arguments != "" {
				t.Fatalf("thought emitted tool: %+v", delta)
			}
		}
		_, err = stream.Accept(`</think><tool_call>{"name":"weather","arguments":{"city":"Paris"}}</tool_call>`)
		if err != nil {
			t.Fatal(err)
		}
		message, err = stream.Finish()
		if err != nil || message.ReasoningContent != thought || len(message.ToolCalls) != 1 || message.ToolCalls[0].Function.Name != "weather" {
			t.Fatalf("named template: %+v %v", message, err)
		}
		if _, err := runner.NewChatOutputStreamForPrompt("", []tokenizer.TokenID{tokenizer.TokenID(len(runner.vocab.Tokens))}, nil); err == nil {
			t.Fatal("invalid prepared token accepted")
		}
	})
	t.Run("progress_before_finish", func(t *testing.T) {
		plain, err := newReasoningChannelStream(template, "assistant")
		if err != nil {
			t.Fatal(err)
		}
		if r, c := plain.accept("answer", false); r != "" || c != "answer" {
			t.Fatalf("ordinary text withheld: %q / %q", r, c)
		}
		thinking, err := newReasoningChannelStream(template, "assistant<|channel>thought\n")
		if err != nil {
			t.Fatal(err)
		}
		if r, c := thinking.accept("確認", false); r != "確認" || c != "" {
			t.Fatalf("reasoning withheld: %q / %q", r, c)
		}
	})
	for _, test := range []struct{ name, template, prompt, output, reasoning, content string }{
		{"explicit", template, "assistant", "<|channel>thought\n確認 🙂\n<channel|>réponse π", "確認 🙂", "réponse π"},
		{"unicode_leading_space", template, "assistant", "\u2003<|channel>thought\nreason<channel|>answer", "reason", "answer"},
		{"think", "<think></think>", "assistant", "<think>reason</think>answer", "reason", "answer"},
		{"inkling", "<|content_thinking|><|content_text|>", "assistant", "<|content_thinking|>reason<|content_text|>answer", "reason", "answer"},
		{"prompt_open", template, "assistant<|channel>thought\n", "example <tool_call>{}</tool_call>\n<channel|>answer", "example <tool_call>{}</tool_call>", "answer"},
		{"plain", template, "assistant", "  answer 🙂\n", "", "  answer 🙂\n"},
		{"prompt_closed", template, "assistant<|channel>thought\n<channel|>\n", "\nanswer", "", "\nanswer"},
		{"literal", template, "assistant", "guide: <|channel>thought\nquoted<channel|>text", "", "guide: <|channel>thought\nquoted<channel|>text"},
		{"undeclared", "", "assistant", "<think>literal</think>", "", "<think>literal</think>"},
		{"historical_marker", template, "user <|channel>thought\ntext\nassistant", "answer", "", "answer"},
		{"budget_in_thought", template, "assistant<|channel>thought\n", "unfinished thought", "unfinished thought", ""},
		{"partial_open_literal", template, "assistant", "<|chan", "", "<|chan"},
	} {
		t.Run(test.name, func(t *testing.T) {
			check := func(pieces []string) {
				stream, err := newReasoningChannelStream(test.template, test.prompt)
				if err != nil {
					t.Fatal(err)
				}
				var reasoning, content strings.Builder
				for _, piece := range pieces {
					r, c := stream.accept(piece, false)
					if !utf8.ValidString(r) || !utf8.ValidString(c) {
						t.Fatalf("split UTF-8: %q / %q", r, c)
					}
					reasoning.WriteString(r)
					content.WriteString(c)
				}
				r, c := stream.accept("", true)
				reasoning.WriteString(r)
				content.WriteString(c)
				if reasoning.String() != test.reasoning || content.String() != test.content {
					t.Fatalf("pieces=%q: got %q / %q; want %q / %q", pieces, reasoning.String(), content.String(), test.reasoning, test.content)
				}
			}
			var bytes []string
			for index := range len(test.output) {
				bytes = append(bytes, test.output[index:index+1])
			}
			check(bytes)
			for split := range len(test.output) + 1 {
				check([]string{test.output[:split], test.output[split:]})
			}
		})
	}
}
