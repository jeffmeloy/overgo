package inference

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

func TestChatMessageDecodesTextContentParts(t *testing.T) {
	var message ChatMessage
	err := json.Unmarshal(
		[]byte(`{"role":"user","content":[{"type":"text","text":"Hello"},{"type":"input_text","text":" world"}]}`),
		&message,
	)
	if err != nil {
		t.Fatal(err)
	}
	if message.Role != "user" || message.Content != "Hello world" {
		t.Fatalf("message = %+v", message)
	}
	var multimodal ChatMessage
	err = json.Unmarshal(
		[]byte(`{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA==","detail":"auto"}},{"type":"input_text","text":"after"},{"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}}]}`),
		&multimodal,
	)
	if err != nil {
		t.Fatal(err)
	}
	if multimodal.Content != "beforeafter" || len(multimodal.Media) != 2 {
		t.Fatalf("multimodal message = %+v", multimodal)
	}
	if multimodal.Media[0].Type != "image" ||
		multimodal.Media[0].TextOffset != len("before") ||
		multimodal.Media[1].Type != "audio" ||
		multimodal.Media[1].Format != "wav" ||
		multimodal.Media[1].TextOffset != len("beforeafter") {
		t.Fatalf("multimodal media = %+v", multimodal.Media)
	}
	for _, data := range []string{
		`{"role":"user","content":[{"type":"image_url","image_url":{"url":""}}]}`,
		`{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AA=="}}]}`,
		`{"role":"user","content":[{"type":"text","text":"x","extra":true}]}`,
		`{"role":"user","content":[{"type":"input_file","file_id":"x"}]}`,
		`{"role":"user"}`,
	} {
		if err := json.Unmarshal([]byte(data), &message); err == nil {
			t.Fatalf("message %s was accepted", data)
		}
	}
}

func TestFormatChatRejectsUnprojectedMedia(t *testing.T) {
	runner := &Runner{vocab: chatTestVocab(t)}
	_, err := runner.FormatChat([]ChatMessage{{
		Role:  "user",
		Media: []ChatMediaPart{{Type: "image", Data: "x"}},
	}})
	if err == nil || !strings.Contains(err.Error(), "requires multimodal projection") {
		t.Fatalf("format error = %v", err)
	}
}

func TestChatMessageDecodesToolCallsAndToolResult(t *testing.T) {
	var assistant ChatMessage
	err := json.Unmarshal(
		[]byte(`{"role":"assistant","content":null,"reasoning_content":"check","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}]}`),
		&assistant,
	)
	if err != nil {
		t.Fatal(err)
	}
	if assistant.Role != "assistant" ||
		assistant.Content != "" ||
		assistant.ReasoningContent != "check" ||
		len(assistant.ToolCalls) != 1 ||
		assistant.ToolCalls[0].Function.Name != "weather" ||
		assistant.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Fatalf("assistant message = %+v", assistant)
	}
	encoded, err := json.Marshal(assistant)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"content":null`) {
		t.Fatalf("encoded assistant tool call = %s", encoded)
	}
	var result ChatMessage
	if err := json.Unmarshal(
		[]byte(`{"role":"tool","tool_call_id":"call_1","name":"weather","content":"failed","is_error":true}`),
		&result,
	); err != nil {
		t.Fatal(err)
	}
	if result.ToolCallID != "call_1" ||
		result.Name != "weather" ||
		!result.ToolResultError {
		t.Fatalf("tool result = %+v", result)
	}
	for _, data := range []string{
		`{"role":"assistant","tool_calls":[]}`,
		`{"role":"assistant","tool_calls":[{"type":"custom","function":{"name":"x","arguments":"{}"}}]}`,
		`{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"","arguments":"{}"}}]}`,
	} {
		if err := json.Unmarshal([]byte(data), &assistant); err == nil {
			t.Fatalf("message %s was accepted", data)
		}
	}
}

func TestFormatChatML(t *testing.T) {
	runner := &Runner{vocab: chatTestVocab(t)}
	prompt, err := runner.FormatChat([]ChatMessage{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nBe concise.<|im_end|>\n" +
		"<|im_start|>user\nHello<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if prompt != want {
		t.Fatalf("prompt = %q, want %q", prompt, want)
	}
}

func TestFormatChatUsesGGUFJinjaTemplateWithoutFilesystemIncludes(t *testing.T) {
	template := `{{ bos_token }}{% for message in messages %}` +
		`[{{ message['role'] }}]{{ message['content'] }}{{ eos_token }}` +
		`{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}`
	runner := &Runner{
		file: &gguf.File{Metadata: []gguf.Metadata{{
			Key: "tokenizer.chat_template",
			Value: gguf.Value{
				Type: gguf.ValueTypeString,
				Data: template,
			},
		}}},
		vocab: &tokenizer.Vocab{
			Tokens: []tokenizer.Token{
				{Text: "<s>", Type: tokenizer.TokenControl},
				{Text: "</s>", Type: tokenizer.TokenControl},
			},
			BOS:    0,
			EOS:    1,
			EOT:    tokenizer.NullToken,
			EOM:    tokenizer.NullToken,
			FIMPad: tokenizer.NullToken,
			FIMRep: tokenizer.NullToken,
			FIMSep: tokenizer.NullToken,
			AddBOS: true,
		},
	}
	got, err := runner.FormatChat([]ChatMessage{
		{Role: "system", Content: "rules"},
		{Role: "user", Content: "hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "[system]rules</s>[user]hello</s>[assistant]"
	if got != want {
		t.Fatalf("Jinja prompt = %q, want %q", got, want)
	}

	runner.file.Metadata[0].Value.Data = `{% include "C:/Windows/win.ini" %}`
	if _, err := runner.FormatChat([]ChatMessage{{
		Role:    "user",
		Content: "hello",
	}}); err == nil {
		t.Fatal("GGUF chat template read a filesystem include")
	}
}

func TestFormatJinjaChatSupportsPinnedRuntimeExtensions(t *testing.T) {
	runner := jinjaChatTestRunner(t,
		`{{ strftime_now('%Y-%m-%d')|length }}:`+
			`{% set ns = namespace(hit=false) %}`+
			`{% for index in range(3) %}`+
			`{% if index == 2 %}{% set ns.hit = true %}{% endif %}`+
			`{% endfor %}{{ ns.hit }}`,
	)
	got, err := runner.FormatChat([]ChatMessage{{Role: "user", Content: "hi"}})
	if err != nil {
		t.Fatal(err)
	}
	if got != "10:True" {
		t.Fatalf("extension prompt = %q, want %q", got, "10:True")
	}

	runner = jinjaChatTestRunner(t, `{{ raise_exception('roles must alternate') }}`)
	if _, err := runner.FormatChat([]ChatMessage{{Role: "user", Content: "hi"}}); err == nil ||
		!strings.Contains(err.Error(), "roles must alternate") {
		t.Fatalf("raise_exception error = %v", err)
	}
}

func TestFormatChatTemplateTimeMatchesPinnedDirectives(t *testing.T) {
	value := time.Date(2026, time.August, 2, 17, 4, 5, 0, time.FixedZone("EDT", -4*60*60))
	got, err := formatChatTemplateTime(
		value,
		"%a|%A|%b|%B|%c|%d|%e|%H|%I|%j|%m|%M|%p|%S|%u|%w|%x|%X|%y|%Y|%z|%Z|%%",
	)
	if err != nil {
		t.Fatal(err)
	}
	want := "Sun|Sunday|Aug|August|Sun Aug 02 17:04:05 2026|02| 2|17|05|214|08|04|PM|05|7|0|08/02/26|17:04:05|26|2026|-0400|EDT|%"
	if got != want {
		t.Fatalf("strftime_now = %q, want %q", got, want)
	}
	if _, err := formatChatTemplateTime(value, "%Q"); err == nil {
		t.Fatal("strftime_now accepted unsupported directive")
	}
}

func TestFormatJinjaChatSuppliesToolsAndToolCalls(t *testing.T) {
	template := `{%- for tool in tools %}{{ tool.function.name }}={{ tool.function.parameters | tojson }};{%- endfor %}` +
		`{%- for message in messages %}{{ message.role }}:{{ message.content }}` +
		`{%- for call in message.tool_calls %}[{{ call.id }}:{{ call.function.name }}({{ call.function.arguments }})]{%- endfor %};{%- endfor %}` +
		`thinking={{ enable_thinking }};generate={{ add_generation_prompt }}`
	runner := jinjaChatTestRunner(t, template)
	got, err := runner.FormatChatWithOptions(
		[]ChatMessage{
			{Role: "user", Content: "weather?"},
			{
				Role: "assistant",
				ToolCalls: []ChatToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: ChatToolFunction{
						Name:      "weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			},
			{Role: "tool", Content: "sunny", ToolCallID: "call_1"},
		},
		ChatFormatOptions{
			Tools: []ChatTool{{
				Type: "function",
				Function: ChatToolDefinition{
					Name:        "weather",
					Description: "Get weather",
					Parameters: map[string]any{
						"type": "object",
					},
				},
			}},
			AddGenerationPrompt: true,
			EnableThinking:      false,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := `weather={"type": "object"};user:weather?;assistant:[call_1:weather({"city":"Paris"})];tool:sunny;thinking=False;generate=True`
	if got != want {
		t.Fatalf("tool-aware Jinja prompt = %q, want %q", got, want)
	}
}

func TestFormatChatSelectsNamedToolUseTemplate(t *testing.T) {
	runner := jinjaChatTestRunner(t, `default`)
	runner.file.Metadata = append(runner.file.Metadata, gguf.Metadata{
		Key: "tokenizer.chat_template.tool_use",
		Value: gguf.Value{
			Type: gguf.ValueTypeString,
			Data: `tool={{ tools[0].function.name }}`,
		},
	})
	plain, err := runner.FormatChat([]ChatMessage{{Role: "user", Content: "hi"}})
	if err != nil {
		t.Fatal(err)
	}
	if plain != "default" {
		t.Fatalf("plain prompt = %q", plain)
	}
	withTool, err := runner.FormatChatWithOptions(
		[]ChatMessage{{Role: "user", Content: "hi"}},
		ChatFormatOptions{
			Tools:               []ChatTool{toolWeatherDefinition()},
			AddGenerationPrompt: true,
			EnableThinking:      true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if withTool != "tool=weather" {
		t.Fatalf("tool-use prompt = %q", withTool)
	}
}

func TestNativeQwenChatTemplateMatchesPinnedOracle(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_QWEN3_MODEL")
	if path == "" {
		t.Skip("LLAMACPP2GO_QWEN3_MODEL is not set")
	}
	runner, err := Open(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	got, err := runner.FormatChat([]ChatMessage{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nBe concise.<|im_end|>\n" +
		"<|im_start|>user\nHello<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("native Qwen Jinja prompt = %q, want %q", got, want)
	}
}

func TestNativeQwenToolChatTemplateMatchesPinnedOracle(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_QWEN3_MODEL")
	if path == "" {
		t.Skip("LLAMACPP2GO_QWEN3_MODEL is not set")
	}
	runner, err := Open(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	got, err := runner.FormatChatWithOptions(
		[]ChatMessage{
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Weather in Paris?"},
		},
		ChatFormatOptions{
			Tools: []ChatTool{{
				Type: "function",
				Function: ChatToolDefinition{
					Name:        "weather",
					Description: "Get weather",
					Parameters: map[string]any{
						"properties": map[string]any{
							"city": map[string]any{"type": "string"},
						},
						"required": []any{"city"},
						"type":     "object",
					},
				},
			}},
			AddGenerationPrompt: true,
			EnableThinking:      true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nBe concise.\n\n# Tools\n\n" +
		"You may call one or more functions to assist with the user query.\n\n" +
		"You are provided with function signatures within <tools></tools> XML tags:\n" +
		"<tools>\n" +
		`{"type": "function", "function": {"name": "weather", "description": "Get weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"], "type": "object"}}}` +
		"\n</tools>\n\n" +
		"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n" +
		"<tool_call>\n" +
		"{\"name\": <function-name>, \"arguments\": <args-json-object>}\n" +
		"</tool_call><|im_end|>\n" +
		"<|im_start|>user\nWeather in Paris?<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("native Qwen tool Jinja prompt = %q, want %q", got, want)
	}
	got, err = runner.FormatChatWithOptions(
		[]ChatMessage{
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Weather in Paris?"},
			{
				Role: "assistant",
				ToolCalls: []ChatToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: ChatToolFunction{
						Name:      "weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			},
			{Role: "tool", Content: "sunny", ToolCallID: "call_1"},
		},
		ChatFormatOptions{
			Tools:               []ChatTool{toolWeatherDefinition()},
			AddGenerationPrompt: true,
			EnableThinking:      true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	want = strings.TrimSuffix(want, "<|im_start|>assistant\n") +
		"<|im_start|>assistant\n" +
		"<tool_call>\n" +
		`{"name": "weather", "arguments": {"city": "Paris"}}` + "\n" +
		"</tool_call><|im_end|>\n" +
		"<|im_start|>user\n" +
		"<tool_response>\nsunny\n</tool_response><|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("native Qwen tool-history Jinja prompt = %q, want %q", got, want)
	}
}

func toolWeatherDefinition() ChatTool {
	return ChatTool{
		Type: "function",
		Function: ChatToolDefinition{
			Name:        "weather",
			Description: "Get weather",
			Parameters: map[string]any{
				"properties": map[string]any{
					"city": map[string]any{"type": "string"},
				},
				"required": []any{"city"},
				"type":     "object",
			},
		},
	}
}

func TestNativeGemmaChatTemplateMatchesPinnedOracle(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_GEMMA3_MODEL")
	if path == "" {
		t.Skip("LLAMACPP2GO_GEMMA3_MODEL is not set")
	}
	runner, err := Open(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	got, err := runner.FormatChat([]ChatMessage{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "<start_of_turn>user\nBe concise.\n\nHello<end_of_turn>\n" +
		"<start_of_turn>model\n"
	if got != want {
		t.Fatalf("native Gemma Jinja prompt = %q, want %q", got, want)
	}
}

func TestNativeQwen35ChatTemplateMatchesPinnedOracle(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_QWEN35_MODEL")
	if path == "" {
		t.Skip("LLAMACPP2GO_QWEN35_MODEL is not set")
	}
	runner, err := Open(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	got, err := runner.FormatChat([]ChatMessage{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nBe concise.<|im_end|>\n" +
		"<|im_start|>user\nHello<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n"
	if got != want {
		t.Fatalf("native Qwen3.5 Jinja prompt = %q, want %q", got, want)
	}
}

func TestNativeQwen35ToolChatTemplateMatchesPinnedOracle(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_QWEN35_MODEL")
	if path == "" {
		t.Skip("LLAMACPP2GO_QWEN35_MODEL is not set")
	}
	runner, err := Open(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	tool := toolWeatherDefinition()
	tool.Function.Parameters["properties"].(map[string]any)["city"] = map[string]any{
		"description": "City name",
		"type":        "string",
	}
	got, err := runner.FormatChatWithOptions(
		[]ChatMessage{
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Weather in Paris?"},
		},
		ChatFormatOptions{
			Tools:               []ChatTool{tool},
			AddGenerationPrompt: true,
			EnableThinking:      true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := hermesToolOracle(false)
	if got != want {
		t.Fatalf("native Qwen3.5 tool Jinja prompt = %q, want %q", got, want)
	}
	got, err = runner.FormatChatWithOptions(
		[]ChatMessage{
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Weather in Paris?"},
			{
				Role: "assistant",
				ToolCalls: []ChatToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: ChatToolFunction{
						Name:      "weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			},
			{Role: "tool", Content: "sunny", ToolCallID: "call_1"},
		},
		ChatFormatOptions{
			Tools:               []ChatTool{tool},
			AddGenerationPrompt: true,
			EnableThinking:      true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	want = hermesToolOracle(true)
	if got != want {
		t.Fatalf(
			"native Qwen3.5 tool-history Jinja prompt = %q, want %q",
			got,
			want,
		)
	}
}

func hermesToolOracle(history bool) string {
	base := "<|im_start|>system\n" +
		"# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
		`{"type": "function", "function": {"name": "weather", "description": "Get weather", "parameters": {"properties": {"city": {"description": "City name", "type": "string"}}, "required": ["city"], "type": "object"}}}` +
		"\n</tools>\n\n" +
		"If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
		"<tool_call>\n<function=example_function_name>\n" +
		"<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
		"<parameter=example_parameter_2>\n" +
		"This is the value for the second parameter\nthat can span\nmultiple lines\n" +
		"</parameter>\n</function>\n</tool_call>\n\n" +
		"<IMPORTANT>\nReminder:\n" +
		"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
		"- Required parameters MUST be specified\n" +
		"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
		"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
		"</IMPORTANT>\n\nBe concise.<|im_end|>\n" +
		"<|im_start|>user\nWeather in Paris?<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n"
	if !history {
		return base
	}
	return strings.TrimSuffix(base, "<|im_start|>assistant\n<think>\n") +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n" +
		"<tool_call>\n<function=weather>\n" +
		"<parameter=city>\nParis\n</parameter>\n" +
		"</function>\n</tool_call><|im_end|>\n" +
		"<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n"
}

func TestNativeBonsaiChatTemplateMatchesPinnedOracle(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_BONSAI_MODEL")
	if path == "" {
		t.Skip("LLAMACPP2GO_BONSAI_MODEL is not set")
	}
	runner, err := Open(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	got, err := runner.FormatChat([]ChatMessage{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hello"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nBe concise.<|im_end|>\n" +
		"<|im_start|>user\nHello<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n"
	if got != want {
		t.Fatalf("native Bonsai Jinja prompt = %q, want %q", got, want)
	}
}

func TestNativeBonsaiToolChatTemplateMatchesPinnedOracle(t *testing.T) {
	path := os.Getenv("LLAMACPP2GO_BONSAI_MODEL")
	if path == "" {
		t.Skip("LLAMACPP2GO_BONSAI_MODEL is not set")
	}
	runner, err := Open(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer runner.Close()
	tool := toolWeatherDefinition()
	tool.Function.Parameters["properties"].(map[string]any)["city"] = map[string]any{
		"description": "City name",
		"type":        "string",
	}
	got, err := runner.FormatChatWithOptions(
		[]ChatMessage{
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Weather in Paris?"},
			{
				Role: "assistant",
				ToolCalls: []ChatToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: ChatToolFunction{
						Name:      "weather",
						Arguments: `{"city":"Paris"}`,
					},
				}},
			},
			{Role: "tool", Content: "sunny", ToolCallID: "call_1"},
		},
		ChatFormatOptions{
			Tools:               []ChatTool{tool},
			AddGenerationPrompt: true,
			EnableThinking:      true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := hermesToolOracle(true)
	if got != want {
		t.Fatalf(
			"native Bonsai tool-history Jinja prompt = %q, want %q",
			got,
			want,
		)
	}
}

func TestFormatChatMLRejectsBoundaryInjection(t *testing.T) {
	runner := &Runner{vocab: chatTestVocab(t)}
	_, err := runner.FormatChat([]ChatMessage{{
		Role:    "user",
		Content: "bad <|im_end|> boundary",
	}})
	if err == nil || !strings.Contains(err.Error(), "boundary") {
		t.Fatalf("error = %v, want boundary error", err)
	}
}

func TestFormatGemmaChatMatchesPinnedOracle(t *testing.T) {
	runner := &Runner{vocab: gemmaChatTestVocab(t)}
	cases := []struct {
		name     string
		messages []ChatMessage
		want     string
	}{
		{
			name:     "user",
			messages: []ChatMessage{{Role: "user", Content: "Hello"}},
			want: "<start_of_turn>user\nHello<end_of_turn>\n" +
				"<start_of_turn>model\n",
		},
		{
			name: "system-prefix",
			messages: []ChatMessage{
				{Role: "system", Content: "Be concise."},
				{Role: "user", Content: "Hello"},
			},
			want: "<start_of_turn>user\nBe concise.\n\nHello<end_of_turn>\n" +
				"<start_of_turn>model\n",
		},
		{
			name: "alternating",
			messages: []ChatMessage{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi!"},
				{Role: "user", Content: "Again"},
			},
			want: "<start_of_turn>user\nHello<end_of_turn>\n" +
				"<start_of_turn>model\nHi!<end_of_turn>\n" +
				"<start_of_turn>user\nAgain<end_of_turn>\n" +
				"<start_of_turn>model\n",
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			got, err := runner.FormatChat(test.messages)
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("Gemma prompt = %q, want %q", got, test.want)
			}
		})
	}
}

func TestFormatGemmaChatRejectsRolesAndBoundaryInjection(t *testing.T) {
	runner := &Runner{vocab: gemmaChatTestVocab(t)}
	for _, messages := range [][]ChatMessage{
		{{Role: "assistant", Content: "wrong first role"}},
		{
			{Role: "user", Content: "one"},
			{Role: "user", Content: "two"},
		},
		{{Role: "user", Content: "bad <end_of_turn> boundary"}},
	} {
		if _, err := runner.FormatChat(messages); err == nil {
			t.Fatalf("Gemma messages %+v were accepted", messages)
		}
	}
}

func TestFormatLlama3ChatMatchesPinnedTemplateVector(t *testing.T) {
	runner := &Runner{vocab: llama3ChatTestVocab(t)}
	prompt, err := runner.FormatChat([]ChatMessage{
		{Role: "system", Content: "You are a helpful assistant"},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
		{Role: "user", Content: "Who are you"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := "<|start_header_id|>system<|end_header_id|>\n\n" +
		"You are a helpful assistant<|eot_id|>" +
		"<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\nHi there<|eot_id|>" +
		"<|start_header_id|>user<|end_header_id|>\n\nWho are you<|eot_id|>" +
		"<|start_header_id|>assistant<|end_header_id|>\n\n"
	if prompt != want {
		t.Fatalf("Llama 3 prompt = %q, want %q", prompt, want)
	}
}

func TestFormatLlama3ChatRejectsRolesAndBoundaryInjection(t *testing.T) {
	runner := &Runner{vocab: llama3ChatTestVocab(t)}
	for _, messages := range [][]ChatMessage{
		{{Role: "assistant", Content: "wrong first role"}},
		{
			{Role: "user", Content: "one"},
			{Role: "user", Content: "two"},
		},
		{{Role: "user", Content: "bad <|eot_id|> boundary"}},
	} {
		if _, err := runner.FormatChat(messages); err == nil {
			t.Fatalf("Llama 3 messages %+v were accepted", messages)
		}
	}
}

func chatTestVocab(t *testing.T) *tokenizer.Vocab {
	t.Helper()
	file := &gguf.File{Metadata: []gguf.Metadata{
		{
			Key:   "tokenizer.ggml.model",
			Value: gguf.Value{Type: gguf.ValueTypeString, Data: "gpt2"},
		},
		{
			Key:   "tokenizer.ggml.pre",
			Value: gguf.Value{Type: gguf.ValueTypeString, Data: "qwen2"},
		},
		{
			Key: "tokenizer.ggml.tokens",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeString,
				Data:      []string{chatMLStart, chatMLEnd, "x"},
			},
		},
		{
			Key: "tokenizer.ggml.token_type",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeInt32,
				Data:      []int32{int32(tokenizer.TokenControl), int32(tokenizer.TokenControl), int32(tokenizer.TokenNormal)},
			},
		},
		{
			Key: "tokenizer.ggml.merges",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeString,
				Data:      []string{},
			},
		},
	}}
	vocab, err := tokenizer.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	return vocab
}

func jinjaChatTestRunner(t *testing.T, template string) *Runner {
	t.Helper()
	return &Runner{
		file: &gguf.File{Metadata: []gguf.Metadata{{
			Key: "tokenizer.chat_template",
			Value: gguf.Value{
				Type: gguf.ValueTypeString,
				Data: template,
			},
		}}},
		vocab: &tokenizer.Vocab{
			Tokens: []tokenizer.Token{
				{Text: "<s>", Type: tokenizer.TokenControl},
				{Text: "</s>", Type: tokenizer.TokenControl},
			},
			BOS:    0,
			EOS:    1,
			EOT:    tokenizer.NullToken,
			EOM:    tokenizer.NullToken,
			FIMPad: tokenizer.NullToken,
			FIMRep: tokenizer.NullToken,
			FIMSep: tokenizer.NullToken,
			AddBOS: true,
		},
	}
}

func gemmaChatTestVocab(t *testing.T) *tokenizer.Vocab {
	t.Helper()
	file := &gguf.File{Metadata: []gguf.Metadata{
		{
			Key:   "tokenizer.ggml.model",
			Value: gguf.Value{Type: gguf.ValueTypeString, Data: "gpt2"},
		},
		{
			Key:   "tokenizer.ggml.pre",
			Value: gguf.Value{Type: gguf.ValueTypeString, Data: "gpt-2"},
		},
		{
			Key: "tokenizer.ggml.tokens",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeString,
				Data:      []string{gemmaTurnStart, gemmaTurnEnd, "x"},
			},
		},
		{
			Key: "tokenizer.ggml.token_type",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeInt32,
				Data: []int32{
					int32(tokenizer.TokenControl),
					int32(tokenizer.TokenControl),
					int32(tokenizer.TokenNormal),
				},
			},
		},
		{
			Key: "tokenizer.ggml.merges",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeString,
				Data:      []string{},
			},
		},
	}}
	vocab, err := tokenizer.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	return vocab
}

func llama3ChatTestVocab(t *testing.T) *tokenizer.Vocab {
	t.Helper()
	file := &gguf.File{Metadata: []gguf.Metadata{
		{
			Key:   "tokenizer.ggml.model",
			Value: gguf.Value{Type: gguf.ValueTypeString, Data: "gpt2"},
		},
		{
			Key:   "tokenizer.ggml.pre",
			Value: gguf.Value{Type: gguf.ValueTypeString, Data: "llama-bpe"},
		},
		{
			Key:   "tokenizer.ggml.bos_token_id",
			Value: gguf.Value{Type: gguf.ValueTypeUint32, Data: uint32(0)},
		},
		{
			Key: "tokenizer.ggml.tokens",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeString,
				Data: []string{
					llama3BeginText,
					llama3HeaderStart,
					llama3HeaderEnd,
					llama3EndTurn,
					"x",
				},
			},
		},
		{
			Key: "tokenizer.ggml.token_type",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeInt32,
				Data: []int32{
					int32(tokenizer.TokenControl),
					int32(tokenizer.TokenControl),
					int32(tokenizer.TokenControl),
					int32(tokenizer.TokenControl),
					int32(tokenizer.TokenNormal),
				},
			},
		},
		{
			Key: "tokenizer.ggml.merges",
			Value: gguf.Value{
				Type:      gguf.ValueTypeArray,
				ArrayType: gguf.ValueTypeString,
				Data:      []string{},
			},
		},
	}}
	vocab, err := tokenizer.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	return vocab
}
