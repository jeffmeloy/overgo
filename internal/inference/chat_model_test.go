//go:build modeltest

package inference

import (
	"strings"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modeltest"
)

func TestNativeQwenChatTemplateMatchesPinnedOracle(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunner(modeltest.Directory(t, qwen3Fixture), 0)
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
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunner(modeltest.Directory(t, qwen3Fixture), 0)
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

func TestNativeGemmaChatTemplateMatchesPinnedOracle(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunner(modeltest.Directory(t, gemma3Fixture), 0)
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
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunner(modeltest.Directory(t, qwen35Fixture), 0)
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
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunner(modeltest.Directory(t, qwen35Fixture), 0)
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

func TestNativeBonsaiChatTemplateMatchesPinnedOracle(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunner(modeltest.Directory(t, bonsaiFixture), 0)
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
	requireIntegration(t)
	cudatest.Require(t)
	runner, err := openFixtureRunner(modeltest.Directory(t, bonsaiFixture), 0)
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
