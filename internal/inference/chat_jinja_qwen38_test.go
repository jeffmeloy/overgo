package inference

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const qwen38ReasoningXHigh = "Reasoning effort is set to xhigh. Please think carefully through the task, " +
	"validate key assumptions, consider plausible alternatives, and prioritize correctness, " +
	"consistency, and clarity in the final answer."

// TestChatJinjaQwen38Template pins the Qwen3.8 chat template against the
// stdlib interpreter: the template's inline conditionals inside call
// arguments and parentheses, namespace state, reasoning-effort framing,
// thinking-mode control, and preserved historical thinking must render
// the exact conversation framing the model documentation prescribes.
func TestChatJinjaQwen38Template(t *testing.T) {
	source, err := os.ReadFile(filepath.Join("testdata", "qwen38_chat_template.jinja"))
	if err != nil {
		t.Fatal(err)
	}
	runner := jinjaChatTestRunner(t, string(source))

	thinking, err := runner.FormatChatWithOptions(
		[]ChatMessage{{Role: ChatRoleUser, Content: "Hello there"}},
		ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true},
	)
	if err != nil {
		t.Fatalf("thinking-mode render: %v", err)
	}
	wantThinking := "<|im_start|>system\n" + qwen38ReasoningXHigh + "<|im_end|>\n" +
		"<|im_start|>user\nHello there<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n"
	if thinking != wantThinking {
		t.Fatalf("thinking-mode framing:\n got=%q\nwant=%q", thinking, wantThinking)
	}

	direct, err := runner.FormatChatWithOptions(
		[]ChatMessage{{Role: ChatRoleUser, Content: "Hello there"}},
		ChatFormatOptions{AddGenerationPrompt: true},
	)
	if err != nil {
		t.Fatalf("non-thinking render: %v", err)
	}
	if !strings.HasSuffix(direct, "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Fatalf("non-thinking mode must close the think block immediately, got %q", direct)
	}
	if strings.Contains(direct, "Reasoning effort") {
		t.Fatalf("non-thinking mode must carry no reasoning instructions, got %q", direct)
	}

	history := []ChatMessage{
		{Role: ChatRoleUser, Content: "What is 2+2?"},
		{Role: ChatRoleAssistant, Content: "4", ReasoningContent: "two plus two is four"},
		{Role: ChatRoleUser, Content: "And doubled?"},
	}
	preserved, err := runner.FormatChatWithOptions(
		history,
		ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true},
	)
	if err != nil {
		t.Fatalf("preserved-thinking render: %v", err)
	}
	if !strings.Contains(preserved, "<|im_start|>assistant\n<think>\ntwo plus two is four\n</think>\n\n4<|im_end|>\n") {
		t.Fatalf("preserved thinking must retain the historical think block, got %q", preserved)
	}

	dropped, err := runner.FormatChatWithOptions(
		history,
		ChatFormatOptions{
			AddGenerationPrompt: true, EnableThinking: true,
			TemplateKwargs: map[string]any{"preserve_thinking": false, "reasoning_effort": "low"},
		},
	)
	if err != nil {
		t.Fatalf("kwargs render: %v", err)
	}
	if strings.Contains(dropped, "two plus two is four") {
		t.Fatalf("preserve_thinking=false must drop the historical think block, got %q", dropped)
	}
	if !strings.Contains(dropped, "Reasoning effort is set to low.") {
		t.Fatalf("reasoning_effort=low must select the low-effort instruction, got %q", dropped)
	}

	if _, err := runner.FormatChatWithOptions(
		history,
		ChatFormatOptions{
			AddGenerationPrompt: true,
			TemplateKwargs:      map[string]any{"messages": "override"},
		},
	); err == nil {
		t.Fatal("a template kwarg overriding a reserved context key must refuse")
	}
}
