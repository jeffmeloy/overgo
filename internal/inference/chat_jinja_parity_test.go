package inference

import (
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// chatParityFixtures returns the message/option battery. Tool-call arguments
// use a single key so gonja's nondeterministic map iteration (|items,
// dict|tojson) stayed reproducible when the golden references were snapshotted.
func chatParityFixtures() []struct {
	name     string
	messages []ChatMessage
	options  ChatFormatOptions
} {
	tool := ChatTool{
		Type: "function",
		Function: ChatToolDefinition{
			Name:        "weather",
			Description: "Get weather",
			Parameters: map[string]any{
				"properties": map[string]any{
					"city": map[string]any{"type": "string", "description": "City name"},
				},
				"required": []any{"city"},
				"type":     "object",
			},
		},
	}
	assistantCall := ChatMessage{
		Role: "assistant",
		ToolCalls: []ChatToolCall{{
			ID: "call_1", Type: "function",
			Function: ChatToolFunction{Name: "weather", Arguments: `{"city":"Paris"}`},
		}},
	}
	return []struct {
		name     string
		messages []ChatMessage
		options  ChatFormatOptions
	}{
		{"sys_user_gen_think", []ChatMessage{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Hello"}}, ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true}},
		{"sys_user_gen_nothink", []ChatMessage{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Hello"}}, ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: false}},
		{"user_only", []ChatMessage{{Role: "user", Content: "Hello"}}, ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true}},
		{"user_nogen", []ChatMessage{{Role: "user", Content: "Hello"}}, ChatFormatOptions{AddGenerationPrompt: false, EnableThinking: true}},
		{"multiturn", []ChatMessage{
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Hi"},
			{Role: "assistant", Content: "Hello there"},
			{Role: "user", Content: "Weather?"},
		}, ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true}},
		{"assistant_reasoning", []ChatMessage{
			{Role: "user", Content: "Q"},
			{Role: "assistant", Content: "answer", ReasoningContent: "because"},
			{Role: "user", Content: "again"},
		}, ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true}},
		{"tools_sys_user", []ChatMessage{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Weather in Paris?"}}, ChatFormatOptions{Tools: []ChatTool{tool}, AddGenerationPrompt: true, EnableThinking: true}},
		{"tools_sys_user_nothink", []ChatMessage{{Role: "system", Content: "Be concise."}, {Role: "user", Content: "Weather in Paris?"}}, ChatFormatOptions{Tools: []ChatTool{tool}, AddGenerationPrompt: true, EnableThinking: false}},
		{"tools_no_system", []ChatMessage{{Role: "user", Content: "Weather in Paris?"}}, ChatFormatOptions{Tools: []ChatTool{tool}, AddGenerationPrompt: true, EnableThinking: true}},
		{"tool_history", []ChatMessage{
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Weather in Paris?"},
			assistantCall,
			{Role: "tool", Content: "sunny", ToolCallID: "call_1"},
		}, ChatFormatOptions{Tools: []ChatTool{tool}, AddGenerationPrompt: true, EnableThinking: true}},
		{"assistant_think_tags", []ChatMessage{
			{Role: "user", Content: "Q"},
			{Role: "assistant", Content: "<think>\nreason here\n</think>\n\nfinal answer"},
			{Role: "user", Content: "more"},
		}, ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true}},
		{"special_chars", []ChatMessage{
			{Role: "system", Content: "quote \" and 'apos' & <lt> \\slash\\ tab\tend"},
			{Role: "user", Content: "unicode café 日本語 \n newline\n line2"},
		}, ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: true}},
		{"two_tools", []ChatMessage{
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Weather?"},
		}, ChatFormatOptions{Tools: []ChatTool{tool, {
			Type: "function",
			Function: ChatToolDefinition{
				Name: "clock", Description: "Get time",
				Parameters: map[string]any{"type": "object", "properties": map[string]any{"tz": map[string]any{"type": "string"}}},
			},
		}}, AddGenerationPrompt: true, EnableThinking: true}},
		{"empty_content_toolcall", []ChatMessage{
			{Role: "user", Content: "Weather in Paris?"},
			{Role: "assistant", Content: "", ToolCalls: []ChatToolCall{{ID: "call_1", Type: "function", Function: ChatToolFunction{Name: "weather", Arguments: `{"city":"Paris"}`}}}},
			{Role: "tool", Content: "rainy & cold", ToolCallID: "call_1"},
		}, ChatFormatOptions{Tools: []ChatTool{tool}, AddGenerationPrompt: true, EnableThinking: true}},
		{"assistant_prose_toolcall", []ChatMessage{
			{Role: "user", Content: "Weather in Paris?"},
			{Role: "assistant", Content: "Let me check.", ToolCalls: []ChatToolCall{{ID: "call_1", Type: "function", Function: ChatToolFunction{Name: "weather", Arguments: `{"city":"Paris"}`}}}},
			{Role: "tool", Content: "sunny", ToolCallID: "call_1"},
		}, ChatFormatOptions{Tools: []ChatTool{tool}, AddGenerationPrompt: true, EnableThinking: true}},
	}
}

// TestChatTemplateJinjaByteParity renders every served chat template through
// the stdlib interpreter (internal/jinja) and asserts byte-identity with the
// gonja reference snapshots in testdata/chat_templates/golden (a .txt golden
// requires the exact bytes; a .err golden requires the interpreter to error
// identically, matching an input gonja itself rejects). Regenerate the goldens
// with OVERGO_SNAPSHOT=1 (see chat_jinja_snapshot generator) whenever the
// template set changes. The committed test has no gonja dependency.
func TestChatTemplateJinjaByteParity(t *testing.T) {
	dir := filepath.Join("testdata", "chat_templates")
	goldenDir := filepath.Join(dir, "golden")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var files []string
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".jinja" {
			files = append(files, e.Name())
		}
	}
	sort.Strings(files)

	runner := jinjaChatTestRunner(t, "")
	fixtures := chatParityFixtures()

	type verdict struct{ total, match, bothErr, mismatch int }
	results := map[string]*verdict{}

	for _, file := range files {
		src, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil {
			t.Fatal(err)
		}
		source := string(src)
		base := file[:len(file)-len(".jinja")]
		v := &verdict{}
		results[file] = v
		for _, fx := range fixtures {
			v.total++
			jinjaOut, jinjaErr := runner.formatJinjaChatNative(source, fx.messages, fx.options)
			stem := filepath.Join(goldenDir, base+"__"+fx.name)
			if _, err := os.Stat(stem + ".err"); err == nil {
				if jinjaErr == nil {
					v.mismatch++
					t.Errorf("%s/%s: gonja reference errored but interpreter succeeded: %q", file, fx.name, jinjaOut)
				} else {
					v.bothErr++
				}
				continue
			}
			want, err := os.ReadFile(stem + ".txt")
			if err != nil {
				t.Fatalf("%s/%s: missing golden: %v", file, fx.name, err)
			}
			if jinjaErr != nil {
				v.mismatch++
				t.Errorf("%s/%s: interpreter errored but gonja reference succeeded: %v", file, fx.name, jinjaErr)
				continue
			}
			if string(want) != jinjaOut {
				v.mismatch++
				t.Errorf("%s/%s: BYTE MISMATCH\n want=%q\n  got=%q\n%s", file, fx.name, string(want), jinjaOut, firstDiff(string(want), jinjaOut))
				continue
			}
			v.match++
		}
	}

	names := make([]string, 0, len(results))
	for n := range results {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		v := results[n]
		t.Logf("PARITY %-16s total=%d match=%d bothErr=%d mismatch=%d", n, v.total, v.match, v.bothErr, v.mismatch)
	}
}

func firstDiff(a, b string) string {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	lo := i - 30
	if lo < 0 {
		lo = 0
	}
	sa := a[lo:min(i+40, len(a))]
	sb := b[lo:min(i+40, len(b))]
	return "  first diff at byte " + itoa(i) + "\n   ...want: " + quoteDiff(sa) + "\n   ...got:  " + quoteDiff(sb)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	if neg {
		b = append([]byte{'-'}, b...)
	}
	return string(b)
}

func quoteDiff(s string) string {
	out := make([]byte, 0, len(s)+2)
	out = append(out, '"')
	for _, r := range []byte(s) {
		switch r {
		case '\n':
			out = append(out, '\\', 'n')
		case '\t':
			out = append(out, '\\', 't')
		default:
			out = append(out, r)
		}
	}
	out = append(out, '"')
	return string(out)
}
