package inference

import (
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// TestAToolCallValueMayHoldALessThanSign holds a parameter-form tool call's value to any text but its closing tag
// (loop-proof/tool-value-markup). In loop-proof/local-demonstrated's run 3 the local worker's edits of AGENTS.md each
// ended where the line goes on with <item>/<step>: its values were drawn under a rule admitting no less-than sign, so
// it could write no Go comparison, channel receive, type list or markup. A value holding "i < n", "<-ch", "a << 2",
// "<<" and "<item>/<step>" is taken; a value holding the closing tag itself is not, since the parser ends a value at
// it.
func TestAToolCallValueMayHoldALessThanSign(t *testing.T) {
	t.Parallel()
	runner := jinjaChatTestRunner(t, `<tool_call><function=example_function_name></function></tool_call>`)
	tool := ChatTool{Type: "function", Function: ChatToolDefinition{Name: "edit", Description: "Edit a file", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"path": map[string]any{"type": "string"}, "new": map[string]any{"type": "string"}},
		"required":   []any{"path", "new"},
	}}}
	source, root, _, err := runner.ChatToolGrammar([]ChatTool{tool}, true, false, false)
	testcheck.NoError(t, err)
	const bytesOfAToken = 256
	pieces := make([][]byte, bytesOfAToken+1)
	for value := range bytesOfAToken {
		pieces[value] = []byte{byte(value)}
	}
	end := bytesOfAToken
	vocabulary := testcheck.Must(sampling.NewGBNFVocabulary(pieces, []int{end}, nil))(t)
	grammar := testcheck.Must(vocabulary.Compile(source, root, sampling.GBNFLazyOptions{}))(t)
	takes := func(text string) bool {
		sampler := testcheck.Must(sampling.New(sampling.Config{GBNF: grammar}))(t)
		for _, want := range append([]int(nil), append(bytesOf(text), end)...) {
			logits := make([]float32, len(pieces))
			logits[want] = 1
			if token, err := sampler.Sample(logits); err != nil || token != want {
				return false
			}
		}
		return true
	}
	call := func(value string) string {
		return "<tool_call><function=edit><parameter=path>a.go</parameter><parameter=new>" + value + "</parameter></function></tool_call>"
	}
	for value, want := range map[string]bool{
		"for i := 0; i < n; i++ {": true,
		"v := <-ch":                true,
		"a << 2 <= b":              true,
		"- Commit it: `go run ./cmd/loop -land <item>/<step> -message-file -`": true,
		"x </parameter> y": false,
	} {
		if got := takes(call(value)); got != want {
			t.Errorf("the grammar takes a value %q: %t, want %t", value, got, want)
		}
	}
}

// bytesOf is text as tokens of a byte each.
func bytesOf(text string) []int {
	tokens := make([]int, len(text))
	for index, value := range []byte(text) {
		tokens[index] = int(value)
	}
	return tokens
}
