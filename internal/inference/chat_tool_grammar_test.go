package inference

import (
	"strings"
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/testcheck"
)

// TestAToolCallGrammarHoldsEachParameterOnceAndTheRequiredOnes holds a parameter-form tool call to what the parser
// takes: Qwen3.8, the local worker's model, drew calls missing a required path and giving path twice, and the chat
// route answered each with a 500 (loop-proof/local-demonstrated, run 2). The grammar for a tool whose path is required
// and whose note is optional takes a call with path, with or without note, and takes no call missing path or giving it
// twice, read by a greedy draw under the compiled grammar, a byte a token.
func TestAToolCallGrammarHoldsEachParameterOnceAndTheRequiredOnes(t *testing.T) {
	t.Parallel()
	runner := jinjaChatTestRunner(t, `<tool_call><function=example_function_name></function></tool_call>`)
	tool := ChatTool{Type: "function", Function: ChatToolDefinition{Name: "edit", Description: "Edit a file", Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"path": map[string]any{"type": "string"}, "note": map[string]any{"type": "string"}},
		"required":   []any{"path"},
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
		tokens := make([]int, 0, len(text)+1)
		for _, value := range []byte(text) {
			tokens = append(tokens, int(value))
		}
		for _, want := range append(tokens, end) {
			logits := make([]float32, len(pieces))
			logits[want] = 1
			if token, err := sampler.Sample(logits); err != nil || token != want {
				return false
			}
		}
		return true
	}
	for text, want := range map[string]bool{
		"<tool_call><function=edit><parameter=path>a.go</parameter></function></tool_call>":                                 true,
		"<tool_call><function=edit><parameter=path>a.go</parameter><parameter=note>n</parameter></function></tool_call>":    true,
		"<tool_call><function=edit><parameter=note>n</parameter></function></tool_call>":                                    false,
		"<tool_call><function=edit><parameter=path>a.go</parameter><parameter=path>b.go</parameter></function></tool_call>": false,
	} {
		if got := takes(text); got != want {
			t.Errorf("the grammar takes %q: %t, want %t", text, got, want)
		}
	}
}

func TestJSONToolGrammarUsesSchemaAndLazyDelimiter(t *testing.T) {
	runner := jinjaChatTestRunner(
		t,
		`{% if tools %}<tool_call>{{ tools|tojson }}</tool_call>{% endif %}`,
	)
	source, root, triggers, err := runner.ChatToolGrammar(
		[]ChatTool{toolWeatherDefinition()},
		false,
		true,
		true,
	)
	testcheck.NoError(t, err)
	for _, fragment := range []string{
		`tool-root ::= tool-call`,
		`tool-call ::= "<tool_call>"`,
		`weather`,
		`arguments`,
		`city`,
		`root ::= "{" space name-kv "," space arguments-kv`,
	} {
		if !strings.Contains(source, fragment) {
			t.Fatalf("grammar does not contain %q:\n%s", fragment, source)
		}
	}
	if root != "tool-root" ||
		len(triggers) != 1 ||
		triggers[0] != `(<tool_call>)` {
		t.Fatalf("root=%q triggers=%q", root, triggers)
	}
}

func TestToolGrammarCanForbidParallelCalls(t *testing.T) {
	for name, template := range map[string]string{
		"json":   `<tool_call>{{ tools|tojson }}</tool_call>`,
		"hermes": `<tool_call><function=example_function_name></function></tool_call>`,
	} {
		t.Run(name, func(t *testing.T) {
			runner := jinjaChatTestRunner(t, template)
			source, _, _, err := runner.ChatToolGrammar(
				[]ChatTool{toolWeatherDefinition()},
				true,
				false,
				false,
			)
			testcheck.NoError(t, err)
			if strings.Contains(source, `(space tool-call)*`) {
				t.Fatalf("single-call grammar allows repetition:\n%s", source)
			}
			if !strings.Contains(source, `tool-root ::= tool-call`) {
				t.Fatalf("single-call root is missing:\n%s", source)
			}
		})
	}
}

func TestHermesToolGrammarUsesFunctionAndParameterTags(t *testing.T) {
	runner := jinjaChatTestRunner(
		t,
		`<tool_call><function=example_function_name></function></tool_call>`,
	)
	source, root, triggers, err := runner.ChatToolGrammar(
		[]ChatTool{toolWeatherDefinition()},
		false,
		true,
		true,
	)
	testcheck.NoError(t, err)
	for _, fragment := range []string{
		`"<function=weather>"`,
		`"<parameter=city>"`,
		`tool-parameter-value ::= ( [^<\x00] | tool-parameter-lt )*`,
		`tool-parameter-lt ::= "<" (`,
	} {
		if !strings.Contains(source, fragment) {
			t.Fatalf("grammar does not contain %q:\n%s", fragment, source)
		}
	}
	if root != "tool-root" ||
		len(triggers) != 1 ||
		triggers[0] != `(<tool_call>)` {
		t.Fatalf("root=%q triggers=%q", root, triggers)
	}
}

func TestToolGrammarRejectsUnsupportedTemplatesAndUnsafeNames(t *testing.T) {
	runner := jinjaChatTestRunner(t, `plain`)
	if _, _, _, err := runner.ChatToolGrammar(
		[]ChatTool{toolWeatherDefinition()},
		false,
		true,
		true,
	); err == nil {
		t.Fatal("unsupported template was accepted")
	}
	runner.file.Metadata[0].Value.Data = `<tool_call>`
	tool := toolWeatherDefinition()
	tool.Function.Name = "bad>name"
	if _, _, _, err := runner.ChatToolGrammar(
		[]ChatTool{tool},
		false,
		true,
		true,
	); err == nil {
		t.Fatal("unsafe tool name was accepted")
	}
}
