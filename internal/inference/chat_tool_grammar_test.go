package inference

import (
	"os"
	"strings"
	"testing"
)

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
	if err != nil {
		t.Fatal(err)
	}
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
			if err != nil {
				t.Fatal(err)
			}
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
	if err != nil {
		t.Fatal(err)
	}
	for _, fragment := range []string{
		`"<function=weather>"`,
		`"<parameter=city>"`,
		`tool-parameter-value ::= [^<\x00]*`,
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

func TestNativeToolGrammarsCompileForLocalTemplateFamilies(t *testing.T) {
	for _, fixture := range []struct {
		name string
		env  string
	}{
		{name: "qwen-json", env: "OVERGO_QWEN3_MODEL"},
		{name: "qwen35-hermes", env: "OVERGO_QWEN35_MODEL"},
		{name: "bonsai-hermes", env: "OVERGO_BONSAI_MODEL"},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			path := os.Getenv(fixture.env)
			if path == "" {
				t.Skip(fixture.env + " is not set")
			}
			runner, err := openFixtureRunner(path, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer runner.Close()
			source, root, patterns, err := runner.ChatToolGrammar(
				[]ChatTool{toolWeatherDefinition()},
				false,
				true,
				true,
			)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := runner.CompileLazyGBNF(
				source,
				root,
				patterns,
				nil,
			); err != nil {
				t.Fatal(err)
			}
			source, root, patterns, err = runner.ChatToolGrammar(
				[]ChatTool{toolWeatherDefinition()},
				true,
				true,
				true,
			)
			if err != nil {
				t.Fatal(err)
			}
			if len(patterns) != 0 {
				t.Fatalf("required grammar triggers = %q", patterns)
			}
			if _, err := runner.CompileGBNF(source, root); err != nil {
				t.Fatal(err)
			}
		})
	}
}
