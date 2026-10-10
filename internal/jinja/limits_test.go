package jinja

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// servedTemplates are the chat templates the served models ship, as the
// inference package keeps them beside the conversations it renders them for.
const servedTemplates = "../inference/testdata/chat_templates"

// The limits the tests and the fuzz target render under: far below a served
// conversation's, so a render that runs away is refused in an instant.
var testLimits = Limits{Steps: 200_000, Bytes: 4 << 20}

// servedContexts reads the battery's conversations as a chat request carries
// them: each a context a served template renders.
func servedContexts(t testing.TB) []map[string]any {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(servedTemplates, "conversations.json"))
	if err != nil {
		t.Fatal(err)
	}
	var battery struct {
		BOS           string                     `json:"bos_token"`
		EOS           string                     `json:"eos_token"`
		Tools         map[string]json.RawMessage `json:"tools"`
		Conversations []struct {
			AddGenerationPrompt bool            `json:"add_generation_prompt"`
			EnableThinking      bool            `json:"enable_thinking"`
			Tools               []string        `json:"tools"`
			Messages            json.RawMessage `json:"messages"`
		} `json:"conversations"`
	}
	if err := json.Unmarshal(raw, &battery); err != nil || len(battery.Conversations) == 0 {
		t.Fatalf("the battery holds no conversation: %v", err)
	}
	var contexts []map[string]any
	for _, conversation := range battery.Conversations {
		messages, err := DecodeJSON(conversation.Messages)
		if err != nil {
			t.Fatal(err)
		}
		context := map[string]any{
			"bos_token": battery.BOS, "eos_token": battery.EOS, "messages": messages,
			"add_generation_prompt": conversation.AddGenerationPrompt, "enable_thinking": conversation.EnableThinking,
		}
		var tools []any
		for _, name := range conversation.Tools {
			tool, err := DecodeJSON(battery.Tools[name])
			if err != nil {
				t.Fatal(err)
			}
			tools = append(tools, tool)
		}
		if tools != nil {
			context["tools"] = tools
		}
		contexts = append(contexts, context)
	}
	return contexts
}

// servedSources reads every served model's template.
func servedSources(t testing.TB) map[string]string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(servedTemplates, "*.jinja"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("no served template is on file: %v", err)
	}
	sources := map[string]string{}
	for _, path := range paths {
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		sources[filepath.Base(path)] = string(source)
	}
	return sources
}

// chatEnvironment is an environment with the two functions a chat template is
// given beside the engine's own, under limits.
func chatEnvironment(limits Limits) *Env {
	env := New()
	env.SetLimits(limits)
	env.SetGlobal("raise_exception", func(args []any, _ map[string]any) (any, error) {
		return nil, errors.New("raised")
	})
	env.SetGlobal("strftime_now", func([]any, map[string]any) (any, error) { return "26 Jul 2024", nil })
	return env
}

// TestTemplatePastItsLimitsIsRefused renders a template past each limit and
// holds the refusal to the limits' own error, with nothing rendered: a loop
// that never ends soon enough, a range and a repeated string too large to
// build, charged before they are built, text written out without end, a
// replacement that grows a string, a search repeated over a large value, a
// power, brackets and blocks nested too deep, a macro that calls itself, and
// a value that holds itself, which the first run of the fuzz target's seeds
// found to exhaust the stack, written, compared, turned into JSON or used as
// a key.
// What Python renders without a limit, the engine renders the same with them:
// a string repeated no times, a division's remainder. A remainder by zero is
// an error, as it is in Python, and not a fault of the process. Every served
// model's template renders every conversation of the battery well inside the
// limits a chat render is given, and the test says how much of them each
// took.
func TestTemplatePastItsLimitsIsRefused(t *testing.T) {
	env := chatEnvironment(testLimits)
	deep := func(open, shut string) string {
		return strings.Repeat(open, maxNesting+1) + "1" + strings.Repeat(shut, maxNesting+1)
	}
	for name, template := range map[string]string{
		"a loop inside a loop":            "{% for i in range(1000) %}{% for j in range(1000) %}{% endfor %}{% endfor %}",
		"a range too long to build":       "{% set all = range(10 ** 12) %}",
		"a range of a wrapping length":    "{{ range(-9223372036854775807, 9223372036854775807)|length }}",
		"a string repeated":               "{{ 'template' * 10 ** 9 }}",
		"a string repeated past a number": "{{ 'template' * 9223372036854775807 }}",
		"text written in a loop":          "{% for i in range(100000) %}" + strings.Repeat("x", 1000) + "{% endfor %}",
		"a value built in a loop":         "{% set ns = namespace(text='x') %}{% for i in range(64) %}{% set ns.text = ns.text ~ ns.text %}{% endfor %}",
		"a replacement that grows":        "{% set long = 'a' * 100000 %}{{ long.replace('a', long)|length }}",
		"a search repeated":               "{% set long = 'a' * 100000 %}{% for i in range(100000) %}{{ 'b' in long }}{% endfor %}",
		"a power":                         "{{ 3 ** 10000000 }}",
		"brackets nested":                 "{{ " + deep("(", ")") + " }}",
		"lists nested":                    "{{ " + deep("[", "]") + " }}",
		"conditionals chained":            "{{ (" + strings.Repeat("1 if 0 else ", maxNesting+1) + "1) }}",
		"operators chained":               "{{ 1" + strings.Repeat(" + 1", maxNesting+1) + " }}",
		"blocks nested":                   strings.Repeat("{% if 1 %}", maxNesting+1) + strings.Repeat("{% endif %}", maxNesting+1),
		"a macro that calls itself":       "{% macro again(n) %}{{ again(n + 1) }}{% endmacro %}{{ again(0) }}",
		"a value that holds itself":       "{% set ns = namespace(a=1) %}{% set ns.a = ns %}{{ ns }}",
		"one compared with itself":        "{% set ns = namespace(a=1) %}{% set ns.a = ns %}{{ ns == ns }}",
		"one written as JSON":             "{% set ns = namespace(a=1) %}{% set ns.a = ns %}{{ ns|tojson }}",
		"one used as a key":               "{% set ns = namespace(a=1) %}{% set ns.a = ns %}{{ {ns: 1} }}",
	} {
		if text, err := env.Render(template, nil); !errors.Is(err, errLimit) || text != "" {
			t.Errorf("%s rendered %d bytes and ended with %v; want it refused past its limits", name, len(text), err)
		}
	}
	for template, want := range map[string]string{
		"{{ 'ab' * 3 }}|{{ 'ab' * 0 }}|{{ 'ab' * -2 }}|":  "ababab|||",
		"{{ 7 % 3 }} {{ 2 ** 10 }} {{ range(3)|length }}": "1 1024 3",
		"{{ 'a-b-c'.replace('-', '+++') }}":               "a+++b+++c",
	} {
		if text, err := env.Render(template, nil); err != nil || text != want {
			t.Errorf("%s rendered %q, %v; want %q", template, text, err, want)
		}
	}
	if text, err := env.Render("{{ 1 % 0 }}", nil); err == nil || errors.Is(err, errLimit) {
		t.Errorf("a remainder by zero rendered %q, %v; want an error that is not a limit's", text, err)
	}

	// The limits a chat render is given hold every served template with room.
	unbounded := chatEnvironment(Limits{})
	contexts := servedContexts(t)
	for name, source := range servedSources(t) {
		most := budget{}
		for _, context := range contexts {
			_, left, err := unbounded.render(source, context)
			if err != nil {
				// A template refuses some of the battery's conversations itself.
				continue
			}
			whole := newBudget(Limits{})
			most.steps, most.bytes = max(most.steps, whole.steps-left.steps), max(most.bytes, whole.bytes-left.bytes)
		}
		t.Logf("%s: at most %d steps and %d bytes for a conversation of the battery", name, most.steps, most.bytes)
		if most.steps == 0 || most.steps > testLimits.Steps || most.bytes > testLimits.Bytes {
			t.Errorf("%s takes %d steps and %d bytes for a conversation of the battery; the tests' limits are %+v", name, most.steps, most.bytes, testLimits)
		}
	}
}

// FuzzTemplate holds the engine to one property under any template: it never
// faults, and a render ends inside the limits it was given or with an error.
// It is seeded with every served model's template, with templates that are
// malformed, cut short or at a boundary of the language, and with each input a
// run of it found to break the property, which testdata keeps.
func FuzzTemplate(f *testing.F) {
	for _, source := range servedSources(f) {
		f.Add(source)
		// A template cut short, where a block or an expression is left open.
		f.Add(source[:len(source)/2])
	}
	for _, seed := range []string{
		"", "{{", "}}", "{%", "%}", "{#", "{{ }}", "{% %}", "{% if %}", "{% for %}", "{% endfor %}", "{% for x in %}{% endfor %}",
		"{{ 1 +", "{{ ( }}", "{{ [1, }}", "{{ {'a': }}", "{{ 'unterminated }}", "{{ x[ }}", "{{ x[::] }}", "{{ x[::0] }}", "{{ x. }}",
		"{{ 1 % 0 }}", "{{ 1 // 0 }}", "{{ 1 / 0 }}", "{{ 'a' * -1 }}", "{{ 2 ** -1 }}", "{{ -9223372036854775808 }}", "{{ 9223372036854775807 + 1 }}",
		"{{ range(0) }}", "{{ range(1, 0, -1) }}", "{{ range(1, 2, 0) }}", "{{ range(10 ** 18) }}", "{{ x|unknown }}", "{{ x is unknown }}",
		"{{ messages[::-1] }}", "{{ messages[9999] }}", "{{ messages[-9999:9999:3] }}", "{{ messages|tojson(indent=2) }}", "{{ messages|map(attribute='role')|list }}",
		"{% set ns = namespace(a=1) %}{% set ns.a = ns %}{{ ns.a.a.a.a }}", "{% macro m(a, b=a) %}{{ a }}{{ b }}{% endmacro %}{{ m(1) }}{{ m() }}",
		"{% for a, b in messages %}{{ a }}{% endfor %}", "{% for m in messages if m %}{{ loop.previtem }}{{ loop.nextitem }}{% else %}none{% endfor %}",
		"{%- if messages -%} x {%+ endif +%}", "{# comment #}{{ '{{' }}{{ '%}' }}", "{{ \"\\u0000\\ud800\" }}", "{{ messages|dictsort }}", "{{ none|items }}",
	} {
		f.Add(seed)
	}
	env := chatEnvironment(testLimits)
	contexts := servedContexts(f)
	f.Fuzz(func(t *testing.T, source string) {
		for _, context := range contexts[:min(len(contexts), 2)] {
			text, left, err := env.render(source, context)
			if err == nil && (left.steps < 0 || left.bytes < 0 || len(text) > testLimits.Bytes) {
				t.Fatalf("a render of %d bytes ended with %d steps and %d bytes left of %+v", len(text), left.steps, left.bytes, testLimits)
			}
		}
	})
}
