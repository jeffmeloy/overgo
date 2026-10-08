package inference

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// chatTemplateRendersScript writes the goldens this test reads, from Python's
// Jinja2 as transformers runs a chat template. It is kept where Python is
// kept, on the machine that ran it and not in git: the goldens are the data.
const chatTemplateRendersScript = "scripts/python_golden_references/chat_template_renders.py"

// chatTemplateBattery is testdata/chat_templates/conversations.json: the
// conversations every served template is rendered for, each as a chat request
// carries it, with the tools they offer named once.
type chatTemplateBattery struct {
	BOS           string              `json:"bos_token"`
	EOS           string              `json:"eos_token"`
	Tools         map[string]ChatTool `json:"tools"`
	Conversations []struct {
		Name                string        `json:"name"`
		AddGenerationPrompt bool          `json:"add_generation_prompt"`
		EnableThinking      bool          `json:"enable_thinking"`
		Tools               []string      `json:"tools"`
		Messages            []ChatMessage `json:"messages"`
	} `json:"conversations"`
}

// chatTemplateRenders is one template's golden: what Python's Jinja2 renders
// for each conversation of the battery, or that it raised.
type chatTemplateRenders struct {
	Template string `json:"template"`
	Source   string `json:"source"`
	Renders  []struct {
		Name  string `json:"name"`
		Text  string `json:"text"`
		Error string `json:"error"`
	} `json:"renders"`
}

// TestChatTemplatesRenderAsTransformersDoes holds the template engine to the
// language the served models' templates are written in: Python's Jinja2 as
// transformers runs a chat template. Every template the served models ship is
// rendered for every conversation of the battery, and the text must be, byte
// for byte, what transformers renders for the same request: with a tool's
// keys in the order the request holds them, a call's arguments as the mapping
// they are, and JSON written as transformers' tojson writes it.
//
// The engine was first written to reproduce another Go template library, and
// its goldens were that library's output, nine refusals among them; the
// models' own renderings refuse none of these conversations.
func TestChatTemplatesRenderAsTransformersDoes(t *testing.T) {
	directory := filepath.Join("testdata", "chat_templates")
	raw, err := os.ReadFile(filepath.Join(directory, "conversations.json"))
	if err != nil {
		t.Fatal(err)
	}
	var battery chatTemplateBattery
	if err := json.Unmarshal(raw, &battery); err != nil || len(battery.Conversations) == 0 {
		t.Fatalf("the battery holds no conversation: %v", err)
	}
	runner := jinjaChatTestRunner(t, "")
	if bos, eos := vocabularyTokenText(runner.vocab, runner.vocab.BOS), vocabularyTokenText(runner.vocab, runner.vocab.EOS); bos != battery.BOS || eos != battery.EOS {
		t.Fatalf("the test's vocabulary opens and closes with %q and %q, the battery with %q and %q", bos, eos, battery.BOS, battery.EOS)
	}
	templates, err := filepath.Glob(filepath.Join(directory, "*.jinja"))
	if err != nil || len(templates) == 0 {
		t.Fatalf("no template is on file: %v", err)
	}
	slices.Sort(templates)
	for _, path := range templates {
		base := strings.TrimSuffix(filepath.Base(path), ".jinja")
		t.Run(base, func(t *testing.T) {
			source, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			raw, err := os.ReadFile(filepath.Join(directory, "golden", base+".json"))
			if err != nil {
				t.Fatalf("%v; write the goldens with %s", err, chatTemplateRendersScript)
			}
			var golden chatTemplateRenders
			if err := json.Unmarshal(raw, &golden); err != nil || len(golden.Renders) != len(battery.Conversations) {
				t.Fatalf("the golden holds %d renders for %d conversations (%v); write the goldens again with %s",
					len(golden.Renders), len(battery.Conversations), err, chatTemplateRendersScript)
			}
			same := 0
			for index, conversation := range battery.Conversations {
				want := golden.Renders[index]
				if want.Name != conversation.Name {
					t.Fatalf("render %d is of %s and conversation %d is %s", index, want.Name, index, conversation.Name)
				}
				options := ChatFormatOptions{AddGenerationPrompt: conversation.AddGenerationPrompt, EnableThinking: conversation.EnableThinking}
				for _, name := range conversation.Tools {
					tool, held := battery.Tools[name]
					if !held {
						t.Fatalf("%s offers %s, which the battery does not hold", conversation.Name, name)
					}
					options.Tools = append(options.Tools, tool)
				}
				got, err := runner.formatJinjaChatNative(string(source), conversation.Messages, options)
				// The vocabulary writes the opening token itself, so the engine's text leaves it out.
				text := strings.TrimPrefix(want.Text, battery.BOS)
				switch {
				case want.Error != "" && err == nil:
					t.Errorf("%s: rendered where transformers raises %s:\n%q", conversation.Name, want.Error, got)
				case want.Error != "":
					same++
				case err != nil:
					t.Errorf("%s: refused where transformers renders: %v", conversation.Name, err)
				case got != text:
					t.Errorf("%s: parts from transformers' rendering%s", conversation.Name, firstDifference(text, got))
				default:
					same++
				}
			}
			t.Logf("%s: %d of %d conversations render as transformers renders them", base, same, len(battery.Conversations))
		})
	}
}

// TestToolReachesATemplateInTheRequestsKeyOrder holds what a template is
// handed for a tool, whatever made the tool. A tool a request wrote, or one
// made from parameters a request wrote, is written out with its keys in the
// request's order and not in a Go map's. A tool built from fields alone has
// no such order and is written as its fields marshal. A tool changed after it
// was read is written as it now is, not as the request wrote it.
func TestToolReachesATemplateInTheRequestsKeyOrder(t *testing.T) {
	const written = `{"function": {"parameters": {"type": "object", "required": ["zeta"], "properties": {"zeta": {"type": "string"}, "alpha": {"type": "integer", "default": 3}}}, "name": "lookup"}, "type": "function"}`
	var requested ChatTool
	if err := json.Unmarshal([]byte(written), &requested); err != nil {
		t.Fatal(err)
	}
	const parameters = `{"type": "object", "required": ["zeta"], "properties": {"zeta": {"type": "string"}, "alpha": {"type": "integer", "default": 3}}}`
	made, err := FunctionTool("lookup", "", json.RawMessage(parameters))
	if err != nil {
		t.Fatal(err)
	}
	built := ChatTool{Type: ChatToolTypeFunction, Function: ChatToolDefinition{Name: "lookup", Parameters: requested.Function.Parameters}}
	renamed := requested
	renamed.Function.Name = "search"
	const sorted = `{"properties": {"alpha": {"default": 3, "type": "integer"}, "zeta": {"type": "string"}}, "required": ["zeta"], "type": "object"}`
	runner := jinjaChatTestRunner(t, `{{ tools[0] | tojson }}`)
	for name, held := range map[string]struct {
		tool ChatTool
		want string
	}{
		"a tool a request wrote":                    {requested, written},
		"a tool made from a request's parameters":   {made, `{"type": "function", "function": {"name": "lookup", "parameters": ` + parameters + `}}`},
		"a tool built from fields":                  {built, `{"type": "function", "function": {"name": "lookup", "parameters": ` + sorted + `}}`},
		"a tool renamed after its request was read": {renamed, `{"type": "function", "function": {"name": "search", "parameters": ` + sorted + `}}`},
	} {
		got, err := runner.FormatChatWithOptions([]ChatMessage{{Role: ChatRoleUser, Content: "hello"}}, ChatFormatOptions{Tools: []ChatTool{held.tool}})
		if err != nil || got != held.want {
			t.Errorf("%s is handed to a template as\n%s\nand not as\n%s\n(%v)", name, got, held.want, err)
		}
	}
	if _, err := FunctionTool("lookup", "", json.RawMessage(`["not", "an", "object"]`)); err == nil {
		t.Error("parameters that are not an object made a tool")
	}
	if _, err := FunctionTool("lookup", "", json.RawMessage(`{"cut": `)); err == nil {
		t.Error("parameters that are not JSON made a tool")
	}
}

// firstDifference shows a stretch of each text around the first byte at which
// they part.
func firstDifference(want, got string) string {
	const before, after = 60, 140
	at := 0
	for at < len(want) && at < len(got) && want[at] == got[at] {
		at++
	}
	from := max(at-before, 0)
	return "\n transformers: " + strings.ReplaceAll(want[from:min(at+after, len(want))], "\n", `\n`) +
		"\n       overgo: " + strings.ReplaceAll(got[from:min(at+after, len(got))], "\n", `\n`)
}
