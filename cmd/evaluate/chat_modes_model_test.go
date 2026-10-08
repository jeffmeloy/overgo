//go:build modeltest

package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// chatModesScript writes the fixtures this test reads, from each model's own
// tokenizer and chat template. It is kept where Python is kept, on the
// machine that ran it and not in git: the fixtures are the data.
const chatModesScript = "scripts/python_golden_references/chat_mode_ids.py"

// The conversations that offer tools, and of them the two that hold an
// earlier call and its answer.
var (
	toolConversations = []string{
		"tool-offered", "tool-offered-thinking-on", "system-two-tools-offered",
		"tool-called-and-answered", "marked-tool-offered", "marked-tool-called-and-answered",
	}
	toolCallConversations = []string{"tool-called-and-answered", "marked-tool-called-and-answered"}
)

// toolTurnsFault is what parts a conversation with tools from the model's own
// rendering of it: the template engine and the tool values it is handed were
// written to reproduce another Go library and not Python Jinja2 as
// transformers runs it, which the models' templates are written for.
const toolTurnsFault = " (finding 399a51cc)"

// knownChatModeFaults are the conversations that do not reach a model as its
// own template and tokenizer give them, each model's with what parts them and
// the finding that records it. A conversation named here must differ or be
// refused, so that mending its fault fails this test until its entry is
// removed; a conversation not named must not.
var knownChatModeFaults = map[string]struct {
	conversations []string
	fault         string
}{
	"MiniCPM5-1B-native-chat": {toolConversations, "tools offered with no system turn are refused, the template's lstrip() with no argument being one the engine does not take; " +
		"with a system turn each tool is written as {}, the engine's tojson given an argument writing a tool as an empty object" + toolTurnsFault},
	"Qwen3.5-4B": {toolConversations, "a tool's keys are written in alphabetical order and its < and & as escapes, where the model's own template keeps " +
		"the request's order and writes the characters" + toolTurnsFault},
	"gemma-4-E4B-it": {toolCallConversations, "an earlier tool call is refused by the template's own check, its arguments being handed to the template " +
		"as a text and not as the mapping they hold" + toolTurnsFault},
	"gemma-4-12B-it-fp8": {toolCallConversations, "an earlier tool call is refused, the template calling get on a call and the engine " +
		"giving a tool call no methods" + toolTurnsFault},
	"gemma-4-12B-it-fp8-native": {toolCallConversations, "an earlier tool call is refused, the template calling get on a call and the engine " +
		"giving a tool call no methods" + toolTurnsFault},
	"Qwen2.5-0.5B": {append([]string{"thinking-off", "thinking-on", "second-question-after-an-answer"}, toolConversations...),
		"the scored file carries no chat template, so a conversation with no system turn is rendered without the one the model's own template " +
			"opens with, and tools are refused (finding e683b1e7)"},
}

// chatModeReference is one model's fixture: the scored file, and each
// conversation as a chat request carries it, with the text the model's own
// template renders for it with the model's turn opened and the tokens its
// own tokenizer gives that text.
type chatModeReference struct {
	Model  string `json:"model"`
	File   string `json:"file"`
	Source string `json:"source"`
	Cases  []struct {
		Name     string                  `json:"name"`
		Thinking bool                    `json:"thinking"`
		Tools    []inference.ChatTool    `json:"tools"`
		Messages []inference.ChatMessage `json:"messages"`
		Text     string                  `json:"text"`
		IDs      []tokenizer.TokenID     `json:"ids"`
	} `json:"cases"`
}

// TestChatModesReachAModelAsItsOwnTemplateGivesThem holds what a chat request
// reaches a model as, past the one shape an evaluation asks in: with the
// thinking switch on and off, with a system turn, with tools offered, with a
// tool called and answered, and with a second question after an answer. Each
// conversation is rendered from the scored file's template and tokenized
// from its vocabulary, with none of its weights, and both the text and the
// tokens are held to what the model's own template and tokenizer give.
func TestChatModesReachAModelAsItsOwnTemplateGivesThem(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	fixtures, err := filepath.Glob(filepath.Join(root, "fixtures", "chat_modes", "*.json"))
	if err != nil || len(fixtures) == 0 {
		t.Fatalf("no model's own conversations are on file (%v); write them with %s", err, chatModesScript)
	}
	for _, fixture := range fixtures {
		raw, err := os.ReadFile(fixture)
		if err != nil {
			t.Fatal(err)
		}
		var reference chatModeReference
		if err := json.Unmarshal(raw, &reference); err != nil || len(reference.Cases) == 0 {
			t.Fatalf("%s holds no conversation: %v", fixture, err)
		}
		t.Run(reference.Model, func(t *testing.T) {
			model := roots.ResolveModelPath(reference.File)
			if _, err := os.Stat(model); err != nil {
				t.Skipf("the scored file %s is not on this machine", reference.File)
			}
			reader, err := inference.OpenPromptReader(model)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = reader.Close() })
			known := knownChatModeFaults[reference.Model]
			held, faulted := 0, 0
			for _, own := range reference.Cases {
				// The tokens are what the model reads. The text is not compared whole:
				// a template's opening token is left to the tokenizer here, and
				// written by the template there.
				text, err := reader.FormatChatWithOptions(own.Messages, inference.ChatFormatOptions{
					Tools: own.Tools, AddGenerationPrompt: true, EnableThinking: own.Thinking,
				})
				var ids []tokenizer.TokenID
				if err == nil {
					ids, err = reader.GenerationTokens(text, true)
				}
				same := err == nil && slices.Equal(ids, own.IDs)
				switch named := slices.Contains(known.conversations, own.Name); {
				case named && same:
					t.Errorf("%s reaches the model as its own files give it: its fault is mended, so remove its entry (%s)", own.Name, known.fault)
				case named:
					faulted++
				case same:
					held++
				case err != nil:
					t.Errorf("%s: the conversation is refused: %v", own.Name, err)
				default:
					ours, theirs := partedAt(text, own.Text)
					t.Errorf("%s: %d tokens where the model's own files give %d; the texts part at\n%q\nwhere the model's own template gives\n%q",
						own.Name, len(ids), len(own.IDs), ours, theirs)
				}
			}
			if faulted != len(known.conversations) {
				t.Errorf("%d conversations are named as known faults and the fixture holds %d of them", len(known.conversations), faulted)
			}
			if faulted != 0 {
				t.Logf("%s: %d conversations reach it as its own files give them and %d do not: %s", reference.Model, held, faulted, known.fault)
				return
			}
			t.Logf("%s: %d conversations reach it as its own files give them", reference.Model, held)
		})
	}
}

// partedAt returns a stretch of each text from a little before the first
// character at which they part, once the opening piece only the second holds
// is set aside.
func partedAt(ours, theirs string) (string, string) {
	const before, after, anchor = 40, 160, 16
	if at := strings.Index(theirs, ours[:min(anchor, len(ours))]); at > 0 {
		theirs = theirs[at:]
	}
	parted := 0
	for parted < len(ours) && parted < len(theirs) && ours[parted] == theirs[parted] {
		parted++
	}
	from := max(0, parted-before)
	return ours[from:min(len(ours), parted+after)], theirs[from:min(len(theirs), parted+after)]
}
