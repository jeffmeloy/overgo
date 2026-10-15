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
	"overgo/internal/evaluation"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// promptIDsScript writes the fixtures this test reads, from each model's own
// tokenizer and chat template. It is kept where Python is kept, on the
// machine that ran it and not in git: the fixtures are the data.
const promptIDsScript = "scripts/python_golden_references/chat_prompt_ids.py"

// knownPromptFaults are the fixtures whose questions do not reach their
// model as its own files give them, by the fixture's name, each with what
// parts the two and the finding that records it. A fixture named here must
// differ, so that mending its fault fails this test until its entry is
// removed; a fixture not named must not. A model may have one fixture for
// each way its questions are asked: Qwen2.5-0.5B's questions asked as plain
// text, the way its reading is taken, reach it as its own tokens.
var knownPromptFaults = map[string]string{
	"Qwen2.5-0.5B": "asked as a conversation, the scored file carrying no chat template, the question is shaped by overgo's built-in ChatML with no " +
		"system turn, where the model's own template opens with one (finding e683b1e7)",
	"Unlimited-OCR-bf16": "the scored file names an earlier DeepSeek tokenizer's splitting rule, so a mark and the line break after it are " +
		"two tokens where the model's own tokenizer joins them (finding 34979bff)",
}

// promptChat and promptSequence are the protocols a fixture names beside raw
// completion: questions asked through the chat template, and the bare
// sequences of a sequence-scoring suite.
const promptChat, promptSequence = "chat", "sequence"

// promptReference is one model's fixture: the scored file, the protocol its
// reading was asked under, and each case with the tokens the model's own
// tokenizer and template give for it.
type promptReference struct {
	Schema   string            `json:"schema"`
	Model    string            `json:"model"`
	File     string            `json:"file"`
	Protocol string            `json:"protocol"`
	Opener   string            `json:"opener"`
	Source   string            `json:"source"`
	Note     string            `json:"note"`
	Files    map[string]string `json:"files"`
	Cases    []struct {
		evaluation.MultipleChoiceCase
		Text          string                `json:"text"`
		IDs           []tokenizer.TokenID   `json:"ids"`
		Continuations [][]tokenizer.TokenID `json:"continuations"`
	} `json:"cases"`
}

// TestPromptsReachEachModelAsItsOwnTokenizerGivesThem holds the first stretch
// of the path every text reading takes, from an evaluation's question to the
// tokens the model reads, to the model's own files. For each model with a
// fixture, the cases of its reading's suite are read as the evaluation gives
// them to the model, from the scored file's template and vocabulary and none
// of its weights: under the chat-template protocol the question as one user
// turn with the model's turn opened, under raw completion the context and
// each candidate's continuation. Every token is held to what the tokenizer
// and chat template the model itself ships give for the same words.
//
// A wrong token here is a fault no score shows as one: the model answers
// whatever it is given. Two models are given other tokens than their own
// files give, and are named with what parts them; the test holds them to
// differing still, so that the entry cannot outlive the fault.
func TestPromptsReachEachModelAsItsOwnTokenizerGivesThem(t *testing.T) {
	root := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	fixtures, err := filepath.Glob(filepath.Join(root, "fixtures/prompt_ids/*.json"))
	if err != nil || len(fixtures) == 0 {
		t.Fatalf("no model's own tokens are on file (%v); write them with %s", err, promptIDsScript)
	}
	held, faulted := 0, map[string]bool{}
	for _, fixture := range fixtures {
		raw, err := os.ReadFile(fixture)
		if err != nil {
			t.Fatal(err)
		}
		var reference promptReference
		if err := json.Unmarshal(raw, &reference); err != nil || len(reference.Cases) == 0 {
			t.Fatalf("%s holds no case: %v", fixture, err)
		}
		t.Run(strings.TrimSuffix(filepath.Base(fixture), filepath.Ext(fixture)), func(t *testing.T) {
			model := roots.ResolveModelPath(reference.File)
			if _, err := os.Stat(model); err != nil {
				t.Skipf("UNAVAILABLE: %s scored file %s cannot be read: %v", reference.Model, reference.File, err)
			}
			// The cases as their suite holds them: a question with its candidates,
			// or the bare sequence of a sequence-scoring suite.
			type listed struct {
				evaluation.MultipleChoiceCase
				Text string `json:"text,omitempty"`
			}
			cases := struct {
				Cases []listed `json:"cases"`
			}{}
			for _, own := range reference.Cases {
				held := listed{MultipleChoiceCase: own.MultipleChoiceCase}
				if reference.Protocol == promptSequence {
					held.Text = own.Text
				}
				cases.Cases = append(cases.Cases, held)
			}
			encoded, err := json.Marshal(cases)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "cases.json")
			if err := os.WriteFile(path, encoded, 0o600); err != nil {
				t.Fatal(err)
			}
			chat := reference.Protocol == promptChat
			read, err := readPrompts(model, path, chat)
			if err != nil || len(read) != len(reference.Cases) {
				t.Fatalf("%s: %d of %d cases were read: %v", reference.Model, len(read), len(reference.Cases), err)
			}
			differ, first := 0, -1
			for index, own := range reference.Cases {
				ours := read[index]
				same := slices.Equal(ours.IDs, own.IDs) && len(ours.Continuations) == len(own.Continuations)
				for candidate := 0; same && candidate < len(own.Continuations); candidate++ {
					same = slices.Equal(ours.Continuations[candidate], own.Continuations[candidate])
				}
				if !same {
					differ++
					if first < 0 {
						first = index
					}
				}
			}
			name := strings.TrimSuffix(filepath.Base(fixture), filepath.Ext(fixture))
			fault, known := knownPromptFaults[name]
			switch {
			case known && differ == 0:
				t.Errorf("%s is given its own tokens on every case: its fault is mended, so remove its entry (%s)", name, fault)
			case known:
				faulted[name] = true
				t.Logf("%s (%s): %d of %d cases reach it as other tokens than its own files give: %s", reference.Model, reference.Protocol, differ, len(read), fault)
			case differ != 0:
				ours, own := read[first], reference.Cases[first]
				at := 0
				for at < len(ours.IDs) && at < len(own.IDs) && ours.IDs[at] == own.IDs[at] {
					at++
				}
				t.Errorf("%s (%s): %d of %d cases reach it as other tokens than its own files give; %s parts at token %d of %d and %d, and the text given is\n%q\nwhere its own files give\n%q",
					reference.Model, reference.Protocol, differ, len(read), own.Name, at, len(ours.IDs), len(own.IDs), ours.Text, own.Text)
			default:
				held++
				t.Logf("%s (%s): %d cases reach it as the tokens its own files give%s", reference.Model, reference.Protocol, len(read), noted(reference.Note))
			}
		})
	}
	for name := range knownPromptFaults {
		if !faulted[name] && held+len(faulted) == len(fixtures) {
			t.Errorf("%s is named as a known fault and no fixture read here shows it", name)
		}
	}
	t.Logf("%d fixtures reach their model as its own tokens, %d do not, of %d on file", held, len(faulted), len(fixtures))
}

// noted is a fixture's note, set off from what precedes it.
func noted(note string) string {
	if note == "" {
		return ""
	}
	return "; " + note
}
