package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestReadPromptsGivesEachCaseAsTheModelIsGivenIt holds the reader of prompts
// on a model file small enough to write in a test. Under raw completion each
// case is the context its candidates share and each candidate's own tokens
// after it. Under the chat-template protocol the question goes into the
// user's turn with the evaluation's instruction, which this file's four
// letters cannot write: the refusal names the case, and the protocol itself
// is held on the scored models by the test of their own tokens. A file of no
// cases, a file that is not JSON, a file that is not there and a model file
// that is not one are refused.
func TestReadPromptsGivesEachCaseAsTheModelIsGivenIt(t *testing.T) {
	t.Parallel()
	const template = "{% for message in messages %}a{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}b{% endif %}"
	model := testutil.HermeticLlamaGGUFWith(t, 64, testutil.GGUFScalar("tokenizer.chat_template", gguf.ValueTypeString, template))
	directory := t.TempDir()
	write := func(name, content string) string {
		t.Helper()
		path := filepath.Join(directory, name)
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	// The file's vocabulary holds four letters, so the cases are written in them.
	cases := write("cases.json", `{"cases":[{"name":"one","prompt":"ab","candidates":["c","d"],"answer":0},{"name":"two","prompt":"dc","candidates":["a","b"],"answer":1}]}`)

	raw, err := readPrompts(model, cases, false)
	if err != nil || len(raw) != 2 {
		t.Fatalf("%d cases were read under raw completion: %v", len(raw), err)
	}
	for _, read := range raw {
		if read.User != "" || read.Text != "" || len(read.IDs) == 0 || len(read.Continuations) != len(read.Candidates) ||
			slices.ContainsFunc(read.Continuations, func(tokens []tokenizer.TokenID) bool { return len(tokens) == 0 }) {
			t.Errorf("under raw completion %s is read as %+v", read.Name, read)
		}
	}
	if slices.Equal(raw[0].Continuations[0], raw[0].Continuations[1]) {
		t.Errorf("two candidates are read as the same tokens: %v", raw[0].Continuations)
	}

	if _, err := readPrompts(model, cases, true); err == nil || !strings.Contains(err.Error(), "case one") {
		// The instruction the evaluation adds is in words the four-letter vocabulary cannot write.
		t.Fatalf("a question the file's vocabulary cannot write was read: %v", err)
	}

	for name, refused := range map[string][2]string{
		"a file of no cases":           {model, write("none.json", `{"cases":[]}`)},
		"a file that is not JSON":      {model, write("broken.json", `{"cases":`)},
		"a file that is not there":     {model, filepath.Join(directory, "absent.json")},
		"a model file that is not one": {cases, cases},
	} {
		if _, err := readPrompts(refused[0], refused[1], false); err == nil {
			t.Errorf("%s was read", name)
		}
	}
}
