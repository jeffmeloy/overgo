package inference

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// TestPromptReaderReadsAPromptWithoutTheModel holds the reader to what it is
// for: from a model file's template and vocabulary alone it shapes a
// conversation as the file's template says, and gives the tokens a
// generation starts from and the tokens a scoring of candidates reads, which
// are the vocabulary's own encodings of those texts. A file that is not there
// is refused.
func TestPromptReaderReadsAPromptWithoutTheModel(t *testing.T) {
	t.Parallel()
	const template = "{% for message in messages %}a{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}b{% endif %}"
	path := testutil.HermeticLlamaGGUFWith(t, 16, testutil.GGUFScalar("tokenizer.chat_template", gguf.ValueTypeString, template))
	reader, err := OpenPromptReader(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := reader.Close(); err != nil {
			t.Error(err)
		}
	})
	shaped, err := reader.FormatChatWithOptions([]ChatMessage{{Role: ChatRoleUser, Content: "cd"}}, ChatFormatOptions{AddGenerationPrompt: true})
	if err != nil || shaped != "acdb" {
		t.Fatalf("the conversation is shaped as %q (%v), and the file's template gives %q", shaped, err, "acdb")
	}
	file, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = file.Close() })
	vocabulary, err := tokenizer.Load(file)
	if err != nil {
		t.Fatal(err)
	}
	encode := func(text string, parseSpecial bool) []tokenizer.TokenID {
		t.Helper()
		ids, err := vocabulary.Encode(text, tokenizer.EncodeOptions{AddSpecial: true, ParseSpecial: parseSpecial})
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	started, err := reader.GenerationTokens(shaped, true)
	if err != nil || len(started) == 0 || !slices.Equal(started, encode(shaped, true)) {
		t.Fatalf("a generation starts from %v (%v), and the vocabulary reads the text as %v", started, err, encode(shaped, true))
	}
	candidates := []string{"c", "d"}
	context, continuations, err := reader.ContinuationTokens("ab", candidates, false)
	if err != nil || len(context) == 0 || len(continuations) != len(candidates) {
		t.Fatalf("the candidates are read as %v after %v: %v", continuations, context, err)
	}
	for index, candidate := range candidates {
		if whole := slices.Concat(context, continuations[index]); len(continuations[index]) == 0 || !slices.Equal(whole, encode("ab"+candidate, false)) {
			t.Errorf("the prompt with %q is read as %v, and the vocabulary reads it as %v", candidate, whole, encode("ab"+candidate, false))
		}
	}
	if _, err := OpenPromptReader(filepath.Join(t.TempDir(), "absent.gguf")); err == nil {
		t.Fatal("a file that is not there was opened")
	}
}
