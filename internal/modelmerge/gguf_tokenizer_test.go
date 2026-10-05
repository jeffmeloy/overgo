package modelmerge

import (
	"testing"

	"overgo/internal/gguf"
)

// TestDifferingTokenizerNamesWhatMakesAMergeNoise holds a merge to models of the
// base's tokenizer: a vocabulary the other declares differently, or does not
// declare, is named, while a chat template, which a fine-tune may change, and
// keys outside the tokenizer are not.
func TestDifferingTokenizerNamesWhatMakesAMergeNoise(t *testing.T) {
	t.Parallel()
	declare := func(pairs ...string) *gguf.File {
		file := &gguf.File{}
		for index := 0; index < len(pairs); index += 2 {
			file.Metadata = append(file.Metadata, gguf.Metadata{Key: pairs[index], Value: gguf.Value{Data: pairs[index+1]}})
		}
		return file
	}
	base := declare("tokenizer.ggml.tokens", "a b c", "tokenizer.chat_template", "plain", "general.name", "base")
	for name, check := range map[string]struct {
		other *gguf.File
		want  string
	}{
		"the same":           {declare("tokenizer.ggml.tokens", "a b c", "tokenizer.chat_template", "plain", "general.name", "base"), ""},
		"another template":   {declare("tokenizer.ggml.tokens", "a b c", "tokenizer.chat_template", "chatml", "general.name", "tuned"), ""},
		"another vocabulary": {declare("tokenizer.ggml.tokens", "a b d"), "tokenizer.ggml.tokens"},
		"no vocabulary":      {declare("general.name", "base"), "tokenizer.ggml.tokens"},
	} {
		if got := differingTokenizer(base, check.other); got != check.want {
			t.Errorf("%s: differing = %q, want %q", name, got, check.want)
		}
	}
}
