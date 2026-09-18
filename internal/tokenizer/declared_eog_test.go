package tokenizer

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/gguf"
	"overgo/internal/testutil"
)

func TestDeclaredEOGTokens(t *testing.T) {
	base := []gguf.Metadata{
		gguf.StringMetadata("tokenizer.ggml.model", "gpt2"),
		gguf.StringMetadata("tokenizer.ggml.pre", "gpt-2"),
		gguf.ArrayMetadata("tokenizer.ggml.tokens", gguf.ValueTypeString, []string{"a", "b", "stop-a", "stop-b", "<think>", "<|im_end|>", "fim-pad"}),
		gguf.ArrayMetadata("tokenizer.ggml.merges", gguf.ValueTypeString, []string{}),
		gguf.Uint32Metadata("tokenizer.ggml.eos_token_id", 2),
		gguf.Uint32Metadata("tokenizer.ggml.fim_pad_token_id", 6),
	}
	for _, test := range []struct {
		name      string
		extension gguf.Metadata
		invalid   bool
	}{
		{"declared_list", gguf.ArrayMetadata(MetadataEOSTokenIDs, gguf.ValueTypeUint32, []uint32{3, 2, 3}), false},
		{"out_of_range", gguf.ArrayMetadata(MetadataEOSTokenIDs, gguf.ValueTypeUint32, []uint32{3, 99}), true},
		{"wrong_element_type", gguf.ArrayMetadata(MetadataEOSTokenIDs, gguf.ValueTypeString, []string{"3"}), true},
		{"wrong_scalar_type", gguf.StringMetadata(MetadataEOSTokenIDs, "3"), true},
	} {
		t.Run(test.name, func(t *testing.T) {
			metadata := append(slices.Clone(base), test.extension)
			vocab, err := Load(&gguf.File{Metadata: metadata})
			if test.invalid {
				if err == nil {
					t.Fatal("invalid declared stop array accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got, want := vocab.EOGTokens(), []TokenID{2, 3, 5, 6}; !slices.Equal(got, want) {
				t.Fatalf("EOG=%v, want %v", got, want)
			}
			if vocab.IsEOG(4) || vocab.IsEOG(NullToken) || vocab.IsEOG(TokenID(vocab.Len())) {
				t.Fatal("undeclared or invalid stop classified as terminal")
			}
			original, err := Load(&gguf.File{Metadata: base})
			if err != nil {
				t.Fatal(err)
			}
			before, err := original.Encode("ab", EncodeOptions{})
			if err != nil {
				t.Fatal(err)
			}
			after, err := vocab.Encode("ab", EncodeOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(before, []TokenID{0, 1}) || !slices.Equal(before, after) {
				t.Fatalf("encoding changed: before=%v after=%v", before, after)
			}
			if original.IsEOG(3) {
				t.Fatal("undeclared stop was already treated as terminal")
			}
			identity := testutil.ArtifactID(t, artifact.KindTokenizer, "declared EOS fixture")
			beforeArtifact, err := original.Artifact(identity)
			if err != nil {
				t.Fatal(err)
			}
			afterArtifact, err := vocab.Artifact(identity)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Contains(afterArtifact.Tokens[3].Roles, VocabularyRoleEOS) {
				t.Fatal("declared EOS role was lost in semantic vocabulary")
			}
			if _, err := ExactVocabularyMapping(beforeArtifact, afterArtifact); err == nil {
				t.Fatal("mapping accepted different declared EOS semantics")
			}
		})
	}
}
