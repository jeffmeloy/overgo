package inference

import (
	"slices"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// boundaryVocab is a SentencePiece vocabulary in which a choice cue and
// its letter merge across the prompt boundary the way the MMLU-Pro
// prompts do under the Qwen tokenizers: "Answer: " ends in a lone space
// token that the candidate " A" replaces, and "Answer:" + " A" merges
// into one token outright.
func boundaryVocab(t *testing.T, mergedCue bool) *tokenizer.Vocab {
	t.Helper()
	tokens := []string{
		"<unk>", "<s>", "</s>", "▁", "A", "n", "s", "w", "e", "r", ":",
		"An", "Ans", "Answ", "Answe", "Answer", "Answer:", "▁A",
	}
	if mergedCue {
		tokens = append(tokens, "Answer:▁A")
	}
	scores := make([]float32, len(tokens))
	for index := 11; index < len(tokens); index++ {
		scores[index] = float32(index)
	}
	types := make([]int32, len(tokens))
	for index := range types {
		types[index] = int32(tokenizer.TokenNormal)
	}
	types[0] = int32(tokenizer.TokenUnknown)
	types[1] = int32(tokenizer.TokenControl)
	types[2] = int32(tokenizer.TokenControl)
	vocab, err := tokenizer.Load(&gguf.File{Metadata: []gguf.Metadata{
		testutil.GGUFScalar("tokenizer.ggml.model", gguf.ValueTypeString, "llama"),
		testutil.GGUFScalar("tokenizer.ggml.bos_token_id", gguf.ValueTypeUint32, uint32(1)),
		testutil.GGUFScalar("tokenizer.ggml.eos_token_id", gguf.ValueTypeUint32, uint32(2)),
		testutil.GGUFScalar("tokenizer.ggml.unknown_token_id", gguf.ValueTypeUint32, uint32(0)),
		testutil.GGUFScalar("tokenizer.ggml.add_bos_token", gguf.ValueTypeBool, true),
		testutil.GGUFScalar("tokenizer.ggml.add_space_prefix", gguf.ValueTypeBool, false),
		testutil.GGUFArray("tokenizer.ggml.tokens", gguf.ValueTypeString, tokens),
		testutil.GGUFArray("tokenizer.ggml.scores", gguf.ValueTypeFloat32, scores),
		testutil.GGUFArray("tokenizer.ggml.token_type", gguf.ValueTypeInt32, types),
	}})
	if err != nil {
		t.Fatal(err)
	}
	return vocab
}

// TestContinuationTokensScoreAcrossAMergedBoundary: a candidate whose
// encoding is no longer than the prompt's still scores when the merge
// happened at the boundary: the shared context stops before the merged
// token and the candidate's tokens past it are the continuation. The
// MMLU-Pro pass refused every Qwen case on this shape.
func TestContinuationTokensScoreAcrossAMergedBoundary(t *testing.T) {
	bos := tokenizer.TokenID(1)
	single := func(vocab *tokenizer.Vocab, text string) tokenizer.TokenID {
		ids, err := vocab.Encode(text, tokenizer.EncodeOptions{})
		if err != nil || len(ids) != 1 {
			t.Fatalf("%q encodes to %v (%v)", text, ids, err)
		}
		return ids[0]
	}
	// "Answer: " + "A": the cue's trailing space token is replaced by
	// "▁A", so the candidate's encoding is exactly as long as the
	// prompt's; the shared context ends at the cue and the letter is the
	// continuation.
	plain := boundaryVocab(t, false)
	runner := &Runner{preparedModel: preparedModel{vocab: plain}}
	prompt, continuations, err := runner.continuationTokens("Answer: ", []string{"A"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(prompt, []tokenizer.TokenID{bos, single(plain, "Answer:")}) ||
		len(continuations) != 1 || !slices.Equal(continuations[0], []tokenizer.TokenID{single(plain, "▁A")}) {
		t.Fatalf("trailing-space boundary: prompt %v continuations %v", prompt, continuations)
	}
	// "Answer:" + " A" where the vocabulary holds the merged token: the
	// candidate's encoding is shorter than the prompt's; the merged token
	// is the continuation after the BOS context.
	merged := boundaryVocab(t, true)
	runner = &Runner{preparedModel: preparedModel{vocab: merged}}
	prompt, continuations, err = runner.continuationTokens("Answer:", []string{" A"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(prompt, []tokenizer.TokenID{bos}) ||
		len(continuations) != 1 || !slices.Equal(continuations[0], []tokenizer.TokenID{single(merged, "Answer:▁A")}) {
		t.Fatalf("merged boundary: prompt %v continuations %v", prompt, continuations)
	}
	// A candidate that adds nothing at all is still refused.
	if _, _, err := runner.continuationTokens("Answer:", []string{""}, false); err == nil {
		t.Fatal("empty candidate accepted")
	}
}
