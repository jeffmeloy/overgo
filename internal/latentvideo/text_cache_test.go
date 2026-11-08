package latentvideo

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestAVideoRequestThatChangesOnlyItsSeedEncodesNoText holds Wan's text contexts to the shared text cache: on
// 2026-11-08 the Wan runtime kept one prompt pair, so every new prompt streamed the standard negative prompt through the
// CPU encoder again. A request that changes only its seed encodes nothing, a new prompt encodes itself alone, a prompt
// met two requests back is not encoded again, and a changed encoder encodes both.
func TestAVideoRequestThatChangesOnlyItsSeedEncodesNoText(t *testing.T) {
	t.Parallel()
	var memo wanConditioning
	var encoded []string
	encode := func(prompts []string) ([][]float32, error) {
		encoded = append(encoded, prompts...)
		contexts := make([][]float32, len(prompts))
		for index, prompt := range prompts {
			contexts[index] = []float32{float32(len(prompt))}
		}
		return contexts, nil
	}
	for _, request := range []struct {
		encoder, prompt string
		want            []string
	}{
		{"umt5", "a red fox", []string{"a red fox", referenceNegativePrompt}},
		{"umt5", "a red fox", nil},
		{"umt5", "a boat", []string{"a boat"}},
		{"umt5", "a red fox", nil},
		{"umt5 changed", "a red fox", []string{"a red fox", referenceNegativePrompt}},
	} {
		encoded = nil
		contexts := testcheck.Must(memo.recall(request.encoder, []string{request.prompt, referenceNegativePrompt}, encode))(t)
		if !slices.Equal(encoded, request.want) || contexts[0][0] != float32(len(request.prompt)) || contexts[1][0] != float32(len(referenceNegativePrompt)) {
			t.Fatalf("a request for %q under %s encoded %q and gave %v, want %q encoded and each prompt's own context", request.prompt, request.encoder, encoded, contexts, request.want)
		}
	}
}
