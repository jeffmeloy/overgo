//go:build integration

package latentvideo

import (
	"os"
	"slices"
	"testing"

	"overgo/internal/testskip"
)

// TestPromptAlreadyEncodedIsNotEncodedAgain holds the encoder's keeping of a
// prompt's rows on the real checkpoint: a prompt conditioned a second time
// makes no pass over the checkpoint and gives the context the first gave;
// prompts conditioned together are encoded in one pass, those of them not
// kept already; and a prompt encoded beside another has the rows it has
// encoded alone, bit for bit.
func TestPromptAlreadyEncodedIsNotEncodedAgain(t *testing.T) {
	testskip.Short(t, "streams the 11.4GB encoder checkpoint five times")
	spec := WanTextConditioningSpec(wanModelDir(t), 512)
	if _, err := os.Stat(spec.EncoderCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: encoder checkpoint absent: %v", err)
	}
	weights, sourceBytes, err := loadProjectionWeights(spec.ProjectionDir)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := promptEncoderFor(spec)
	if err != nil {
		t.Fatal(err)
	}
	passes := func() int {
		encoder.mu.Lock()
		defer encoder.mu.Unlock()
		return encoder.passes
	}
	// Prompts no other test of this process conditions.
	prompts := []string{
		"a heron stands in reeds at first light, held for this test",
		"the heron lifts off over grey water, held for this test",
		"rain begins on the reeds, held for this test",
	}
	before := passes()
	first, err := TextConditioning(t.Context(), spec, prompts[0])
	if err != nil {
		t.Fatal(err)
	}
	again, err := TextConditioning(t.Context(), spec, prompts[0])
	if err != nil {
		t.Fatal(err)
	}
	if passes() != before+1 || first.Encoder.Layers == 0 || again.Encoder.Layers != 0 || !slices.Equal(first.Context, again.Context) {
		t.Fatalf("a prompt conditioned twice made %d passes (the first through %d blocks, the second %d) and gave the same context: %t",
			passes()-before, first.Encoder.Layers, again.Encoder.Layers, slices.Equal(first.Context, again.Context))
	}
	together, err := textConditioningsWithWeights(t.Context(), spec, weights, sourceBytes, prompts...)
	if err != nil {
		t.Fatal(err)
	}
	if passes() != before+2 || !slices.Equal(together[0].Context, first.Context) {
		t.Fatalf("three prompts, one of them kept, made %d passes more", passes()-before-1)
	}
	for index, prompt := range prompts[1:] {
		tokens, mask, err := encoder.tokenizer.EncodeWithMask(prompt)
		if err != nil {
			t.Fatal(err)
		}
		alone, _, err := EncodeTokensStreamed(t.Context(), spec.EncoderCheckpoint, encoder.plan, tokens, mask)
		if err != nil {
			t.Fatal(err)
		}
		beside, _, _, err := encodePrompts(t.Context(), spec, prompt)
		if err != nil {
			t.Fatal(err)
		}
		if together[index+1].TokenCount*encoder.plan.Config.Dim != len(alone) || !slices.Equal(beside[0], alone) {
			t.Errorf("prompt %d encoded beside another is not the prompt encoded alone", index+1)
		}
	}
	if got := passes(); got != before+2 {
		t.Errorf("reading kept rows made %d passes more", got-before-2)
	}
	t.Logf("three prompts conditioned in %d passes over the checkpoint; %d tokens went through the second", passes()-before, together[1].Encoder.TokenRows)
}
