//go:build integration

package latentvideo

import (
	"os"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testskip"
)

// TestCommonPromptRowsAreKeptForTheNextProcess holds the keeping of a common
// prompt's rows in a store, on the real checkpoint. The common prompt is the
// negative prompt every prompt-form request of the generator carries, 126
// tokens, and beside it each request brings a prompt of its own.
//
// The first process encodes both in one pass and keeps the common prompt's
// rows in the store under the identity its build counted. The next process,
// which holds no rows, encodes the request's own prompt alone: the common
// prompt's rows come from the store, bit for bit the rows the first computed,
// and the request's own prompt, which is no common one, was kept nowhere and
// has the same rows encoded again. A process asked for the common prompt
// alone makes no pass. A build of another identity finds nothing under its
// own name and encodes the common prompt again. And a build that counted no
// identity writes nothing to the store, whatever prompt it is told is common.
func TestCommonPromptRowsAreKeptForTheNextProcess(t *testing.T) {
	testskip.Short(t, "streams the 11.4GB encoder checkpoint four times")
	spec := WanTextConditioningSpec(wanModelDir(t), 512)
	if _, err := os.Stat(spec.EncoderCheckpoint); err != nil {
		t.Skipf("UNAVAILABLE: encoder checkpoint absent: %v", err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	const own = "a heron stands in reeds at first light, held for the kept-rows test"
	spec.Common, spec.Kept, spec.KeptUnder = referenceNegativePrompt, store, "identity-one"
	// nextProcess forgets every row this process's encoders keep, as a process that has just started holds none.
	nextProcess := func() *promptEncoder {
		t.Helper()
		promptEncoders.Lock()
		clear(promptEncoders.byKey)
		promptEncoders.Unlock()
		encoder, err := promptEncoderFor(spec)
		if err != nil {
			t.Fatal(err)
		}
		return encoder
	}
	encoder := nextProcess()
	tokens := func(prompt string) (kept int) {
		t.Helper()
		_, mask, err := encoder.tokenizer.EncodeWithMask(prompt)
		if err != nil {
			t.Fatal(err)
		}
		for _, bit := range mask {
			kept += bit
		}
		return kept
	}
	ownTokens, commonTokens := tokens(own), tokens(spec.Common)
	bound := func(of TextConditioningSpec) bool {
		t.Helper()
		_, bound, err := artifact.ResolveAlias(ctx, store, encoder.commonRowsName(of))
		if err != nil {
			t.Fatal(err)
		}
		return bound
	}

	first, _, stats, err := encodePrompts(ctx, spec, own, spec.Common)
	if err != nil {
		t.Fatal(err)
	}
	if stats.TokenRows != ownTokens+commonTokens || !bound(spec) {
		t.Fatalf("the first process encoded %d rows and kept the common prompt's in the store: %t; want the %d rows of both prompts, and kept", stats.TokenRows, bound(spec), ownTokens+commonTokens)
	}

	nextProcess()
	second, _, stats, err := encodePrompts(ctx, spec, own, spec.Common)
	if err != nil {
		t.Fatal(err)
	}
	if stats.TokenRows != ownTokens {
		t.Fatalf("the next process encoded %d rows; want the %d of the request's own prompt alone, the common prompt's %d read from the store", stats.TokenRows, ownTokens, commonTokens)
	}
	if !slices.Equal(second[1], first[1]) || !slices.Equal(second[0], first[0]) || len(second[1]) != commonTokens*encoder.plan.Config.Dim {
		t.Fatal("the rows the next process holds are not, bit for bit, the rows the first computed")
	}

	nextProcess()
	if _, _, stats, err = encodePrompts(ctx, spec, spec.Common); err != nil || stats.Layers != 0 {
		t.Fatalf("a process asked for the common prompt alone passed through %d blocks (%v); want no pass", stats.Layers, err)
	}

	another := spec
	another.KeptUnder = "identity-two"
	nextProcess()
	if bound(another) {
		t.Fatal("a build of another identity finds rows under its own name before it has kept any")
	}
	if _, _, stats, err = encodePrompts(ctx, another, spec.Common); err != nil || stats.TokenRows != commonTokens || !bound(another) {
		t.Fatalf("a build of another identity encoded %d rows (%v) and kept them: %t; want the common prompt's %d encoded again and kept under its own name", stats.TokenRows, err, bound(another), commonTokens)
	}

	uncounted := spec
	uncounted.Common, uncounted.KeptUnder = own, ""
	nextProcess()
	_, before := store.Head()
	if _, _, stats, err = encodePrompts(ctx, uncounted, own); err != nil || stats.TokenRows != ownTokens {
		t.Fatalf("a build that counted no identity encoded %d rows (%v); want the prompt's %d", stats.TokenRows, err, ownTokens)
	}
	if _, after := store.Head(); after != before {
		t.Fatalf("a build that counted no identity wrote to the store: its head moved from %d to %d", before, after)
	}
}
