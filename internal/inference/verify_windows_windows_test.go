//go:build windows

package inference

import (
	"slices"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/recipe"
	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// strataFlashNextWindows is strataFlashNext with its gated-delta layers at the geometry Strata's verify kernels are
// compiled for - a state 128 wide over two key heads and four value heads - so the runner decodes through Strata's
// verifier.
var strataFlashNextWindows = func() flashNextGeometry {
	geometry := strataFlashNext
	geometry.ssmState, geometry.ssmGroups, geometry.ssmValueHeads = 128, 2, 4
	return geometry
}()

// TestVerifyWindowsMatchPlainDecoding holds Strata's verifier (src/core/verify.cpp) to plain decoding: on the
// Flash-Next fixture at Strata's geometry, its gated-delta, QSA and lexical layers' fixed state where it sits, a greedy
// request's windows - of one token and of two, its drafts mostly rejected by a random draft layer and accepted by a
// copying one - give plain decoding's tokens, token for token, each window's commit replaying the tokens it keeps; plain
// decoding is the same runner's one-token windows, and both equal the tokens the runner gives with the verifier off, its
// states written out a step.
func TestVerifyWindowsMatchPlainDecoding(t *testing.T) {
	requireIntegration(t)
	cudatest.Require(t)
	const generated = 24
	trunk := writeFlashNextFixture(t, strataFlashNextWindows)
	greedy, err := sampling.New(sampling.Config{Temperature: 0})
	if err != nil {
		t.Fatal(err)
	}
	prompt := make([]tokenizer.TokenID, 40)
	for index := range prompt {
		prompt[index] = tokenizer.TokenID(3 + (index*7+index/5)%int(flashNextVocab-3))
	}
	generate := func(runner *Runner, plain bool, onDraft func(count, accepted int)) []tokenizer.TokenID {
		t.Helper()
		// The plain text runs past the generation, which a copying draft's last window reads.
		tokens := generated
		if plain {
			tokens += hyperDraftTokens + 1
		}
		ids, _, err := runner.Generate(t.Context(), "", GenerateOptions{
			MaxNewTokens: tokens, Sampler: greedy, DeviceGreedy: true, PromptTokenIDs: prompt, ContinueAfterEOG: true,
			PlainDecode: plain, OnDraftSpan: onDraft,
		})
		if err != nil {
			t.Fatal(err)
		}
		return ids[len(prompt):]
	}
	// The verifier off: the steps write their state out, as before the port.
	stepped, err := openFixtureRunnerWithResidency(trunk, OpenOptions{}, recipe.ResidencyDeviceNative)
	if err != nil {
		t.Fatal(err)
	}
	stepped.qsaWindows = false
	written := generate(stepped, true, nil)
	if err := stepped.Close(); err != nil {
		t.Fatal(err)
	}
	if len(slices.Compact(slices.Clone(written))) == 1 {
		t.Fatalf("the fixture selects one token throughout (%v)", written)
	}
	keptOverall := 0
	for _, copyTrunk := range []bool{false, true} {
		runner, err := openFixtureRunnerWithResidency(trunk,
			OpenOptions{Draft: writeFlashNextMTPFixture(t, trunk, copyTrunk, strataFlashNextWindows)}, recipe.ResidencyDeviceNative)
		if err != nil {
			t.Fatal(err)
		}
		if !runner.windowsDecode() {
			_ = runner.Close()
			t.Fatal("the fixture at Strata's geometry does not decode through the verifier")
		}
		plain := generate(runner, true, nil)
		var drafted, accepted int
		speculated := generate(runner, false, func(count, kept int) { drafted, accepted = drafted+count, accepted+kept })
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(plain, written) {
			t.Fatalf("copy=%t: the verifier's plain decoding %v differs from the steps' %v", copyTrunk, plain, written)
		}
		if !slices.Equal(speculated, plain[:generated]) {
			t.Fatalf("copy=%t: the windows decode %v, plain decoding %v", copyTrunk, speculated, plain[:generated])
		}
		// The copying layer drafts the token the trunk selected, kept exactly where the plain text repeats it.
		wantDrafted, wantAccepted := copyDraftAcceptance(plain, generated)
		if drafted == 0 || copyTrunk && (drafted != wantDrafted || accepted != wantAccepted) || !copyTrunk && accepted == drafted {
			t.Fatalf("copy=%t: %d of %d drafts kept; want windows of two tokens, the random layer's mostly rejected, the copying layer's %d of %d",
				copyTrunk, accepted, drafted, wantAccepted, wantDrafted)
		}
		t.Logf("copy=%t: %d tokens through windows, %d of %d drafts kept", copyTrunk, generated, accepted, drafted)
		keptOverall += accepted
	}
	// A window that keeps its draft commits both its tokens, one that rejects it its first alone: both ran.
	if keptOverall == 0 {
		t.Fatal("no window kept its draft, so no commit kept a whole window")
	}
}
