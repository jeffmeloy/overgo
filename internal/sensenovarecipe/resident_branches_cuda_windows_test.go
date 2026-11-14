//go:build modeltest && windows

package sensenovarecipe

import (
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/jsonfile"
	"overgo/internal/testcheck"
)

// requestUploadMost bounds the bytes a request sends to the card while it prepares, once the generator holds both
// branches: the prompts' rows and the flow session's inputs. Each branch's layer weights are about 16 GB.
const requestUploadMost = 1 << 30

// TestASenseNovaRequestUploadsNoBranchWeights holds SenseNova's resident branches (guidance-ops/resident-branches): a
// generator that keeps both branches' weights on the card lends them to every request, so a request's prepare sends
// the card its prompts and inputs, not the 32 GB of weights it read and uploaded for every picture before.
func TestASenseNovaRequestUploadsNoBranchWeights(t *testing.T) {
	cudatest.Require(t)
	var oracle generationLeadershipOracle
	testcheck.NoError(t, jsonfile.Decode(senseNovaGenerationGold, &oracle))
	_, _, generator := newProductionGenerationFixture(t)
	for _, prompt := range []string{oracle.Request.Prompt, "a lighthouse on a cliff at dawn"} {
		request := GenerationRequest{Prompt: prompt, Width: oracle.Request.Width, Height: oracle.Request.Height, Steps: 2,
			Seed: oracle.Request.Seed, CFGScale: oracle.Request.CFGScale, TimestepShift: oracle.Request.TimestepShift}
		before := testcheck.Must(generator.worker.ExecutionStats(t.Context()))(t)
		testcheck.Must(generator.prepare(t.Context(), request))(t)
		after := testcheck.Must(generator.worker.ExecutionStats(t.Context()))(t)
		uploaded := after.HostToDeviceBytes - before.HostToDeviceBytes
		t.Logf("%q prepared in %.2fs, sending the card %.2f GB", prompt, generator.Stats().Prepare.Seconds(), float64(uploaded)/1e9)
		if uploaded > requestUploadMost {
			t.Errorf("%q sent the card %.2f GB while it prepared, want at most %.2f: the request uploaded branch weights",
				prompt, float64(uploaded)/1e9, float64(requestUploadMost)/1e9)
		}
	}
}
