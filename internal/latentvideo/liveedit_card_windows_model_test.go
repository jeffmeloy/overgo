//go:build windows && modeltest

package latentvideo

import (
	"math"
	"testing"

	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/modeltest"
	"overgo/internal/testcheck"
)

// TestLiveEditEncodesItsInstructionOnTheCard holds LiveEdit's condition to the card's text encoder: on 2026-11-08 its
// instruction still streamed the 11 GB umT5 checkpoint through the host, about 62 s of an 88 s clip, after Wan's prompt
// and the student's had moved to the card. The condition an instruction resolves to is encoded on the card, which holds
// the encoder with no pass over the checkpoint, and its context agrees with the host's within the gates two BF16
// engines are held to (goldenCrossEngineMaxTolerance, goldenCrossEngineMeanTolerance).
func TestLiveEditEncodesItsInstructionOnTheCard(t *testing.T) {
	cudatest.Require(t)
	editor := modeltest.Directory(t, "LiveEdit")
	wan, checkpoint := LiveEditArtifacts(editor)
	policy := testcheck.Must(resolveEditPolicy(editor))(t)
	profile := testcheck.Must(ResolveProfile(wan))(t)
	base := testcheck.Must(LoadDenoiserConfig(wan, profile.Policy))(t)
	const instruction = "turn the daylight into a snowy evening"
	forgetPromptEncoders(t)
	card := testcheck.Must(chunkCausalCondition(t.Context(), wan, checkpoint, policy, profile, instruction))(t)
	encoder := testcheck.Must(promptEncoderFor(WanTextConditioningSpec(wan, base.TextLen)))(t)
	if encoder.card == nil || encoder.passes != 0 {
		t.Fatalf("the encoder holds card %v after %d passes over the checkpoint, want the instruction encoded on the card", encoder.card, encoder.passes)
	}
	forgetPromptEncoders(t)
	compiled := testcheck.Must(CompileReferenceEditCheckpoint(checkpoint, base))(t)
	var projectionBytes int64
	projection := testcheck.Must(func() (projectionWeights, error) {
		weights, read, err := compiled.loadTextProjection()
		projectionBytes = read
		return weights, err
	}())(t)
	host := testcheck.Must(textConditioningWithWeights(t.Context(), WanTextConditioningSpec(wan, base.TextLen), instruction, projection, projectionBytes))(t)
	if host.Encoder.Engine == "card" || len(host.Context) != len(card.TextContext) {
		t.Fatalf("the host's context ran on %q with %d values, want the host's with the card's %d", host.Encoder.Engine, len(host.Context), len(card.TextContext))
	}
	var worst, sum float64
	for index, want := range host.Context {
		difference := math.Abs(float64(card.TextContext[index]) - float64(want))
		worst, sum = max(worst, difference), sum+difference
	}
	mean := sum / float64(len(host.Context))
	t.Logf("LiveEdit's instruction on the card against the host: max abs diff %.6g, mean abs diff %.6g", worst, mean)
	if worst > goldenCrossEngineMaxTolerance || mean > goldenCrossEngineMeanTolerance {
		t.Fatalf("max abs diff %g and mean %g, want under %g and %g", worst, mean, goldenCrossEngineMaxTolerance, goldenCrossEngineMeanTolerance)
	}
}
