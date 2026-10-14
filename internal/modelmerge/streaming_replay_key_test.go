package modelmerge

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

// TestAReplayKeyNamesTheCodeThatComputesIt holds the offline merge's stage to what it was written from.
//   - one request = one plan, one directory for each source, one destination
//   - the first run seals its first tensor and stops at its second, whose sum is not finite, so a stage is left
//   - then one source's bytes change in its own directory, or the plan's coefficients change, and the merge is asked for again at the same destination
//   - the repeat => a refusal that publishes nothing, or a whole merge of what it was given
//   - a published tensor the earlier run sealed is refused, because it is a checkpoint mixed from two requests
func TestAReplayKeyNamesTheCodeThatComputesIt(t *testing.T) {
	t.Parallel()
	const sealed, stopped = "layer.0", "layer.1"
	tensors := []streamingFixtureTensor{{name: sealed, shape: []uint64{2}}, {name: stopped, shape: []uint64{2}}}
	for _, repeat := range []struct {
		name         string
		bytes        map[string][]float32
		coefficients []float64
		want         []float32
	}{
		{"offline-merge", map[string][]float32{sealed: {100, 200}, stopped: {3, 4}}, []float64{1, 1}, []float32{110, 220}},
		{"offline-merge-plan", nil, []float64{0.5, 0.5}, []float32{5.5, 11}},
	} {
		t.Run(repeat.name, func(t *testing.T) {
			models := []artifact.ID{
				testutil.ArtifactID(t, artifact.KindModel, "replay-key merge first"),
				testutil.ArtifactID(t, artifact.KindModel, "replay-key merge second"),
			}
			first := writeStreamingFixture(t, map[string][]float32{sealed: {1, 2}, stopped: {math.MaxFloat32, 1}})
			second := writeStreamingFixture(t, map[string][]float32{sealed: {10, 20}, stopped: {math.MaxFloat32, 1}})
			sources := []StreamingSource{{Model: models[0], Directory: first}, {Model: models[1], Directory: second}}
			destination := filepath.Join(t.TempDir(), "output")
			plan := sealStreamingPlan(t, composition.OfflineArtifactTaskArithmetic, models, []float64{1, 1}, tensors)
			if _, err := ExecuteStreaming(t.Context(), plan, sources, destination); err == nil {
				t.Fatal("the first run published a sum that is not finite, want it stopped at its second tensor")
			}
			if _, err := os.Stat(destination); !os.IsNotExist(err) {
				t.Fatalf("the first run left its destination visible (%v), want only a stage", err)
			}
			if repeat.bytes != nil {
				if err := safetensors.Save(filepath.Join(first, streamingSingleShardName), repeat.bytes, map[string][]int{sealed: {2}, stopped: {2}}, nil); err != nil {
					t.Fatal(err)
				}
			}
			again := sealStreamingPlan(t, composition.OfflineArtifactTaskArithmetic, models, repeat.coefficients, tensors)
			if _, err := ExecuteStreaming(t.Context(), again, sources, destination); err != nil {
				if _, statErr := os.Stat(destination); !os.IsNotExist(statErr) {
					t.Fatalf("the repeat refused (%v) and left its destination visible (%v), want a refusal to publish nothing", err, statErr)
				}
				return
			}
			if got := readStreamingTensor(t, destination, sealed); !slices.Equal(got, repeat.want) {
				t.Fatalf("the repeat published %s = %v, want %v: it went on from the stage an earlier run left for other bytes or another plan", sealed, got, repeat.want)
			}
		})
	}
}
