package latentimage

import (
	"slices"
	"testing"

	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestAWideningAdapterGivesTheDenoiserItsControl holds MOSS's control input (moss-ultra-sections/depth-control): an
// adapter that replaces video_embedder with a wider one compiles a denoiser, in bfloat16 storage and in F32, that
// reads a control of the extra channels beside the latent, its replaced weight bound from the adapter's file in the
// model's storage; a program without such an adapter reads none. On 2026-11-17 the replacement bound in F32 left the
// bfloat16 graph's product refused and its build's next weight a nil binding.
func TestAWideningAdapterGivesTheDenoiserItsControl(t *testing.T) {
	t.Parallel()
	spec := denoiserSpec()
	adapter := &mediaAdapter{lowRank: map[string]lowRankPatch{}, diffs: map[string]string{}, replaced: map[string]replacement{
		"video_embedder.weight": {tensor: "video_embedder.weight", shape: []uint64{uint64(spec.Hidden), uint64(3 * spec.InChannels)}},
		"video_embedder.bias":   {tensor: "video_embedder.bias", shape: []uint64{uint64(spec.Hidden)}},
	}}
	for _, storage := range []dtype.Type{dtype.BF16, dtype.F32} {
		p := testcheck.Must(compileRotaryDenoiserProgram(spec, 1e-6, 2, 3, 4, nil, storage, rotaryOptions{adapter: adapter}))(t)
		if p.control == nil || !slices.Equal(p.control.Shape.Slice(), []uint64{6, uint64(2 * spec.InChannels)}) {
			t.Fatalf("a %v denoiser with a widening adapter reads the control %v; want 6 tokens of %d channels", storage, p.control, 2*spec.InChannels)
		}
		weight := p.adapterInputs.Node("video_embedder.weight")
		if weight == nil || weight.Type != storage {
			t.Errorf("a %v denoiser binds the replaced embedder as %v; want it from the adapter in the model's storage", storage, weight)
		}
		if p.weightInputs.Node("video_embedder.weight") != nil {
			t.Errorf("a %v denoiser binds the model's embedder beside the adapter's", storage)
		}
	}
	plain := testcheck.Must(compileRotaryDenoiserProgram(spec, 1e-6, 2, 3, 4, nil, dtype.BF16, rotaryOptions{}))(t)
	if plain.control != nil {
		t.Error("a denoiser without a widening adapter reads a control")
	}
}
