package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestAMOSSBlocksElementwiseChainsLaunchOnce holds the elementwise chains a MOSS block makes around its multiplies, built
// as the denoiser builds them (latentimage rotaryGraph.norm, modulate, rotate and joint), to one launch each
// (moss-fusion/neighbors): a block norm with its rounded weight, modulation scale and shift, and a head's norm with its
// weight; a gated residual sum; and a head's norm with its rotation. Each had been a launch a step, every step reading
// and writing the whole activation: the elementwise kernels took 45 ms of a depth control pass's 162 ms at 1024 x 576.
func TestAMOSSBlocksElementwiseChainsLaunchOnce(t *testing.T) {
	t.Parallel()
	const width, tokens, head, heads, epsilon = 96, 24, 128, 3, 1e-6
	chains := map[string]struct {
		build    func(b *tensor.Builder) []*tensor.Tensor
		launches int
	}{
		"a block norm and a head norm": {func(b *tensor.Builder) []*tensor.Tensor {
			round := b.BF16RoundNative
			stream := b.Input("stream", dtype.F32, tensor.MustShape(width, tokens))
			weight, scale, shift := b.Input("weight", dtype.F32, tensor.MustShape(width)), b.Input("scale", dtype.F32, tensor.MustShape(width)), b.Input("shift", dtype.F32, tensor.MustShape(width))
			queries, queryWeight := b.Input("queries", dtype.F32, tensor.MustShape(head, heads, tokens)), b.Input("query_weight", dtype.F32, tensor.MustShape(head))
			normed := round(b.RMSNormNativeMean(stream, epsilon))
			modulated := round(b.Add(round(b.Multiply(round(b.Multiply(normed, weight)), scale)), shift))
			return []*tensor.Tensor{modulated, round(b.Multiply(round(b.RMSNormNativeMean(queries, epsilon)), queryWeight))}
		}, 2},
		"a gated residual sum": {func(b *tensor.Builder) []*tensor.Tensor {
			round := b.BF16RoundNative
			stream, update := b.Input("stream", dtype.F32, tensor.MustShape(width, tokens)), b.Input("update", dtype.F32, tensor.MustShape(width, tokens))
			gate := b.Input("gate", dtype.F32, tensor.MustShape(width))
			return []*tensor.Tensor{round(b.Add(stream, round(b.Multiply(update, gate))))}
		}, 1},
		"a head norm and its rotation": {func(b *tensor.Builder) []*tensor.Tensor {
			round := b.BF16RoundNative
			shape := tensor.MustShape(head, heads, tokens)
			query, weight := b.Input("query", dtype.F32, shape), b.Input("weight", dtype.F32, tensor.MustShape(head))
			cosine, sine := b.Input("cosine", dtype.F32, tensor.MustShape(head, 1, tokens)), b.Input("sine", dtype.F32, tensor.MustShape(head, 1, tokens))
			normed := round(b.Multiply(round(b.RMSNormNativeMean(query, epsilon)), weight))
			part := func(start uint64) *tensor.Tensor {
				return b.Reshape(b.GroupSlice(normed, start, head/2, tensor.SingletonExtent, head/2), head/2, heads, tokens)
			}
			turned := b.Concat(b.Scale(part(head/2), -tensor.UnitScale), part(0), tensor.FirstOffset)
			return []*tensor.Tensor{round(b.Add(round(b.Multiply(normed, cosine)), round(b.Multiply(turned, sine))))}
		}, 1},
	}
	for name, chain := range chains {
		compiled := testcheck.Must(Compile(chain.build(tensor.NewBuilder())...))(t)
		if len(compiled.launches) != chain.launches {
			t.Errorf("%s compiles to %d launches; want %d", name, len(compiled.launches), chain.launches)
		}
	}
}
