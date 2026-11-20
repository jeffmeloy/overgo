//go:build windows

package executor

import (
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestANormAndItsStepsFoldIntoOneKernel holds MOSS's rounded native-mean RMS norm with its rounded weight, modulation
// scale and shift after it, built as the denoiser builds them (latentimage rotaryGraph.norm and modulate), to one
// launch, and the norm with its weight alone over a head's channels to one more, each changing no bit against the same
// chain launched step by step, which a graph that also keeps each chain's rounded norm as an output does. The block
// stream's width is under 128 values and the head's 128, the norm's two reductions.
func TestANormAndItsStepsFoldIntoOneKernel(t *testing.T) {
	// Serial: device fixtures share the one card.
	const width, tokens, head, heads, epsilon = 96, 24, 128, 3, 1e-6
	build := func(keep bool) (*CompiledGraph, []*tensor.Tensor, map[*tensor.Tensor]reference.Value) {
		b := tensor.NewBuilder()
		round := b.BF16RoundNative
		stream := b.Input("stream", dtype.F32, tensor.MustShape(width, tokens))
		weight := b.Input("weight", dtype.F32, tensor.MustShape(width))
		scale := b.Input("scale", dtype.F32, tensor.MustShape(width))
		shift := b.Input("shift", dtype.F32, tensor.MustShape(width))
		queries := b.Input("queries", dtype.F32, tensor.MustShape(head, heads, tokens))
		queryWeight := b.Input("query_weight", dtype.F32, tensor.MustShape(head))
		normed := round(b.RMSNormNativeMean(stream, epsilon))
		modulated := round(b.Add(round(b.Multiply(round(b.Multiply(normed, weight)), scale)), shift))
		queryNormed := round(b.RMSNormNativeMean(queries, epsilon))
		query := round(b.Multiply(queryNormed, queryWeight))
		outputs := []*tensor.Tensor{modulated, query}
		if keep {
			outputs = append(outputs, normed, queryNormed)
		}
		return testcheck.Must(Compile(outputs...))(t), []*tensor.Tensor{modulated, query}, map[*tensor.Tensor]reference.Value{
			stream: patternedValue(stream.Shape, 1, 2.3, 0.1), weight: patternedValue(weight.Shape, 2, 0.9, 1),
			scale: patternedValue(scale.Shape, 3, 0.4, 1), shift: patternedValue(shift.Shape, 4, 0.7, -0.2),
			queries: patternedValue(queries.Shape, 5, 1.7, -0.3), queryWeight: patternedValue(queryWeight.Shape, 6, 0.5, 1),
		}
	}
	cuda := newFixtureExecutor(t)
	run := func(keep bool) [][]float32 {
		compiled, results, feeds := build(keep)
		folded := 0
		for _, frame := range compiled.nodes {
			if frame.fusion != nil && frame.fusion.kind == compiledFusionMOSSNorm {
				folded++
			}
		}
		if want := map[bool]int{false: 2, true: 0}[keep]; folded != want || !keep && len(compiled.launches) != 2 {
			t.Fatalf("keeping the norms %t, %d norms fold in %d launches, want %d", keep, folded, len(compiled.launches), want)
		}
		got := testcheck.Must(cuda.ExecuteCompiled(t.Context(), compiled, feeds, nil))(t)
		return [][]float32{got[results[0]].Data, got[results[1]].Data}
	}
	folded, unfolded := run(false), run(true)
	for index := range folded {
		compareBits(t, folded[index], unfolded[index])
	}
}

// TestAGatedAddFoldsIntoOneKernel holds a rounded residual sum of a stream and a rounded product of an update and a
// per-channel gate, as the denoiser adds a block's projections to its stream (latentimage rotaryGraph.joint), to one
// launch, changing no bit against the same sum launched in two, which a graph that also keeps the product does.
func TestAGatedAddFoldsIntoOneKernel(t *testing.T) {
	// Serial: device fixtures share the one card.
	const width, tokens = 96, 24
	build := func(keep bool) (*CompiledGraph, *tensor.Tensor, map[*tensor.Tensor]reference.Value) {
		b := tensor.NewBuilder()
		round := b.BF16RoundNative
		stream := b.Input("stream", dtype.F32, tensor.MustShape(width, tokens))
		update := b.Input("update", dtype.F32, tensor.MustShape(width, tokens))
		gate := b.Input("gate", dtype.F32, tensor.MustShape(width))
		gated := round(b.Multiply(update, gate))
		sum := round(b.Add(stream, gated))
		outputs := []*tensor.Tensor{sum}
		if keep {
			outputs = append(outputs, gated)
		}
		return testcheck.Must(Compile(outputs...))(t), sum, map[*tensor.Tensor]reference.Value{
			stream: patternedValue(stream.Shape, 1, 2.3, 0.1), update: patternedValue(update.Shape, 2, 1.9, -0.4),
			gate: patternedValue(gate.Shape, 3, 0.6, 0.2),
		}
	}
	cuda := newFixtureExecutor(t)
	run := func(keep bool) []float32 {
		compiled, sum, feeds := build(keep)
		folded := 0
		for _, frame := range compiled.nodes {
			if frame.fusion != nil && frame.fusion.kind == compiledFusionGatedAdd {
				folded++
			}
		}
		if want := map[bool]int{false: 1, true: 0}[keep]; folded != want || !keep && len(compiled.launches) != 1 {
			t.Fatalf("keeping the product %t, %d sums fold in %d launches, want %d", keep, folded, len(compiled.launches), want)
		}
		return testcheck.Must(cuda.ExecuteCompiled(t.Context(), compiled, feeds, nil))(t)[sum].Data
	}
	compareBits(t, run(false), run(true))
}

// TestAQueryNormAndItsRotationFoldIntoOneKernel holds MOSS's q norm with its weight and the rotary embedding of the
// normalized heads, built as the denoiser builds them (latentimage rotaryGraph.norm and rotate), to one launch that
// never stores the normalized heads, changing no bit against the norm and the rotation launched apart, which a graph
// that also keeps the normalized heads as an output does.
func TestAQueryNormAndItsRotationFoldIntoOneKernel(t *testing.T) {
	// Serial: device fixtures share the one card.
	const width, heads, tokens, epsilon = 128, 3, 12, 1e-6
	build := func(keep bool) (*CompiledGraph, *tensor.Tensor, map[*tensor.Tensor]reference.Value) {
		b := tensor.NewBuilder()
		round := b.BF16RoundNative
		shape := tensor.MustShape(width, heads, tokens)
		query := b.Input("query", dtype.F32, shape)
		weight := b.Input("weight", dtype.F32, tensor.MustShape(width))
		cosine := b.Input("cosine", dtype.F32, tensor.MustShape(width, 1, tokens))
		sine := b.Input("sine", dtype.F32, tensor.MustShape(width, 1, tokens))
		normed := round(b.Multiply(round(b.RMSNormNativeMean(query, epsilon)), weight))
		part := func(start uint64) *tensor.Tensor {
			return b.Reshape(b.GroupSlice(normed, start, width/2, tensor.SingletonExtent, width/2), width/2, heads, tokens)
		}
		turned := b.Concat(b.Scale(part(width/2), -tensor.UnitScale), part(0), tensor.FirstOffset)
		rotated := round(b.Add(round(b.Multiply(normed, cosine)), round(b.Multiply(turned, sine))))
		outputs := []*tensor.Tensor{rotated}
		if keep {
			outputs = append(outputs, normed)
		}
		return testcheck.Must(Compile(outputs...))(t), rotated, map[*tensor.Tensor]reference.Value{
			query: patternedValue(shape, 1, 1.7, -0.3), weight: patternedValue(weight.Shape, 2, 0.5, 1),
			cosine: patternedValue(cosine.Shape, 3, 0.11, 0.02), sine: patternedValue(sine.Shape, 4, 0.09, -0.01),
		}
	}
	cuda := newFixtureExecutor(t)
	run := func(keep bool) []float32 {
		compiled, rotated, feeds := build(keep)
		folded := 0
		for _, frame := range compiled.nodes {
			if frame.fusion != nil && frame.fusion.kind == compiledFusionMOSSNormRotary {
				folded++
			}
		}
		if want := map[bool]int{false: 1, true: 0}[keep]; folded != want || !keep && len(compiled.launches) != 1 {
			t.Fatalf("keeping the normalized heads %t, %d rotations fold their norms in %d launches, want %d", keep, folded, len(compiled.launches), want)
		}
		return testcheck.Must(cuda.ExecuteCompiled(t.Context(), compiled, feeds, nil))(t)[rotated].Data
	}
	compareBits(t, run(false), run(true))
}
