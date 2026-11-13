package model

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"overgo/internal/sampling"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestABlocksNAGIsThePapersEquations holds a conditioned diffusion block's NAG (guidance-ops/nag-wan) to Eqs. 7 to 10 of
// its paper, written out here on the block's own cross-attention outputs for the prompt alone and the negative prompt
// alone, the block evaluated by the reference backend on small weights: the guided output is the equations', and it is
// what the output projection takes, both with the paper's Wan bound, which tokens' ratios pass, and with a bound no
// token reaches; and a program with NAG given no negative context builds the block it built before, bit for bit.
func TestABlocksNAGIsThePapersEquations(t *testing.T) {
	const dim, heads, ffn, tokens, contextTokens = 12, 2, 20, 4, 3
	builder := tensor.NewBuilder()
	feeds := map[*tensor.Tensor]reference.Value{}
	fed := func(offset float64, dimensions ...uint64) *tensor.Tensor {
		shape := tensor.MustShape(dimensions...)
		data := make([]float32, testcheck.Must(shape.Elements())(t))
		for index := range data {
			data[index] = float32(offset + 0.3*math.Sin(float64(len(feeds))*1.7+float64(index)*0.61))
		}
		node := builder.Input(fmt.Sprintf("input_%d", len(feeds)), dtype.F32, shape)
		feeds[node] = reference.Value{Shape: shape, Data: data}
		return node
	}
	attention := func() ConditionedDiffusionAttentionWeights {
		return ConditionedDiffusionAttentionWeights{
			Query: fed(0, dim, dim), QueryBias: fed(0, dim), Key: fed(0, dim, dim), KeyBias: fed(0, dim),
			Value: fed(0, dim, dim), ValueBias: fed(0, dim), Output: fed(0, dim, dim), OutputBias: fed(0, dim),
			QueryNorm: fed(1, dim), KeyNorm: fed(1, dim),
		}
	}
	weights := ConditionedDiffusionBlockWeights{
		Modulation: fed(0, 6*dim), SelfAttention: attention(), CrossAttention: attention(),
		CrossNormWeight: fed(1, dim), CrossNormBias: fed(0, dim),
		FFNExpand: fed(0, dim, ffn), FFNExpandBias: fed(0, ffn), FFNContract: fed(0, ffn, dim), FFNContractBias: fed(0, dim),
	}
	input, conditioning := fed(0, dim, tokens), fed(0, 6*dim)
	options := ConditionedDiffusionBlockOptions{
		Dim: dim, Heads: heads, FFNDim: ffn, Epsilon: 1e-6, RotaryBase: 10000, AxisChannels: ThreeAxisRotaryChannels(dim / heads),
		AxisPositions: [3][]uint32{{0, 0, 1, 1}, {0, 0, 0, 0}, {0, 1, 0, 1}},
	}
	program := func(guidance sampling.NormalizedAttention) conditionedDiffusionProgram {
		options.Attention = guidance
		return testcheck.Must(CompileConditionedDiffusionProgram(options))(t)
	}
	block := func(result ConditionedDiffusionBlockResult, err error) ConditionedDiffusionBlockResult {
		t.Helper()
		testcheck.NoError(t, err)
		return result
	}
	cross := func(context *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
		key, value, err := program(sampling.NormalizedAttention{}).BuildCrossContext(builder, context, weights.CrossAttention)
		testcheck.NoError(t, err)
		return key, value
	}
	key, value := cross(fed(0, dim, contextTokens))
	negativeKey, negativeValue := cross(fed(0, dim, contextTokens))
	plain := program(sampling.NormalizedAttention{})
	if _, err := plain.BuildGuidedBlock(builder, input, conditioning, key, value, negativeKey, negativeValue, nil, nil, weights); err == nil {
		t.Fatal("a program without NAG built a guided block")
	}
	// The paper's Table 5 gives Wan2.1 phi 4, tau 2.5 and alpha 0.25.
	wan := sampling.NormalizedAttention{Scale: 4, Tau: 2.5, Alpha: 0.25}
	unbounded := sampling.NormalizedAttention{Scale: 4, Tau: 1e6, Alpha: 0.25}
	unguided := block(plain.BuildBlock(builder, input, conditioning, key, value, weights))
	unguidedUnderNAG := block(program(wan).BuildBlock(builder, input, conditioning, key, value, weights))
	negativeAlone := block(plain.BuildBlock(builder, input, conditioning, negativeKey, negativeValue, weights))
	guided := map[sampling.NormalizedAttention]ConditionedDiffusionBlockResult{}
	outputs := []*tensor.Tensor{unguided.Output, unguidedUnderNAG.Output, unguided.CrossAttention, negativeAlone.CrossAttention}
	for _, guidance := range []sampling.NormalizedAttention{wan, unbounded} {
		guided[guidance] = block(program(guidance).BuildGuidedBlock(builder, input, conditioning, key, value, negativeKey, negativeValue, nil, nil, weights))
		outputs = append(outputs, guided[guidance].CrossAttention, guided[guidance].CrossProjected, guided[guidance].Output)
	}
	testcheck.NoError(t, builder.Err())
	values := testcheck.Must(reference.Execute(outputs, feeds))(t)
	if !slices.Equal(values[unguided.Output].Data, values[unguidedUnderNAG.Output].Data) {
		t.Fatal("a program with NAG given no negative context builds another block than one without NAG")
	}
	positive, negative := values[unguided.CrossAttention].Data, values[negativeAlone.CrossAttention].Data
	for _, guidance := range []sampling.NormalizedAttention{wan, unbounded} {
		want, bounded := normalizedAttention(positive, negative, dim, guidance)
		if guidance == wan && bounded == 0 || guidance == unbounded && bounded != 0 {
			t.Fatalf("with tau %g, %d of %d tokens' ratios pass the bound; the fixture must exercise %v", guidance.Tau, bounded, tokens, guidance)
		}
		got := values[guided[guidance].CrossAttention].Data
		for index := range want {
			if math.Abs(float64(got[index])-want[index]) > 1e-5*(1+math.Abs(want[index])) {
				t.Fatalf("with tau %g, element %d of the guided cross-attention is %g, Eqs. 7 to 10 give %g", guidance.Tau, index, got[index], want[index])
			}
		}
		// The guided output is what the output projection takes: W_o Z + b_o, a column of W_o an output's weights.
		projection, bias := feeds[weights.CrossAttention.Output].Data, feeds[weights.CrossAttention.OutputBias].Data
		projected := values[guided[guidance].CrossProjected].Data
		for token := range tokens {
			for output := range dim {
				sum := float64(bias[output])
				for feature := range dim {
					sum += float64(projection[output*dim+feature]) * float64(got[token*dim+feature])
				}
				if math.Abs(float64(projected[token*dim+output])-sum) > 1e-5*(1+math.Abs(sum)) {
					t.Fatalf("with tau %g, token %d's projected output %d is %g, the projection of the guided output %g", guidance.Tau, token, output, projected[token*dim+output], sum)
				}
			}
		}
		if slices.Equal(values[guided[guidance].Output].Data, values[unguided.Output].Data) {
			t.Fatalf("with tau %g, the guided block's output is the unguided block's", guidance.Tau)
		}
	}
}

// normalizedAttention is NAG's Eqs. 7 to 10 on a cross-attention's outputs for the prompt and the negative prompt, a
// token's dim features after another's, in float64: Z~ = Z+ + phi (Z+ - Z-); R = |Z~|_1 / |Z+|_1 over the token's
// features; Z^ = min(R, tau) / R Z~; Z = alpha Z^ + (1 - alpha) Z+. It gives the guided output and how many tokens'
// ratios pass tau.
func normalizedAttention(positive, negative []float32, dim int, guidance sampling.NormalizedAttention) ([]float64, int) {
	guided, bounded := make([]float64, 0, len(positive)), 0
	for start := 0; start < len(positive); start += dim {
		extrapolated := make([]float64, dim)
		var positiveNorm, extrapolatedNorm float64
		for feature := range dim {
			plus, minus := float64(positive[start+feature]), float64(negative[start+feature])
			extrapolated[feature] = plus + guidance.Scale*(plus-minus)
			positiveNorm += math.Abs(plus)
			extrapolatedNorm += math.Abs(extrapolated[feature])
		}
		ratio := extrapolatedNorm / positiveNorm
		if ratio > guidance.Tau {
			bounded++
		}
		for feature := range dim {
			guided = append(guided, guidance.Alpha*min(ratio, guidance.Tau)/ratio*extrapolated[feature]+(1-guidance.Alpha)*float64(positive[start+feature]))
		}
	}
	return guided, bounded
}
