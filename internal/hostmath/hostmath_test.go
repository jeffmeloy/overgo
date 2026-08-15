package hostmath

import (
	"math"
	"slices"
	"testing"
)

func TestTranspose2D(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6}
	transposed := Transpose2D(input, 2, 3)
	if want := []float32{1, 4, 2, 5, 3, 6}; !slices.Equal(transposed, want) {
		t.Fatalf("transpose = %v, want %v", transposed, want)
	}
	if roundTrip := Transpose2D(transposed, 3, 2); !slices.Equal(roundTrip, input) {
		t.Fatalf("round trip = %v, want %v", roundTrip, input)
	}
}

func TestGradientSlot(t *testing.T) {
	gradients := map[string][]float32{}
	slot := GradientSlot(gradients, "weight", 3)
	slot[1] = 2
	if reused := GradientSlot(gradients, "weight", 3); !slices.Equal(reused, []float32{0, 2, 0}) {
		t.Fatalf("reused slot = %v", reused)
	}
}

func TestChannelMixF64Into(t *testing.T) {
	const channels, positions = 2, 2
	input := []float32{1, 2, 3, 4}
	identity := []float32{1, 0, 0, 1}
	bias := []float32{5, 6}
	output := make([]float32, channels*positions)
	if err := ChannelMixF64Into(output, input, identity, bias, channels, channels, positions); err != nil {
		t.Fatal(err)
	}
	if want := []float32{6, 7, 9, 10}; !slices.Equal(output, want) {
		t.Fatalf("channel mix = %v, want %v", output, want)
	}
	if err := ChannelMixF64Into(output[:len(output)-1], input, identity, bias, channels, channels, positions); err == nil {
		t.Fatal("channel mix accepted a short destination")
	}
}

func TestLinearBF16MatchesPromotedLinear(t *testing.T) {
	x := []float32{0.25, -2, 3.5, 1}
	words := []uint16{0x3f80, 0xc000, 0x3f00, 0x4040}
	weights := make([]float32, len(words))
	for index, word := range words {
		weights[index] = math.Float32frombits(uint32(word) << 16)
	}
	want, got := make([]float32, 4), make([]float32, 4)
	Linear(want, x, weights, 2, 2, 2)
	LinearBF16(got, x, words, 2, 2, 2)
	if !slices.Equal(got, want) {
		t.Fatalf("native BF16 linear = %v, promoted = %v", got, want)
	}
}

func TestLinearBF16BackwardInput(t *testing.T) {
	words := []uint16{0x3f80, 0xc000, 0x3f00, 0x4040}
	dy := []float32{2, -1}
	dx := make([]float32, 2)
	LinearBF16BackwardInput(dx, dy, words, 1, 2, 2)
	if want := []float32{1.5, -7}; !slices.Equal(dx, want) {
		t.Fatalf("input gradient = %v, want %v", dx, want)
	}
}

func TestLinearBackwardWeightOnly(t *testing.T) {
	x := []float32{1, 2, -1, 3}
	dy := []float32{2, -1, 4, 0.5}
	dW := make([]float32, 4)
	LinearBackward(nil, dW, nil, x, nil, dy, 2, 2, 2, false)
	if want := []float32{-2, 16, -1.5, -0.5}; !slices.Equal(dW, want) {
		t.Fatalf("weight-only gradient = %v, want %v", dW, want)
	}
}

// TestGELUTanh pins the tanh approximation against PyTorch gelu(approximate="tanh").
func TestGELUTanh(t *testing.T) {
	v := []float32{0, 1, -1, 2, -2, 0.5}
	want := []float64{0, 0.8411919906082768, -0.15880801, 1.9545976, -0.04540229, 0.34571400982514394}
	GELUTanhInPlace(v)
	for i := range v {
		if diff := math.Abs(float64(v[i]) - want[i]); diff > 1e-6 {
			t.Fatalf("gelu[%d] = %g, want %g", i, v[i], want[i])
		}
	}
}

// TestLayerNormInto checks mean/biased-variance normalization, the affine
// path, the no-affine path, aliasing, and the affine width panic.
func TestLayerNormInto(t *testing.T) {
	const d, eps = 4, 1e-6
	x := []float32{1, 2, 3, 4, -2, 0, 2, 8}
	weight := []float32{1.5, -0.5, 2, 1}
	bias := []float32{0.1, -0.2, 0.3, 0}
	out := make([]float32, len(x))
	LayerNormInto(out, x, weight, bias, 2, d, eps)
	for r := 0; r < 2; r++ {
		row := x[r*d : (r+1)*d]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= d
		var variance float64
		for _, v := range row {
			dv := float64(v) - mean
			variance += dv * dv
		}
		variance /= d
		inv := 1 / math.Sqrt(variance+eps)
		for j := 0; j < d; j++ {
			want := (float64(row[j])-mean)*inv*float64(weight[j]) + float64(bias[j])
			if diff := math.Abs(float64(out[r*d+j]) - want); diff > 1e-7 {
				t.Fatalf("ln[%d][%d] = %g, want %g", r, j, out[r*d+j], want)
			}
		}
	}
	plain := make([]float32, d)
	LayerNormInto(plain, x[:d], nil, nil, 1, d, eps)
	var sum, ss float64
	for _, v := range plain {
		sum += float64(v)
		ss += float64(v) * float64(v)
	}
	if math.Abs(sum) > 1e-6 || math.Abs(ss/d-1) > 1e-5 {
		t.Fatalf("no-affine row not standardized: sum=%g meanSq=%g", sum, ss/d)
	}
	aliased := append([]float32(nil), x[:d]...)
	LayerNormInto(aliased, aliased, nil, nil, 1, d, eps)
	for j := range plain {
		if aliased[j] != plain[j] {
			t.Fatalf("aliased LayerNorm differs at %d", j)
		}
	}
	defer func() {
		if recover() == nil {
			t.Fatal("want panic on mismatched affine width")
		}
	}()
	LayerNormInto(out, x, weight[:2], bias, 2, d, eps)
}

// TestGELUErf pins the exact-erf GELU against closed-form Phi values.
func TestGELUErf(t *testing.T) {
	v := []float32{0, 1, -1, 2, 0.5}
	want := []float64{0, 0.8413447460685429, -0.15865525393145707, 1.9544997361036416, 0.34573123063700656}
	GELUErfInPlace(v)
	for i := range v {
		if diff := math.Abs(float64(v[i]) - want[i]); diff > 1e-6 {
			t.Fatalf("geluErf[%d] = %g, want %g", i, v[i], want[i])
		}
	}
	if g := GELUErf(10); math.Abs(g-10) > 1e-9 {
		t.Fatalf("geluErf(10) = %g, want ~10", g)
	}
}

// bruteBidirectional: direct per-query softmax over an explicit key subset.
func bruteBidirectional(q, k, v []float32, querySeq, keySeq, heads, headDim int, allowed func(int) bool) []float32 {
	out := make([]float32, querySeq*heads*headDim)
	for h := 0; h < heads; h++ {
		for qi := 0; qi < querySeq; qi++ {
			var keys []int
			var scores []float64
			for ki := 0; ki < keySeq; ki++ {
				if !allowed(ki) {
					continue
				}
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(q[(qi*heads+h)*headDim+d]) * float64(k[(ki*heads+h)*headDim+d])
				}
				keys = append(keys, ki)
				scores = append(scores, dot)
			}
			mx := math.Inf(-1)
			for _, s := range scores {
				mx = math.Max(mx, s)
			}
			var sum float64
			for i, s := range scores {
				scores[i] = math.Exp(s - mx)
				sum += scores[i]
			}
			for i, ki := range keys {
				w := scores[i] / sum
				for d := 0; d < headDim; d++ {
					out[(qi*heads+h)*headDim+d] += float32(w * float64(v[(ki*heads+h)*headDim+d]))
				}
			}
		}
	}
	return out
}

func TestMaskedBidirectionalAttention(t *testing.T) {
	const querySeq, keySeq, heads, headDim = 3, 5, 2, 4
	fill := func(n int, seed float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(math.Sin(seed + float64(i)*0.7))
		}
		return out
	}
	q := fill(querySeq*heads*headDim, 0.1)
	k := fill(keySeq*heads*headDim, 0.5)
	v := fill(keySeq*heads*headDim, 0.9)

	got := make([]float32, querySeq*heads*headDim)
	MaskedBidirectionalAttention(got, q, k, v, querySeq, keySeq, heads, heads, headDim, nil)
	want := bruteBidirectional(q, k, v, querySeq, keySeq, heads, headDim, func(int) bool { return true })
	for i := range got {
		if diff := math.Abs(float64(got[i]) - float64(want[i])); diff > 1e-6 {
			t.Fatalf("unmasked[%d] = %g, want %g", i, got[i], want[i])
		}
	}

	mask := []bool{true, true, false, true, false}
	MaskedBidirectionalAttention(got, q, k, v, querySeq, keySeq, heads, heads, headDim, mask)
	want = bruteBidirectional(q, k, v, querySeq, keySeq, heads, headDim, func(ki int) bool { return mask[ki] })
	for i := range got {
		if diff := math.Abs(float64(got[i]) - float64(want[i])); diff > 1e-6 {
			t.Fatalf("masked[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

// TestCausalAttentionStepMatchesFull: stepping a query at each position over
// the accumulated cache must be BIT-identical to the full-sequence core's
// row — the contract incremental decode relies on.
func TestCausalAttentionStepMatchesFull(t *testing.T) {
	const seq, heads, kvHeads, headDim = 6, 4, 2, 8
	fill := func(n int, seed float64) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(math.Sin(seed + float64(i)*0.7))
		}
		return out
	}
	q := fill(seq*heads*headDim, 0.2)
	k := fill(seq*kvHeads*headDim, 0.6)
	v := fill(seq*kvHeads*headDim, 1.1)

	full := make([]float32, seq*heads*headDim)
	CausalAttention(full, q, k, v, seq, heads, kvHeads, headDim)

	step := make([]float32, heads*headDim)
	for pos := 0; pos < seq; pos++ {
		cached := (pos + 1) * kvHeads * headDim
		CausalAttentionStep(step, q[pos*heads*headDim:(pos+1)*heads*headDim], k[:cached], v[:cached], pos+1, heads, kvHeads, headDim)
		for i, value := range step {
			if value != full[pos*heads*headDim+i] {
				t.Fatalf("pos %d element %d: step %g != full %g", pos, i, value, full[pos*heads*headDim+i])
			}
		}
	}
}
