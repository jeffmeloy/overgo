package latentimage

import (
	"math"
	"testing"
)

// attentionAgreement reads how near an attention output lies to a reference's: the relative L2 of their difference and
// the fraction of its entries whose bits are the reference's.
func attentionAgreement(got, want []float32) (relative, equal float64) {
	var differenceSquares, wantSquares float64
	same := 0
	for index, value := range want {
		difference := float64(got[index]) - float64(value)
		differenceSquares, wantSquares = differenceSquares+difference*difference, wantSquares+float64(value)*float64(value)
		if math.Float32bits(got[index]) == math.Float32bits(value) {
			same++
		}
	}
	return math.Sqrt(differenceSquares / wantSquares), float64(same) / float64(len(want))
}

// blockUpdateAgreement reads how near a block's update of its stream, its output got less its input, lies to a
// reference's, want less the same input, over tokens of width values: the relative L2 of the whole difference, and the
// lowest cosine of a token's update with the reference's and that token. A token whose update is zero where the
// reference's is not, a residual skipped, has no cosine and is held at 0, so it fails where it lies and no later
// token's cosine stands in for it (SQA-443); two zero updates agree.
func blockUpdateAgreement(input, got, want []float32, width int) (relative, lowest float64, lowestToken int) {
	var differenceSquares, wantSquares float64
	lowest = 1
	for token := range len(input) / width {
		var dot, gotSquares, referenceSquares float64
		for column := token * width; column < (token+1)*width; column++ {
			g, r := float64(got[column]-input[column]), float64(want[column]-input[column])
			dot += g * r
			gotSquares += g * g
			referenceSquares += r * r
			differenceSquares += (g - r) * (g - r)
		}
		wantSquares += referenceSquares
		cosine := 1.0
		if gotSquares != 0 || referenceSquares != 0 {
			cosine = 0
			if gotSquares != 0 && referenceSquares != 0 {
				cosine = dot / math.Sqrt(gotSquares*referenceSquares)
			}
		}
		if cosine < lowest {
			lowest, lowestToken = cosine, token
		}
	}
	return math.Sqrt(differenceSquares / wantSquares), lowest, lowestToken
}

// TestABlockWhoseTokenSkipsItsUpdateFailsAtThatToken holds the block check's per-token reading (blockUpdateAgreement,
// SQA-443) to the review's counterexample: of three tokens whose reference updates are nonzero, the first's update
// skipped (its output its input) reads cosine 0 at token 0, though the whole difference's relative L2 stays small; the
// cosine of a zero update had been 0/0, a NaN the next token's finite cosine replaced, and the check passed.
func TestABlockWhoseTokenSkipsItsUpdateFailsAtThatToken(t *testing.T) {
	t.Parallel()
	const width, tokens = 4, 3
	input := make([]float32, width*tokens)
	want := make([]float32, width*tokens)
	for index := range input {
		input[index] = float32(index%5) - 2
		want[index] = input[index] + 0.25 + float32(index%3)*0.125
	}
	got := append([]float32(nil), want...)
	copy(got[:width], input[:width])
	relative, lowest, token := blockUpdateAgreement(input, got, want, width)
	if lowest != 0 || token != 0 {
		t.Errorf("a token that skipped its update reads lowest cosine %g at token %d; want 0 at token 0", lowest, token)
	}
	if relative >= 1 {
		t.Errorf("one token of three skipped reads relative L2 %g; want under 1, so the cosine alone fails it", relative)
	}
	if same, lowest, _ := blockUpdateAgreement(input, want, want, width); same != 0 || lowest != 1 {
		t.Errorf("a block equal to its reference reads relative L2 %g and lowest cosine %g; want 0 and 1", same, lowest)
	}
}
