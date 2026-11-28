package hostexpert

import (
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestASplitRowStandsNearTheValues holds the host's quantized activations to the values they stand for
// (flashnext-projection-exact): a row's product with a token's activations quantized as Q8_K and split in two
// (quantizeSplitQ8K), the values' and what their rounding left, stands at least a hundred times nearer the product
// with the f64 values than the product with the first half alone, ggml's CPU arithmetic, at Flash-Next's hidden width
// in each of its gate formats; and so does an intermediate's split as Q8_0 in each down format.
func TestASplitRowStandsNearTheValues(t *testing.T) {
	t.Parallel()
	draws := rand.New(rand.NewPCG(2560, 640))
	for _, formats := range testFormats {
		experts := testExperts(t, formats, []int{1})
		for _, position := range []struct {
			name           string
			format         dtype.Type
			weights        []byte
			depth, rows    int
			rowsOf         rowFunction
			split          func([]float32, []byte) error
			quantizedBytes int
		}{
			{"gate", formats.Gate, experts[0].Gate, testHidden, testIntermediate, rowsOf[formats.Gate], quantizeSplitQ8K, Q8KBytes(testHidden)},
			{"down", formats.Down, experts[0].Down, testIntermediate, testHidden, rowsOf[formats.Down], quantizeSplitQ80, splitParts * testIntermediate / blockValues * q80BlockBytes},
		} {
			values := make([]float32, position.depth)
			for index := range values {
				values[index] = float32(draws.NormFloat64())
			}
			quantized := make([]byte, position.quantizedBytes)
			testcheck.NoError(t, position.split(values, quantized))
			half := len(quantized) / splitParts
			out := [][]float32{make([]float32, position.rows), make([]float32, position.rows)}
			testcheck.NoError(t, position.rowsOf(position.weights, position.depth, [][]byte{quantized[:half], quantized[half:]}, out, 0, position.rows))
			weights := testcheck.Must(quant.Dequantize(position.format, position.weights, uint64(position.depth*position.rows)))(t)
			exact := make([]float64, position.rows)
			for row := range position.rows {
				for index, value := range values {
					exact[row] += float64(weights[row*position.depth+index]) * float64(value)
				}
			}
			var alone, split float64
			for row := range position.rows {
				alone = max(alone, math.Abs(float64(out[0][row])-exact[row]))
				split = max(split, math.Abs(float64(out[0][row]+out[1][row])-exact[row]))
			}
			t.Logf("%v %s: the split row stands %.3g from the f64 product, the first half alone %.3g", formats, position.name, split, alone)
			if split*100 > alone {
				t.Errorf("%v %s: the split row stands %.3g from the f64 product, the first half alone %.3g; want a hundred times nearer", formats, position.name, split, alone)
			}
		}
	}
}
