package thoughtbank

import (
	"fmt"
	"math"

	"overgo/internal/checked"
)

// Attention primitives for the DeepSeek-V4-derived hybrid attention: rotary
// position embedding, the attention-sink softmax, and the two KV compression
// schemes the even/odd layers alternate between. Ported from adaptive's
// go/extmodel/compressed_attention_ops.go at b8fef3cc2, re-expressed through
// this package's owners (dot, rmsNormNew).

// RotaryBase is the sinusoidal frequency base. 10000 is the value from the
// original rotary formulation and is a VENDOR fact of this checkpoint, not a
// derived quantity: changing it changes what the trained weights mean.
const RotaryBase = 10000.0

// rotaryCache holds cos/sin for positions 0..seqLen-1 over dim/2 frequencies.
type rotaryCache struct {
	Cos  []float32 // [seqLen, dim/2]
	Sin  []float32 // [seqLen, dim/2]
	Half int
}

// NewRotaryCache builds the table for one sequence length and head dimension.
// cos/sin are rounded to float32 so the decode's single-position rotary can
// reproduce the full-forward's arithmetic bit-for-bit.
func NewRotaryCache(seqLen, dim int) *rotaryCache {
	half := dim / 2
	c := &rotaryCache{Cos: make([]float32, seqLen*half), Sin: make([]float32, seqLen*half), Half: half}
	for i := 0; i < half; i++ {
		invFreq := 1.0 / math.Pow(RotaryBase, float64(i)/float64(half))
		for t := 0; t < seqLen; t++ {
			angle := float64(t) * invFreq
			c.Cos[t*half+i] = float32(math.Cos(angle))
			c.Sin[t*half+i] = float32(math.Sin(angle))
		}
	}
	return c
}

// ApplyRotaryInto rotates rows of x using the SPLIT-HALF pairing (coordinate i
// with coordinate i+half), not the interleaved one.
//
// x holds outer x seqLen x dim values; the result has the same shape.
func ApplyRotaryInto(out, x []float32, outer, seqLen, dim int, c *rotaryCache) {
	half := dim / 2
	for o := 0; o < outer; o++ {
		for t := 0; t < seqLen; t++ {
			base := (o*seqLen + t) * dim
			for i := 0; i < half; i++ {
				cs := float64(c.Cos[t*half+i])
				sn := float64(c.Sin[t*half+i])
				xe := float64(x[base+i])
				xo := float64(x[base+half+i])
				out[base+i] = float32(xe*cs - xo*sn)
				out[base+half+i] = float32(xe*sn + xo*cs)
			}
		}
	}
}

// AttentionSinkSoftmaxInto is softmax with a learnable per-head addend in the
// DENOMINATOR only, which lets a head attend to nothing by pushing mass into a
// term that has no value vector.
//
// The max subtraction applied to the logits is deliberately NOT applied to the
// sink term: the reference computes exp(logit-max) for entries but exp(sink)
// raw, so the sink's effective strength depends on the row maximum. A port that
// "corrects" this by subtracting the max from the sink too produces a different
// model.
//
// logits and out hold rows x heads x n values; sink holds one logit per head.
func AttentionSinkSoftmaxInto(out, logits, sink []float32, rows, heads, n int) {
	for r := 0; r < rows; r++ {
		for h := 0; h < heads; h++ {
			row := logits[(r*heads+h)*n : (r*heads+h+1)*n]
			dst := out[(r*heads+h)*n : (r*heads+h+1)*n]
			first, ok := checked.First(row)
			if !ok {
				continue
			}
			maxLogit := float64(first)
			for _, v := range row[1:] {
				if f := float64(v); f > maxLogit {
					maxLogit = f
				}
			}
			denom := math.Exp(float64(sink[h])) // deliberately un-shifted
			for i, v := range row {
				e := math.Exp(float64(v) - maxLogit)
				dst[i] = float32(e)
				denom += e
			}
			inv := 1.0 / denom
			for i := range dst {
				dst[i] = float32(float64(dst[i]) * inv)
			}
		}
	}
}

func attentionSinkSoftmaxVector(out, logits, sink []float32, heads, n int) {
	AttentionSinkSoftmaxInto(out, logits, sink, len(logits)/(heads*n), heads, n)
}

type compressionSeries struct {
	value, score, position []float32
	blockOffset            int
}

func compressKV(hPad []float32, series []compressionSeries, blocks, m, dModel, dHead int) ([]float32, error) {
	if !checked.PositiveInts(blocks, m, dModel, dHead) || len(hPad) != blocks*m*dModel || len(series) == 0 {
		return nil, fmt.Errorf("compress kv: input or series shape differs")
	}
	for _, source := range series {
		if len(source.value) != dHead*dModel || len(source.score) != dHead*dModel ||
			len(source.position) != m*dHead {
			return nil, fmt.Errorf("compress kv: projection shape differs")
		}
	}
	candidates := len(series) * m
	values, scores := make([]float64, candidates*dHead), make([]float64, candidates*dHead)
	out := make([]float32, blocks*dHead)
	negativeInfinity := math.Inf(-len(series))
	for block := 0; block < blocks; block++ {
		for sourceIndex, source := range series {
			sourceBlock := block + source.blockOffset
			for position := 0; position < m; position++ {
				candidate := sourceIndex*m + position
				if sourceBlock < 0 {
					for feature := 0; feature < dHead; feature++ {
						scores[candidate*dHead+feature] = negativeInfinity
					}
					continue
				}
				token := hPad[(sourceBlock*m+position)*dModel : (sourceBlock*m+position+1)*dModel]
				for feature := 0; feature < dHead; feature++ {
					values[candidate*dHead+feature] = dot(source.value[feature*dModel:(feature+1)*dModel], token)
					scores[candidate*dHead+feature] = dot(source.score[feature*dModel:(feature+1)*dModel], token) +
						float64(source.position[position*dHead+feature])
				}
			}
		}
		for feature := 0; feature < dHead; feature++ {
			maxScore := scores[feature]
			for candidate := range candidates {
				maxScore = max(maxScore, scores[candidate*dHead+feature])
			}
			var denominator, numerator float64
			for candidate := 0; candidate < candidates; candidate++ {
				weight := math.Exp(scores[candidate*dHead+feature] - maxScore)
				denominator += weight
				numerator += weight * values[candidate*dHead+feature]
			}
			out[block*dHead+feature] = float32(numerator / denominator)
		}
	}
	return out, nil
}

// CompressKVHeavy collapses each non-overlapping block through one learned series.
func CompressKVHeavy(hPad, value, score, position []float32, blocks, m, dModel, dHead int) ([]float32, error) {
	return compressKV(hPad, []compressionSeries{{value: value, score: score, position: position}},
		blocks, m, dModel, dHead)
}

// CompressKVSparse combines the current block with a second series over its predecessor.
func CompressKVSparse(hPad, valueA, valueB, scoreA, scoreB, positionA, positionB []float32,
	blocks, m, dModel, dHead int,
) ([]float32, error) {
	return compressKV(hPad, []compressionSeries{
		{value: valueA, score: scoreA, position: positionA},
		{value: valueB, score: scoreB, position: positionB, blockOffset: -1},
	}, blocks, m, dModel, dHead)
}
