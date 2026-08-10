package thoughtbank

import (
	"fmt"
	"math"
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
			maxLogit := math.Inf(-1)
			for _, v := range row {
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

// CompressKVHeavy is the HCA (heavily compressed) scheme: non-overlapping blocks
// of m tokens, each collapsed by a softmax over the block computed from a gate
// projection plus a learned per-position bias.
//
// The gate carries the positional bias; the value projection does NOT --
// position influences HOW the block is summarised, never WHAT is summarised.
//
// hPad holds blocks*m x dModel values (already padded to a block multiple); the
// result holds blocks x dHead.
func CompressKVHeavy(hPad, wKV, wZ, pos []float32, blocks, m, dModel, dHead int) ([]float32, error) {
	if len(hPad) != blocks*m*dModel {
		return nil, fmt.Errorf("hca compress: input has %d values, want %d", len(hPad), blocks*m*dModel)
	}
	if len(pos) != m*dHead {
		return nil, fmt.Errorf("hca compress: pos has %d values, want %d", len(pos), m*dHead)
	}
	out := make([]float32, blocks*dHead)
	c := make([]float64, m*dHead)
	z := make([]float64, m*dHead)
	for b := 0; b < blocks; b++ {
		for j := 0; j < m; j++ {
			tok := hPad[(b*m+j)*dModel : (b*m+j+1)*dModel]
			for e := 0; e < dHead; e++ {
				c[j*dHead+e] = dot(wKV[e*dModel:(e+1)*dModel], tok)
				z[j*dHead+e] = dot(wZ[e*dModel:(e+1)*dModel], tok) + float64(pos[j*dHead+e])
			}
		}
		// Softmax runs over the BLOCK axis independently per feature dimension.
		for e := 0; e < dHead; e++ {
			maxZ := math.Inf(-1)
			for j := 0; j < m; j++ {
				if z[j*dHead+e] > maxZ {
					maxZ = z[j*dHead+e]
				}
			}
			var sum, acc float64
			for j := 0; j < m; j++ {
				w := math.Exp(z[j*dHead+e] - maxZ)
				sum += w
				acc += w * c[j*dHead+e]
			}
			out[b*dHead+e] = float32(acc / sum)
		}
	}
	return out, nil
}

// CompressKVSparse is the CSA (compressed sparse) scheme: two projection series
// whose blocks OVERLAP by one, so block i summarises its own m tokens together
// with the previous block's.
//
// Block 0 has no predecessor. The reference masks the phantom one with -inf
// before the softmax rather than dropping it, keeping the softmax's shape
// identical for every block; this reproduces that, so block 0's weights come
// only from the a-series.
func CompressKVSparse(hPad, wKVa, wKVb, wZa, wZb, posA, posB []float32, blocks, m, dModel, dHead int) ([]float32, error) {
	if len(hPad) != blocks*m*dModel {
		return nil, fmt.Errorf("csa compress: input has %d values, want %d", len(hPad), blocks*m*dModel)
	}
	if len(posA) != m*dHead || len(posB) != m*dHead {
		return nil, fmt.Errorf("csa compress: pos_a/pos_b must each hold %d values", m*dHead)
	}
	out := make([]float32, blocks*dHead)
	// Per block: 2m candidates (m from series a on this block, m from series b
	// on the PREVIOUS block).
	cCat := make([]float64, 2*m*dHead)
	zCat := make([]float64, 2*m*dHead)
	for b := 0; b < blocks; b++ {
		for j := 0; j < m; j++ {
			tok := hPad[(b*m+j)*dModel : (b*m+j+1)*dModel]
			for e := 0; e < dHead; e++ {
				cCat[j*dHead+e] = dot(wKVa[e*dModel:(e+1)*dModel], tok)
				zCat[j*dHead+e] = dot(wZa[e*dModel:(e+1)*dModel], tok) + float64(posA[j*dHead+e])
			}
		}
		for j := 0; j < m; j++ {
			idx := (m + j) * dHead
			if b == 0 {
				// Phantom predecessor: value zero, gate -inf so softmax ignores it.
				for e := 0; e < dHead; e++ {
					cCat[idx+e] = 0
					zCat[idx+e] = math.Inf(-1)
				}
				continue
			}
			tok := hPad[((b-1)*m+j)*dModel : ((b-1)*m+j+1)*dModel]
			for e := 0; e < dHead; e++ {
				cCat[idx+e] = dot(wKVb[e*dModel:(e+1)*dModel], tok)
				zCat[idx+e] = dot(wZb[e*dModel:(e+1)*dModel], tok) + float64(posB[j*dHead+e])
			}
		}
		for e := 0; e < dHead; e++ {
			maxZ := math.Inf(-1)
			for j := 0; j < 2*m; j++ {
				if v := zCat[j*dHead+e]; v > maxZ {
					maxZ = v
				}
			}
			var sum, acc float64
			for j := 0; j < 2*m; j++ {
				w := math.Exp(zCat[j*dHead+e] - maxZ)
				sum += w
				acc += w * cCat[j*dHead+e]
			}
			out[b*dHead+e] = float32(acc / sum)
		}
	}
	return out, nil
}
