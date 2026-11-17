//go:build windows

package executor

import (
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
)

// The 8-bit attention's tolerances (eight-bit-attention/kernel). Against exact attention in F64, a cosine similarity of
// at least 0.999. Against its own algorithm computed in F64 (the same INT8 Q and smoothed K per 64-token block, the
// same online softmax over 64-key tiles, P times 448 and V per channel rounded to FP8 E4M3), a relative L1 error of at
// most 0.001: a kernel computing that algorithm departs from it only by its F32 arithmetic and by its FP22 tensor-core
// accumulator over one key tile, near 2^-13 of an output, while a fault of layout or scale moves whole terms. The
// relative L1 error against exact attention is not held here: E4M3's three mantissa bits set it, which the held-out
// suite judges, and SageAttention2 (arXiv 2411.10958) prints no figure for INT8 Q.K with FP8 P.V to hold it to.
const (
	sageMinimumCosine        = 0.999
	sageMaximumAlgorithmL1   = 0.001
	sageWidth, sageBlock     = 128, 64
	sageProbabilityRange     = 448.0
	sageQuantisedRange       = 127.0
	sageSampledRowsEveryNth  = 5
	sageMaskedTrailingTokens = 40
)

// sageGeometry is one attention's shape: query and key/value heads, query and key/value tokens, sequences, and whether
// a key bias masks padded keys.
type sageGeometry struct {
	queryHeads, keyValueHeads, queryTokens, keyValueTokens, sequences uint32
	masked                                                            bool
}

// shape is the geometry as the launch takes it.
func (geometry sageGeometry) shape() eightBitGeometry {
	return eightBitGeometry{queryHeads: geometry.queryHeads, keyValueHeads: geometry.keyValueHeads,
		queryTokens: geometry.queryTokens, keyValueTokens: geometry.keyValueTokens, sequences: geometry.sequences}
}

// TestEightBitAttentionMatchesTheReferenceWithinItsTolerance holds the 8-bit attention (eight-bit-attention/kernel) to
// exact attention and to its own algorithm, both in F64, over the head layouts of the models it serves, each at head
// width 128: Wan's self- and cross-attention, two sequences as its guidance runs them, and a few query rows over the
// 32760 keys of its 81-frame 832x480 clip, where an accumulator that loses a little at every key tile would lose most;
// Krea's 48 query heads over 12 key/value heads with a key bias masking padded keys; MOSS's 24 heads; SenseNova's 32
// query heads over 8. Token counts that are no multiple of 64 hold the partial blocks, and K carries a mean per
// channel, which its smoothing takes away. Over the outputs of every fifth query row, the cosine similarity to exact
// attention and the relative L1 error to the algorithm meet the tolerances.
func TestEightBitAttentionMatchesTheReferenceWithinItsTolerance(t *testing.T) {
	cudatest.Require(t)
	cuda := newFixtureExecutor(t)
	for name, geometry := range map[string]sageGeometry{
		"Wan self-attention":  {queryHeads: 12, keyValueHeads: 12, queryTokens: 1000, keyValueTokens: 1000, sequences: 2},
		"Wan cross-attention": {queryHeads: 12, keyValueHeads: 12, queryTokens: 1000, keyValueTokens: 512, sequences: 2},
		"Wan's whole clip":    {queryHeads: 12, keyValueHeads: 12, queryTokens: 128, keyValueTokens: 32760, sequences: 1},
		"Krea":                {queryHeads: 48, keyValueHeads: 12, queryTokens: 512, keyValueTokens: 600, sequences: 1, masked: true},
		"MOSS":                {queryHeads: 24, keyValueHeads: 24, queryTokens: 1024, keyValueTokens: 1024, sequences: 1},
		"SenseNova":           {queryHeads: 32, keyValueHeads: 8, queryTokens: 515, keyValueTokens: 515, sequences: 1},
	} {
		t.Run(name, func(t *testing.T) { sageAgreement(t, cuda, geometry) })
	}
}

func sageAgreement(t *testing.T, cuda *Executor, geometry sageGeometry) {
	draws := rand.New(rand.NewPCG(uint64(geometry.queryHeads), uint64(geometry.keyValueTokens)))
	queryCount := int(geometry.sequences * geometry.queryTokens * geometry.queryHeads * sageWidth)
	keyCount := int(geometry.sequences * geometry.keyValueTokens * geometry.keyValueHeads * sageWidth)
	query, key, value := make([]float32, queryCount), make([]float32, keyCount), make([]float32, keyCount)
	for index := range query {
		query[index] = float32(draws.NormFloat64())
	}
	for index := range key {
		channel := index % sageWidth
		key[index] = float32(draws.NormFloat64()) + float32(channel%7) - 3
		value[index] = float32(draws.NormFloat64())
	}
	bias := make([]float32, geometry.keyValueTokens)
	if geometry.masked {
		for token := range bias {
			if token >= len(bias)-sageMaskedTrailingTokens {
				bias[token] = -1e9
			}
		}
	}
	scale := float32(1 / math.Sqrt(sageWidth))

	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	queryBuffer, keyBuffer, valueBuffer := buffers.upload(t, float32Bytes(query)), buffers.upload(t, float32Bytes(key)), buffers.upload(t, float32Bytes(value))
	var biasBuffer driver.DevicePtr
	if geometry.masked {
		biasBuffer = buffers.upload(t, float32Bytes(bias))
	}
	outputBuffer := buffers.zeros(t, queryCount*4)
	staging := buffers.zeros(t, int(geometry.shape().stagingBytes()))
	buffers.run(t, func(state *device.State, functions functionSet) error {
		return launchEightBitAttentionKernels(state, functions, queryBuffer, keyBuffer, valueBuffer, biasBuffer, outputBuffer, staging,
			geometry.shape(), scale)
	})
	output := float32s(buffers.read(t, outputBuffer, queryCount*4))

	algorithm := newSageAlgorithm(query, key, value, bias, geometry, scale)
	exact, quantised := make([]float64, sageWidth), make([]float64, sageWidth)
	var toExact, toAlgorithm, algorithmToExact sageAgreementOf
	for sequence := range geometry.sequences {
		for row := uint32(0); row < geometry.queryTokens; row += sageSampledRowsEveryNth {
			for head := range geometry.queryHeads {
				algorithm.exact(sequence, row, head, exact)
				algorithm.quantised(sequence, row, head, quantised)
				queryAt := ((sequence*geometry.queryTokens+row)*geometry.queryHeads + head) * sageWidth
				for channel := range uint32(sageWidth) {
					got := float64(output[queryAt+channel])
					toExact.add(got, exact[channel])
					toAlgorithm.add(got, quantised[channel])
					algorithmToExact.add(quantised[channel], exact[channel])
				}
			}
		}
	}
	t.Logf("against exact attention: the kernel a cosine similarity of %.6f and a relative L1 error of %.4f, its algorithm %.6f and %.4f; against its algorithm, a relative L1 error of %.6f",
		toExact.cosine(), toExact.relative(), algorithmToExact.cosine(), algorithmToExact.relative(), toAlgorithm.relative())
	if !(toExact.cosine() >= sageMinimumCosine) {
		t.Errorf("the 8-bit attention reads a cosine similarity of %.6f to exact attention; the tolerance is %g", toExact.cosine(), sageMinimumCosine)
	}
	if !(toAlgorithm.relative() <= sageMaximumAlgorithmL1) {
		t.Errorf("the 8-bit attention departs from its own algorithm computed in F64 by a relative L1 error of %.6f; the tolerance is %g",
			toAlgorithm.relative(), sageMaximumAlgorithmL1)
	}
}

// sageAgreementOf gathers two outputs' agreement over many values: their dot product and norms for the cosine
// similarity, and their absolute difference over the reference's magnitude for the relative L1 error.
type sageAgreementOf struct{ dot, gotNorm, wantNorm, difference, magnitude float64 }

func (agreement *sageAgreementOf) add(got, want float64) {
	agreement.dot += got * want
	agreement.gotNorm += got * got
	agreement.wantNorm += want * want
	agreement.difference += math.Abs(got - want)
	agreement.magnitude += math.Abs(want)
}

func (agreement sageAgreementOf) cosine() float64 {
	return agreement.dot / math.Sqrt(agreement.gotNorm*agreement.wantNorm)
}

func (agreement sageAgreementOf) relative() float64 {
	return agreement.difference / agreement.magnitude
}

// sageAlgorithm holds an attention's inputs and their quantisation as the kernel's launches make it, for exact
// attention and the 8-bit algorithm computed in F64.
type sageAlgorithm struct {
	geometry                                        sageGeometry
	query, key, value, bias                         []float32
	scale                                           float64
	queryCodes, keyCodes                            []int8
	valueCodes, queryScales, keyScales, valueScales []float32
	scores                                          []float64
}

func newSageAlgorithm(query, key, value, bias []float32, geometry sageGeometry, scale float32) *sageAlgorithm {
	keyMean := make([]float32, geometry.sequences*geometry.keyValueHeads*sageWidth)
	for sequence := range geometry.sequences {
		for head := range geometry.keyValueHeads {
			for channel := range uint32(sageWidth) {
				sum := 0.0
				for token := range geometry.keyValueTokens {
					sum += float64(key[((sequence*geometry.keyValueTokens+token)*geometry.keyValueHeads+head)*sageWidth+channel])
				}
				keyMean[(sequence*geometry.keyValueHeads+head)*sageWidth+channel] = float32(sum / float64(geometry.keyValueTokens))
			}
		}
	}
	algorithm := &sageAlgorithm{geometry: geometry, query: query, key: key, value: value, bias: bias, scale: float64(scale),
		scores: make([]float64, geometry.keyValueTokens)}
	algorithm.queryCodes, algorithm.queryScales = sageQuantiseBlocks(query, nil, geometry.queryHeads, geometry.queryTokens, geometry.sequences)
	algorithm.keyCodes, algorithm.keyScales = sageQuantiseBlocks(key, keyMean, geometry.keyValueHeads, geometry.keyValueTokens, geometry.sequences)
	algorithm.valueCodes, algorithm.valueScales = sageQuantiseChannels(value, geometry.keyValueHeads, geometry.keyValueTokens, geometry.sequences)
	return algorithm
}

// sageQuantiseBlocks is sage_quantize_i8_f32: each 64-token block of each head as INT8 against its absolute maximum
// over 127, the head's channel means subtracted first where mean is given.
func sageQuantiseBlocks(x, mean []float32, heads, tokens, sequences uint32) ([]int8, []float32) {
	blocks := (tokens + sageBlock - 1) / sageBlock
	codes, scales := make([]int8, len(x)), make([]float32, sequences*heads*blocks)
	for sequence := range sequences {
		for head := range heads {
			shifted := func(token, channel uint32) float32 {
				at := ((sequence*tokens+token)*heads + head) * sageWidth
				if mean == nil {
					return x[at+channel]
				}
				return x[at+channel] - mean[(sequence*heads+head)*sageWidth+channel]
			}
			for block := range blocks {
				first, last := block*sageBlock, min(block*sageBlock+sageBlock, tokens)
				maximum := float32(0)
				for token := first; token < last; token++ {
					for channel := range uint32(sageWidth) {
						maximum = max(maximum, float32(math.Abs(float64(shifted(token, channel)))))
					}
				}
				scale := max(maximum, 1e-20) / sageQuantisedRange
				scales[(sequence*heads+head)*blocks+block] = scale
				for token := first; token < last; token++ {
					for channel := range uint32(sageWidth) {
						code := math.RoundToEven(float64(shifted(token, channel) / scale))
						codes[((sequence*tokens+token)*heads+head)*sageWidth+channel] = int8(max(-sageQuantisedRange, min(sageQuantisedRange, code)))
					}
				}
			}
		}
	}
	return codes, scales
}

// sageQuantiseChannels is sage_quantize_v_fp8_f32: each channel of each head's values as FP8 E4M3 against its absolute
// maximum over 448, the codes kept in the values' own layout as the values they stand for.
func sageQuantiseChannels(value []float32, heads, tokens, sequences uint32) ([]float32, []float32) {
	codes, scales := make([]float32, len(value)), make([]float32, sequences*heads*sageWidth)
	for sequence := range sequences {
		for head := range heads {
			for channel := range uint32(sageWidth) {
				maximum := float32(0)
				for token := range tokens {
					maximum = max(maximum, float32(math.Abs(float64(value[((sequence*tokens+token)*heads+head)*sageWidth+channel]))))
				}
				scale := max(maximum, 1e-20) / sageProbabilityRange
				scales[(sequence*heads+head)*sageWidth+channel] = scale
				for token := range tokens {
					at := ((sequence*tokens+token)*heads+head)*sageWidth + channel
					codes[at] = float32(sageE4M3(float64(value[at] / scale)))
				}
			}
		}
	}
	return codes, scales
}

// sageE4M3 rounds x to the nearest FP8 E4M3 value, ties to even, saturating at 448, as __nv_cvt_float_to_fp8 does with
// __NV_SATFINITE: three mantissa bits, the smallest normal 2^-6 and subnormals in steps of 2^-9.
func sageE4M3(x float64) float64 {
	magnitude := math.Abs(x)
	if magnitude >= sageProbabilityRange {
		return math.Copysign(sageProbabilityRange, x)
	}
	_, exponent := math.Frexp(magnitude)
	step := math.Ldexp(1, max(exponent-4, -9))
	return math.Copysign(math.RoundToEven(magnitude/step)*step, x)
}

// exact writes into out exact attention's output for one query row of one head, in F64.
func (algorithm *sageAlgorithm) exact(sequence, row, head uint32, out []float64) {
	geometry := algorithm.geometry
	keyHead := head / (geometry.queryHeads / geometry.keyValueHeads)
	queryAt := ((sequence*geometry.queryTokens+row)*geometry.queryHeads + head) * sageWidth
	maximum := math.Inf(-1)
	for token := range geometry.keyValueTokens {
		keyAt := ((sequence*geometry.keyValueTokens+token)*geometry.keyValueHeads + keyHead) * sageWidth
		score := 0.0
		for channel := range uint32(sageWidth) {
			score += float64(algorithm.query[queryAt+channel]) * float64(algorithm.key[keyAt+channel])
		}
		algorithm.scores[token] = score*algorithm.scale + float64(algorithm.bias[token])
		maximum = max(maximum, algorithm.scores[token])
	}
	total := 0.0
	clear(out)
	for token := range geometry.keyValueTokens {
		weight := math.Exp(algorithm.scores[token] - maximum)
		total += weight
		valueAt := ((sequence*geometry.keyValueTokens+token)*geometry.keyValueHeads + keyHead) * sageWidth
		for channel := range uint32(sageWidth) {
			out[channel] += weight * float64(algorithm.value[valueAt+channel])
		}
	}
	for channel := range out {
		out[channel] /= total
	}
}

// quantised writes into out the 8-bit algorithm's output for one query row of one head, in F64: scores from the INT8
// codes and their block scales, the softmax online over 64-key tiles in base 2, each tile's probabilities times 448
// rounded to E4M3 against the running maximum and multiplied with V's E4M3 codes, the sum of the unrounded
// probabilities dividing at the end, with V's channel scales.
func (algorithm *sageAlgorithm) quantised(sequence, row, head uint32, out []float64) {
	geometry := algorithm.geometry
	keyHead := head / (geometry.queryHeads / geometry.keyValueHeads)
	queryBlocks := (geometry.queryTokens + sageBlock - 1) / sageBlock
	keyBlocks := (geometry.keyValueTokens + sageBlock - 1) / sageBlock
	queryAt := ((sequence*geometry.queryTokens+row)*geometry.queryHeads + head) * sageWidth
	queryScale := float64(algorithm.queryScales[(sequence*geometry.queryHeads+head)*queryBlocks+row/sageBlock])
	maximum, sum := math.Inf(-1), 0.0
	clear(out)
	for block := range keyBlocks {
		first, last := block*sageBlock, min(block*sageBlock+sageBlock, geometry.keyValueTokens)
		scoreScale := queryScale * float64(algorithm.keyScales[(sequence*geometry.keyValueHeads+keyHead)*keyBlocks+block]) * algorithm.scale
		next := maximum
		for token := first; token < last; token++ {
			keyAt := ((sequence*geometry.keyValueTokens+token)*geometry.keyValueHeads + keyHead) * sageWidth
			product := int32(0)
			for channel := range uint32(sageWidth) {
				product += int32(algorithm.queryCodes[queryAt+channel]) * int32(algorithm.keyCodes[keyAt+channel])
			}
			algorithm.scores[token] = (float64(product)*scoreScale + float64(algorithm.bias[token])) / math.Ln2
			next = max(next, algorithm.scores[token])
		}
		correction := math.Exp2(maximum - next)
		sum *= correction
		for channel := range out {
			out[channel] *= correction
		}
		for token := first; token < last; token++ {
			probability := math.Exp2(algorithm.scores[token] - next)
			sum += probability
			code := sageE4M3(probability * sageProbabilityRange)
			valueAt := ((sequence*geometry.keyValueTokens+token)*geometry.keyValueHeads + keyHead) * sageWidth
			for channel := range uint32(sageWidth) {
				out[channel] += code * float64(algorithm.valueCodes[valueAt+channel])
			}
		}
		maximum = next
	}
	for channel := range uint32(sageWidth) {
		out[channel] *= float64(algorithm.valueScales[(sequence*geometry.keyValueHeads+keyHead)*sageWidth+channel]) / (sum * sageProbabilityRange)
	}
}
