package hostmath

import (
	"fmt"
	"math"

	"overgo/internal/checked"
	"overgo/internal/tensor/dtype"
)

// BatchedHeadDotF32 computes one row-major batched attention-head dot product
// with selectable float32 or float64 accumulation.
func BatchedHeadDotF32(q, k []float32, batch, query, key, head, rows, heads, headWidth int, float32Accumulation bool) float64 {
	queryOffset := ((batch*rows+query)*heads + head) * headWidth
	keyOffset := ((batch*rows+key)*heads + head) * headWidth
	if float32Accumulation {
		var score float32
		for dimension := range headWidth {
			score += q[queryOffset+dimension] * k[keyOffset+dimension]
		}
		return float64(score)
	}
	var score float64
	for dimension := range headWidth {
		score += float64(q[queryOffset+dimension]) * float64(k[keyOffset+dimension])
	}
	return score
}

// RelativePositionAttentionF32 applies batched self-attention with additive
// relative-position bucket bias and an optional integer key mask.
func RelativePositionAttentionF32(q, k, v []float32, mask, buckets []int, embedding []float32, batches, rows, heads, headWidth, bucketCount int, bf16 bool) ([]float32, error) {
	for name, values := range map[string][]float32{"query": q, "key": k, "value": v} {
		if err := checked.Length(values, batches, rows, heads, headWidth); err != nil {
			return nil, fmt.Errorf("relative-position attention: %s: %w", name, err)
		}
	}
	if err := checked.Length(embedding, bucketCount, heads); err != nil {
		return nil, fmt.Errorf("relative-position attention: embedding: %w", err)
	}
	if err := checked.Length(buckets, rows, rows); err != nil {
		return nil, fmt.Errorf("relative-position attention: buckets: %w", err)
	}
	for _, bucket := range buckets {
		if !checked.ValidIndex(bucket, bucketCount) {
			return nil, fmt.Errorf("relative-position attention: bucket %d is invalid", bucket)
		}
	}
	if mask != nil {
		if err := checked.Length(mask, batches, rows); err != nil {
			return nil, fmt.Errorf("relative-position attention: mask: %w", err)
		}
	}
	output := make([]float32, len(q))
	for batch := range batches {
		ParallelRangeF64(rows, heads*rows*headWidth*2, func(startQuery, endQuery int) {
			scores := make([]float32, rows)
			for query := startQuery; query < endQuery; query++ {
				for head := range heads {
					for key := range rows {
						score := BatchedHeadDotF32(q, k, batch, query, key, head, rows, heads, headWidth, bf16)
						score += float64(embedding[buckets[query*rows+key]*heads+head])
						if mask != nil && mask[batch*rows+key] == 0 {
							score = math.Inf(-1)
						}
						scores[key] = float32(score)
					}
					roundBF16Slices(bf16, scores)
					softmaxF32Accumulation(scores, bf16)
					roundBF16Slices(bf16, scores)
					destination := output[((batch*rows+query)*heads+head)*headWidth:]
					for key, probability := range scores {
						value := v[((batch*rows+key)*heads+head)*headWidth:]
						for dimension := range headWidth {
							destination[dimension] += probability * value[dimension]
						}
					}
					roundBF16Slices(bf16, destination[:headWidth])
				}
			}
		})
	}
	return output, nil
}

func roundBF16Slices(enabled bool, values ...[]float32) {
	if !enabled {
		return
	}
	for _, value := range values {
		for index := range value {
			value[index] = dtype.RoundBF16(value[index])
		}
	}
}

func softmaxF32Accumulation(values []float32, bf16 bool) {
	if !bf16 {
		SoftmaxInPlace(values)
		return
	}
	maximum := float32(math.Inf(-1))
	for _, value := range values {
		maximum = max(maximum, value)
	}
	var total float32
	for index, value := range values {
		values[index] = float32(math.Exp(float64(value - maximum)))
		total += values[index]
	}
	for index := range values {
		values[index] /= total
	}
}

const (
	qkvComponents  = 3
	qkvKeyOffset   = 1
	qkvValueOffset = 2
)

func BatchedAttentionF32(q, k, v []float32, batches, queryRows, keyRows, hidden, heads int) []float32 {
	headWidth := hidden / heads
	output := make([]float32, batches*queryRows*hidden)
	scores := make([]float64, keyRows)
	scale := 1 / math.Sqrt(float64(headWidth))
	for batch := range batches {
		for query := range queryRows {
			for head := range heads {
				maximum := math.Inf(-1)
				for key := range keyRows {
					var score float64
					for channel := range headWidth {
						qIndex := (batch*queryRows+query)*hidden + head*headWidth + channel
						kIndex := (batch*keyRows+key)*hidden + head*headWidth + channel
						score += float64(q[qIndex] * k[kIndex])
					}
					scores[key] = score * scale
					maximum = math.Max(maximum, scores[key])
				}
				var total float64
				for key := range scores {
					scores[key] = math.Exp(scores[key] - maximum)
					total += scores[key]
				}
				for key := range keyRows {
					factor := float32(scores[key] / total)
					for channel := range headWidth {
						outIndex := (batch*queryRows+query)*hidden + head*headWidth + channel
						vIndex := (batch*keyRows+key)*hidden + head*headWidth + channel
						output[outIndex] += factor * v[vIndex]
					}
				}
			}
		}
	}
	return output
}

// InterleavedQKVAttentionF64: exact f64 score and value reductions.
func InterleavedQKVAttentionF64(qkv []float32, rows, hidden, heads int, scale float64) []float32 {
	headWidth := hidden / heads
	output := make([]float32, rows*hidden)
	ParallelRangeF64(heads, qkvValueOffset*rows*rows*headWidth, func(startHead, endHead int) {
		scores := make([]float64, rows)
		for head := startHead; head < endHead; head++ {
			for query := range rows {
				qOffset := query*qkvComponents*hidden + head*headWidth
				maximum := math.Inf(-1)
				for key := range rows {
					kOffset := key*qkvComponents*hidden + qkvKeyOffset*hidden + head*headWidth
					var dot float64
					for dimension := range headWidth {
						dot += float64(qkv[qOffset+dimension]) * float64(qkv[kOffset+dimension])
					}
					scores[key] = dot * scale
					maximum = max(maximum, scores[key])
				}
				var total float64
				for key := range scores {
					scores[key] = math.Exp(scores[key] - maximum)
					total += scores[key]
				}
				for dimension := range headWidth {
					var value float64
					for source := range rows {
						vOffset := source*qkvComponents*hidden + qkvValueOffset*hidden + head*headWidth
						value += scores[source] / total * float64(qkv[vOffset+dimension])
					}
					output[query*hidden+head*headWidth+dimension] = float32(value)
				}
			}
		}
	})
	return output
}

// GroupedBidirectionalAttentionF64 applies full-sequence grouped-query
// attention to row-major float64 projections.
func GroupedBidirectionalAttentionF64(q, k, v []float64, rows, heads, kvHeads, headWidth int) []float64 {
	group := heads / kvHeads
	scale := 1 / math.Sqrt(float64(headWidth))
	output := make([]float64, rows*heads*headWidth)
	ParallelRangeF64(heads, rows*rows*headWidth, func(startHead, endHead int) {
		scores := make([]float64, rows)
		for head := startHead; head < endHead; head++ {
			keyHead := head / group
			for query := range rows {
				queryVector := q[(query*heads+head)*headWidth : (query*heads+head+1)*headWidth]
				maximum := math.Inf(-1)
				for key := range rows {
					keyVector := k[(key*kvHeads+keyHead)*headWidth : (key*kvHeads+keyHead+1)*headWidth]
					var dot float64
					for dimension := range headWidth {
						dot += queryVector[dimension] * keyVector[dimension]
					}
					scores[key] = dot * scale
					maximum = max(maximum, scores[key])
				}
				var total float64
				for key := range scores {
					scores[key] = math.Exp(scores[key] - maximum)
					total += scores[key]
				}
				destination := output[(query*heads+head)*headWidth : (query*heads+head+1)*headWidth]
				for key := range rows {
					probability := scores[key] / total
					valueVector := v[(key*kvHeads+keyHead)*headWidth : (key*kvHeads+keyHead+1)*headWidth]
					for dimension := range headWidth {
						destination[dimension] += probability * valueVector[dimension]
					}
				}
			}
		}
	})
	return output
}

// GroupedCausalAttentionF64 applies masked causal grouped-query attention to
// row-major float64 projections. A nil keyMask attends every causal key.
func GroupedCausalAttentionF64(q, k, v []float64, rows, heads, kvHeads, headWidth int, keyMask []bool) []float64 {
	group := heads / kvHeads
	scale := 1 / math.Sqrt(float64(headWidth))
	output := make([]float64, rows*heads*headWidth)
	ParallelRangeF64(heads, rows*rows*headWidth, func(startHead, endHead int) {
		scores := make([]float64, rows)
		for head := startHead; head < endHead; head++ {
			keyHead := head / group
			for query := range rows {
				queryVector := q[(query*heads+head)*headWidth : (query*heads+head+1)*headWidth]
				maximum := math.Inf(-1)
				for key := 0; key <= query; key++ {
					if keyMask != nil && !keyMask[key] {
						continue
					}
					keyVector := k[(key*kvHeads+keyHead)*headWidth : (key*kvHeads+keyHead+1)*headWidth]
					var dot float64
					for dimension := range headWidth {
						dot += queryVector[dimension] * keyVector[dimension]
					}
					scores[key] = dot * scale
					maximum = max(maximum, scores[key])
				}
				var total float64
				for key := 0; key <= query; key++ {
					if keyMask != nil && !keyMask[key] {
						continue
					}
					scores[key] = math.Exp(scores[key] - maximum)
					total += scores[key]
				}
				destination := output[(query*heads+head)*headWidth : (query*heads+head+1)*headWidth]
				for key := 0; key <= query; key++ {
					if keyMask != nil && !keyMask[key] {
						continue
					}
					probability := scores[key] / total
					valueVector := v[(key*kvHeads+keyHead)*headWidth : (key*kvHeads+keyHead+1)*headWidth]
					for dimension := range headWidth {
						destination[dimension] += probability * valueVector[dimension]
					}
				}
			}
		}
	})
	return output
}
