package hostmath

import "math"

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
