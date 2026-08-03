package projector

import "math"

func visionAttention(qkv []float32, rows, hidden, heads int) []float32 {
	headWidth := hidden / heads
	output := make([]float32, rows*hidden)
	scale := 1 / math.Sqrt(float64(headWidth))
	parallelRows(heads, func(startHead, endHead int) {
		scores := make([]float64, rows)
		for head := startHead; head < endHead; head++ {
			for query := 0; query < rows; query++ {
				qOffset := query*3*hidden + head*headWidth
				maximum := math.Inf(-1)
				for key := 0; key < rows; key++ {
					kOffset := key*3*hidden + hidden + head*headWidth
					dot := 0.0
					for dimension := 0; dimension < headWidth; dimension++ {
						dot += float64(qkv[qOffset+dimension]) * float64(qkv[kOffset+dimension])
					}
					scores[key] = dot * scale
					maximum = max(maximum, scores[key])
				}
				total := 0.0
				for key := range scores {
					scores[key] = math.Exp(scores[key] - maximum)
					total += scores[key]
				}
				for dimension := 0; dimension < headWidth; dimension++ {
					value := 0.0
					for source := 0; source < rows; source++ {
						vOffset := source*3*hidden + 2*hidden + head*headWidth
						value += scores[source] / total * float64(qkv[vOffset+dimension])
					}
					output[query*hidden+head*headWidth+dimension] = float32(value)
				}
			}
		}
	})
	return output
}
