package projector

import (
	"math"

	"overgo/internal/hostmath"
)

func (p visionAttentionPlan) cpu(qkv []float32, rows int) []float32 {
	headWidth := p.headWidth()
	output := make([]float32, rows*p.hidden)
	scale := float64(p.scale())
	hostmath.ParallelRangeF64(p.heads, 2*rows*rows*headWidth, func(startHead, endHead int) {
		scores := make([]float64, rows)
		for head := startHead; head < endHead; head++ {
			for query := 0; query < rows; query++ {
				qOffset := query*attentionProjectionCount*p.hidden + head*headWidth
				maximum := math.Inf(-1)
				for key := 0; key < rows; key++ {
					kOffset := key*attentionProjectionCount*p.hidden + p.hidden + head*headWidth
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
						vOffset := source*attentionProjectionCount*p.hidden + 2*p.hidden + head*headWidth
						value += scores[source] / total * float64(qkv[vOffset+dimension])
					}
					output[query*p.hidden+head*headWidth+dimension] = float32(value)
				}
			}
		}
	})
	return output
}
