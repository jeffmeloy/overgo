package reference

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
)

// gatedDeltaChunk preserves the native CPU chunk arithmetic and packed-state layout.
// Scratch scales with one chunk and one head, independently of prompt length.
func gatedDeltaChunk(shape tensor.Shape, inputs []Value, attributes tensor.GatedDeltaNetAttributes) (Value, error) {
	const chunk = tensor.GatedDeltaChunkSize
	query, key, value := inputs[0], inputs[1], inputs[tensor.PairedExtent]
	g, beta, state := inputs[tensor.TripleExtent], inputs[tensor.TripleExtent+1], inputs[tensor.TripleExtent+tensor.PairedExtent]
	width, heads, tokens, batches := dims4(value.Shape)
	queryHeads, keyHeads := int(query.Shape.Dims[1]), int(key.Shape.Dims[1])
	if width <= 0 || heads <= 0 || tokens <= 0 || batches <= 0 || g.Shape.Dims[0] != 1 || heads%queryHeads != 0 || heads%keyHeads != 0 {
		return Value{}, fmt.Errorf("chunk GDN shape is unsupported")
	}
	count, err := shape.Elements()
	if err != nil {
		return Value{}, err
	}
	output := Value{Shape: shape, Data: make([]float32, count)}
	attentionElements := width * heads * tokens * batches
	repeat := attributes.RepeatInterleave
	exp := func(x float32) float32 {
		return float32(math.Exp(float64(x)))
	}
	scale := float32(1 / math.Sqrt(float64(width)))
	q, k, kb, vb := make([]float32, chunk*width), make([]float32, chunk*width), make([]float32, chunk*width), make([]float32, chunk*width)
	cumulative, decay, triangle := make([]float32, chunk), make([]float32, chunk*chunk), make([]float32, chunk*chunk)
	mixed, kWeighted, kCumulative, vNew := make([]float32, chunk*width), make([]float32, chunk*width), make([]float32, chunk*width), make([]float32, chunk*width)
	scores := make([]float32, chunk*chunk)
	current, next := make([]float32, width*width), make([]float32, width*width)
	before := make([]float32, chunk)
	for batch := range batches {
		for head := range heads {
			qHead, kHead := head%queryHeads, head%keyHeads
			if repeat {
				qHead = head / (heads / queryHeads)
				kHead = head / (heads / keyHeads)
			}
			for ki := range width {
				for vi := range width {
					current[ki*width+vi] = state.Data[((batch*heads+head)*width+vi)*width+ki]
				}
			}
			for start := 0; start < tokens; start += chunk {
				clear(q)
				clear(k)
				clear(kb)
				clear(vb)
				clear(decay)
				clear(triangle)
				var sum float64
				valid := min(chunk, tokens-start)
				for row := range chunk {
					if row < valid {
						token := start + row
						sum += float64(g.Data[(batch*tokens+token)*heads+head])
						b := beta.Data[(batch*tokens+token)*heads+head]
						for column := range width {
							at := row*width + column
							q[at] = float32(query.Data[((batch*tokens+token)*queryHeads+qHead)*width+column] * scale)
							k[at] = key.Data[((batch*tokens+token)*keyHeads+kHead)*width+column]
							kb[at] = float32(k[at] * b)
							vb[at] = float32(value.Data[((batch*tokens+token)*heads+head)*width+column] * b)
						}
					}
					cumulative[row] = float32(sum)
				}
				for row := range chunk {
					for col := 0; col <= row; col++ {
						decay[row*chunk+col] = exp(float32(cumulative[row] - cumulative[col]))
						if col < row {
							var dot float32
							for inner := range width {
								dot = chunkFMA(kb[row*width+inner], k[col*width+inner], dot)
							}
							triangle[row*chunk+col] = -float32(dot * decay[row*chunk+col])
						}
					}
				}
				for row := 1; row < chunk; row++ {
					copy(before, triangle[row*chunk:row*chunk+row])
					for col := 0; col < row; col++ {
						triangle[row*chunk+col] = float32(before[col] + chunkOuterSum(before, triangle, row, col, chunk))
					}
				}
				for row := range chunk {
					triangle[row*chunk+row] += 1
					factor := exp(cumulative[row])
					for col := range width {
						kWeighted[row*width+col] = float32(kb[row*width+col] * factor)
					}
				}
				chunkMatrixProduct(mixed, triangle, vb, chunk, chunk, width)
				chunkMatrixProduct(kCumulative, triangle, kWeighted, chunk, chunk, width)
				chunkMatrixProduct(vNew, kCumulative, current, chunk, width, width)
				for i := range vNew {
					vNew[i] = float32(mixed[i] - vNew[i])
				}
				for row := range chunk {
					for col := range chunk {
						var dot float32
						for inner := range width {
							dot = chunkFMA(q[row*width+inner], k[col*width+inner], dot)
						}
						scores[row*chunk+col] = float32(dot * decay[row*chunk+col])
					}
				}
				chunkMatrixProduct(mixed, scores, vNew, chunk, chunk, width)
				for row := range valid {
					factor := exp(cumulative[row])
					for vi := range width {
						var inter float32
						for ki := range width {
							inter = chunkFMA(float32(q[row*width+ki]*factor), current[ki*width+vi], inter)
						}
						output.Data[((batch*tokens+start+row)*heads+head)*width+vi] = float32(inter + mixed[row*width+vi])
					}
				}
				finalDecay := exp(cumulative[chunk-1])
				for row := range chunk {
					factor := exp(float32(cumulative[chunk-1] - cumulative[row]))
					for ki := range width {
						kWeighted[row*width+ki] = float32(k[row*width+ki] * factor)
					}
				}
				for ki := range width {
					for vi := range width {
						var update float32
						for row := range chunk {
							update = chunkFMA(kWeighted[row*width+ki], vNew[row*width+vi], update)
						}
						next[ki*width+vi] = float32(float32(current[ki*width+vi]*finalDecay) + update)
					}
				}
				current, next = next, current
			}
			for ki := range width {
				for vi := range width {
					output.Data[attentionElements+((batch*heads+head)*width+vi)*width+ki] = current[ki*width+vi]
				}
			}
		}
	}
	return output, nil
}

func chunkMatrixProduct(out, a, b []float32, m, k, n int) {
	for row := range m {
		for col := range n {
			var sum float32
			for inner := range k {
				sum = chunkFMA(a[row*k+inner], b[inner*n+col], sum)
			}
			out[row*n+col] = sum
		}
	}
}

// chunkSumVectors follows SumKernel.cpp's four AVX2 vectors and sixteen-element partial sums.
// Reconsider these CPU arithmetic parameters with a different native fixture backend.
const (
	chunkSumVectors       = 4
	chunkSumLanes         = 8
	chunkSumPartialLength = 16
)

// chunkOuterSum follows SumKernel.cpp's AVX2 outer-sum and tail dispatch.
func chunkOuterSum(before, triangle []float32, row, col, stride int) float32 {
	const vectorBlock = chunkSumVectors * chunkSumLanes
	leading := row / vectorBlock * vectorBlock
	if row < chunkSumLanes {
		leading = row / chunkSumVectors * chunkSumVectors
	}
	if col < leading {
		var sum, partial float32
		for inner := range row {
			partial = float32(partial + float32(before[inner]*triangle[inner*stride+col]))
			if (inner+1)%chunkSumPartialLength == 0 {
				sum = float32(sum + partial)
				partial = 0
			}
		}
		return float32(partial + sum)
	}
	var sums [chunkSumVectors]float32
	for inner := 0; inner < row/chunkSumVectors*chunkSumVectors; inner++ {
		sums[inner%chunkSumVectors] = float32(sums[inner%chunkSumVectors] + float32(before[inner]*triangle[inner*stride+col]))
	}
	for inner := row / chunkSumVectors * chunkSumVectors; inner < row; inner++ {
		sums[0] = float32(sums[0] + float32(before[inner]*triangle[inner*stride+col]))
	}
	result := sums[0]
	for _, sum := range sums[1:] {
		result = float32(result + sum)
	}
	return result
}

func chunkFMA(a, b, c float32) float32 { return float32(math.FMA(float64(a), float64(b), float64(c))) }
