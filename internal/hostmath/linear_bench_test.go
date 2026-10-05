package hostmath

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// benchShapes are two projections of Qwen2.5-0.5B over a 512-token window, the
// shapes the host trainer's three linear kernels spend a LoRA step on: a
// square attention projection and a wide MLP one.
var benchShapes = []struct{ rows, in, out int }{{512, 896, 512}, {512, 896, 4864}}

func benchValues(count int) []float32 {
	random := rand.New(rand.NewPCG(5, uint64(count)))
	values := make([]float32, count)
	for index := range values {
		values[index] = random.Float32()*2 - 1
	}
	return values
}

// BenchmarkLinearKernels times the forward dot kernel on one thread, and the
// weight and input gradients on the worker pool.
func BenchmarkLinearKernels(b *testing.B) {
	for _, shape := range benchShapes {
		x, w, dy := benchValues(shape.rows*shape.in), benchValues(shape.out*shape.in), benchValues(shape.rows*shape.out)
		dst, dW, dx := make([]float32, shape.rows*shape.out), make([]float32, shape.out*shape.in), make([]float32, shape.rows*shape.in)
		name := fmt.Sprintf("%dx%dx%d", shape.rows, shape.in, shape.out)
		b.Run("forward/"+name, func(b *testing.B) {
			for range b.N {
				linearCols(dst, x, w, shape.rows, shape.in, shape.out, 0, min(shape.out, 512))
			}
		})
		b.Run("weight/"+name, func(b *testing.B) {
			for range b.N {
				LinearWeightGradient(dW, x, dy, shape.rows, shape.in, shape.out)
			}
		})
		b.Run("input/"+name, func(b *testing.B) {
			for range b.N {
				linearInputGradient(dx, dy, w, shape.rows, shape.in, shape.out, false)
			}
		})
	}
}
