package executor

import (
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestNativeRotarySelection(t *testing.T) {
	for _, native := range []bool{false, true} {
		b := tensor.NewBuilder()
		b.SetCacheAppendPlan(tensor.CacheAppendPlan{ActiveTokens: 2, CapacityTokens: 5})
		x := b.Input("x", dtype.F32, tensor.MustShape(64, 2, 3))
		past := b.Input("past", dtype.F32, tensor.MustShape(64, 2, 5))
		rope := b.RoPEWithOptions(x, tensor.RoPEOptions{NativeBF16: native, Layout: tensor.RoPELayoutNeoX, Positions: []uint32{2, 3, 4}, RotaryDimensions: 16, FrequencyBase: 10000, FrequencyScale: 1})
		out := b.AppendCache(past, rope, 2)
		if b.Err() != nil {
			t.Fatal(b.Err())
		}
		graph, err := Compile(out)
		if err != nil {
			t.Fatal(err)
		}
		fusion := graph.nodes[graph.orderIndexes[out]].fusion
		if fused := fusion != nil && fusion.kind == compiledFusionRopeAppend; fused == native {
			t.Fatalf("rotary fusion violates policy: native=%v", native)
		}
		parameters := graph.NewRuntimeAttributes()
		next := rope.Attrs.(tensor.RoPEAttributes)
		next.NativeBF16 = !native
		if err := parameters.Set(rope, next); err == nil {
			t.Fatal("runtime native rotary policy changed")
		}
		next.NativeBF16 = native
		next.Positions = []uint32{3, 4, 5}
		if err := parameters.Set(rope, next); err != nil {
			t.Fatal(err)
		}
	}
	for _, invalid := range []string{"layout", "scaling", "base", "factors", "sections"} {
		t.Run(invalid, func(t *testing.T) {
			b := tensor.NewBuilder()
			x := b.Input("x", dtype.F32, tensor.MustShape(64, 2, 1))
			options := tensor.RoPEOptions{NativeBF16: true, Layout: tensor.RoPELayoutNeoX, Positions: []uint32{0}, RotaryDimensions: 16, FrequencyBase: 10000, FrequencyScale: 1}
			switch invalid {
			case "layout":
				options.Layout = tensor.RoPELayoutNormal
			case "scaling":
				options.FrequencyScale = .5
			case "base":
				options.FrequencyBase = float32(math.Inf(1))
			case "factors":
				options.FrequencyFactors = b.Input("factors", dtype.F32, tensor.MustShape(8))
			case "sections":
				options.Positions = nil
				options.MultiPositions = &[4][]uint32{{0}, {0}, {0}, {0}}
				options.Sections = [4]int32{1, 1, 1, 1}
				options.InterleavedSections = true
			}
			if b.RoPEWithOptions(x, options) != nil || b.Err() == nil {
				t.Fatal("unsupported native rotary combination admitted")
			}
		})
	}
}
