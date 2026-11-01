package executor

import (
	"fmt"
	"math"
	"testing"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func chunkTestGraph(t *testing.T, tokens uint64) *tensor.Tensor {
	t.Helper()
	b := tensor.NewBuilder()
	shapes := []tensor.Shape{
		tensor.MustShape(8, 2, tokens, 2), tensor.MustShape(8, 2, tokens, 2),
		tensor.MustShape(8, 4, tokens, 2), tensor.MustShape(1, 4, tokens, 2),
		tensor.MustShape(1, 4, tokens, 2), tensor.MustShape(8, 8, 4, 2),
	}
	inputs := make([]*tensor.Tensor, len(shapes))
	for i, shape := range shapes {
		inputs[i] = b.Input(fmt.Sprint(i), dtype.F32, shape)
	}
	out := b.GatedDeltaNetChunked(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], true)
	if b.Err() != nil {
		t.Fatal(b.Err())
	}
	return out
}

func TestGatedDeltaChunkStaging(t *testing.T) {
	var staging uint64
	for _, tokens := range []uint64{1, 63, 64, 65, 129, 4096} {
		graph, err := Compile(chunkTestGraph(t, tokens))
		if err != nil || !graph.needBlas || graph.matmulStagingBytes == 0 {
			t.Fatalf("compile chunk graph: %v", err)
		}
		if staging != 0 && staging != graph.matmulStagingBytes {
			t.Fatal("chunk workspace grows with prompt length")
		}
		staging = graph.matmulStagingBytes
	}
	for _, invalid := range []string{"vector-gate", "width-limit", "token-limit", "group-limit", "byte-overflow"} {
		t.Run(invalid, func(t *testing.T) {
			node := chunkTestGraph(t, 65)
			dims := &node.Inputs[2].Shape.Dims
			switch invalid {
			case "vector-gate":
				node.Inputs[3].Shape.Dims[0] = dims[0]
			case "width-limit":
				dims[0] = math.MaxInt32 + 1
			case "token-limit":
				dims[2] = math.MaxUint32 + 1
			case "group-limit":
				dims[1], dims[3] = math.MaxInt32, 2
			case "byte-overflow":
				dims[0], dims[1], dims[3] = math.MaxInt32, 1, 1
			}
			if _, err := deltaChunkStagingBytes(node); err == nil {
				t.Fatal("unsupported workspace admitted")
			}
		})
	}
	node := chunkTestGraph(t, 65)
	in := node.Inputs
	b := tensor.NewBuilder()
	if b.GatedDeltaNetChunked(in[0], in[1], in[2], in[2], in[4], in[5], true) != nil || b.Err() == nil {
		t.Fatal("builder accepts vector gate for chunk policy")
	}
	b = tensor.NewBuilder()
	recurrent := b.GatedDeltaNet(in[0], in[1], in[2], in[2], in[4], in[5])
	if b.Err() != nil || recurrent.Attrs.(tensor.GatedDeltaNetAttributes).Chunked {
		t.Fatal("recurrent vector gate default changed", b.Err())
	}
}
