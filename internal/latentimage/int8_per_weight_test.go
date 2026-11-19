package latentimage

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/comfyquant"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testcheck"
)

// TestEachProjectionMultipliesAsItsFileHoldsIt holds MOSS's joint projections to the form their file holds each in
// (int8-per-weight, SQA-423): of a layer whose fused qkv and gate-and-up projection are int8 ConvRot over 256 columns,
// whose output projection is plain BF16, and whose down projection is int8 ConvRot over 64 columns, the q, k, v and
// gate-and-up weights read as int8 ConvRot and the rotary graph holds them so (dtype.I8ConvRot), and the output and
// down weights are held in the graph's storage. One flag read from the first q projection had held every projection in
// int8, and the plain output projection was refused as sampling bound it.
func TestEachProjectionMultipliesAsItsFileHoldsIt(t *testing.T) {
	t.Parallel()
	const width, intermediate = comfyquant.Group, 2 * comfyquant.Group
	tensors := map[string]safetensors.Tensor{}
	add := func(name, dataType string, shape []uint64, data []byte) {
		tensors[name] = testcheck.Must(safetensors.NewTensor(name, dataType, shape, bytes.NewReader(data), 0, int64(len(data))))(t)
	}
	convRot := func(base string, rows, columns, group uint64) {
		record := []byte(`{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":` + map[uint64]string{256: "256", 64: "64"}[group] + `}`)
		scales := make([]byte, 4*rows)
		for row := range rows {
			binary.LittleEndian.PutUint32(scales[4*row:], math.Float32bits(1))
		}
		add(base+".weight", "I8", []uint64{rows, columns}, make([]byte, rows*columns))
		add(base+".weight_scale", "F32", []uint64{rows, 1}, scales)
		add(base+".comfy_quant", "U8", []uint64{uint64(len(record))}, record)
	}
	convRot("layers.0.proj.qkv", 3*width, width, comfyquant.Group)
	convRot("layers.0.proj.ffn_up_gate", 2*intermediate, width, comfyquant.Group)
	convRot("layers.0.proj.ffn_down", width, intermediate, 64)
	add("layers.0.proj.o.weight", "BF16", []uint64{width, width}, make([]byte, 2*width*width))
	held := testcheck.Must(convRotProjections(&safetensors.Source{Tensors: tensors}, 1))(t)
	for _, projection := range jointProjections {
		name := "layers.0." + projection + ".weight"
		want := projection != "proj.o" && projection != "proj.ffn_down"
		if held[name] != want {
			t.Errorf("%s reads as held in int8 ConvRot over 256 columns: %t; want %t", name, held[name], want)
		}
	}
	var weights tensor.WeightBindings
	graph := testcheck.Must(newRotaryGraph(TransformerSpec{Hidden: width, Heads: 2, HeadDim: width / 2, Intermediate: intermediate}, 1e-6, 4, dtype.BF16, &weights))(t)
	graph.use(rotaryOptions{int8: held}, nil)
	x := graph.b.Input("x", dtype.F32, tensor.MustShape(width, 4))
	graph.projection("layers.0.proj.q", x, width, width)
	graph.projection("layers.0.proj.o", x, width, width)
	testcheck.NoError(t, graph.b.Err())
	for name, want := range map[string]dtype.Type{"layers.0.proj.q.weight": dtype.I8ConvRot, "layers.0.proj.o.weight": dtype.BF16} {
		if node := weights.Node(name); node == nil || node.Type != want {
			t.Errorf("the graph holds %s as %v; want %s", name, node, want)
		}
	}
}
