package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// hostBridgeOperation finds an operation among the outputs' inputs that the CUDA executor computes on the host.
//   - each such operation waits for the device and copies its inputs to the host, once a token in decode
func hostBridgeOperation(outputs ...*tensor.Tensor) (tensor.Op, bool) {
	seen := map[*tensor.Tensor]bool{}
	pending := append([]*tensor.Tensor(nil), outputs...)
	for len(pending) > 0 {
		node := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		if node == nil || seen[node] {
			continue
		}
		seen[node] = true
		switch node.Op {
		case tensor.OpHyperConnectionInit, tensor.OpHyperConnectionPre, tensor.OpHyperConnectionPost,
			tensor.OpHyperConnectionHead, tensor.OpCompressedAttention:
			return node.Op, true
		}
		pending = append(pending, node.Inputs...)
	}
	return 0, false
}

// TestFlashNextGatedBlocks compares loaded GGUF-layout weights with the original
// Python model's FP32 forward and continuation state, including distinct streams,
// and holds the block to operations the device computes itself.
func TestFlashNextGatedBlocks(t *testing.T) {
	data, err := os.ReadFile("testdata/flashnext_gated.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Tolerance float64
		Weights   map[string]struct {
			Shape []uint64
			Data  []float32
		}
		Cases []struct {
			Name                       string
			Start, Tokens              uint32
			Input, Output, Conv, State []float32
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	spec := qwen35TestSpec()
	spec.Architecture, spec.BlockCount, spec.VocabularySize = "qwen4exp", 1, 32
	spec.HyperConnectionCount, spec.HyperConnectionRank = 4, 3
	spec.SSMGroupCount, spec.SSMTimeStepRank, spec.SSMInnerSize = 2, 4, 8
	spec.ExpertCount, spec.ExpertUsedCount, spec.ExpertFeedForward, spec.SharedExpertFF = 3, 2, 6, 4
	spec.ExpertWeightsScale, spec.ExpertWeightsNorm = 1, true
	spec = bindFixtureSpec(spec)
	file := &gguf.File{Tensors: []gguf.TensorInfo{tensorInfo("token_embd.weight", 8, 32)}}
	for name, value := range fixture.Weights {
		file.Tensors = append(file.Tensors, tensorInfo(name, value.Shape...))
	}
	inventory, err := ReadWeights(file, spec)
	if err != nil {
		t.Fatal(err)
	}
	var conv, state reference.Value
	for _, test := range fixture.Cases {
		t.Run(fmt.Sprintf("%s/%d", test.Name, test.Start), func(t *testing.T) {
			builder := tensor.NewBuilder()
			feeds := map[*tensor.Tensor]reference.Value{}
			input := func(name string, value reference.Value) *tensor.Tensor {
				node := builder.Input(name, dtype.F32, value.Shape)
				feeds[node] = value
				return node
			}
			weights := LayerGraphWeights{Recurrent: true}
			if err := BindLayerGraphInventory(&inventory.Layers[0], &weights, func(_ LayerBindingSlot, info *gguf.TensorInfo) (*tensor.Tensor, error) {
				value := fixture.Weights[info.Name]
				return input(info.Name, reference.Value{Shape: tensor.MustShape(value.Shape...), Data: value.Data}), nil
			}); err != nil {
				t.Fatal(err)
			}
			if test.Start == 0 {
				conv = reference.ZeroValue(tensor.MustShape(2, 16))
				state = reference.ZeroValue(tensor.MustShape(2, 2, 4, 1))
			}
			shape := tensor.MustShape(8, uint64(test.Tokens))
			if test.Name == "distinct-streams" {
				shape = tensor.MustShape(8, 4, uint64(test.Tokens))
			}
			positions := make([]uint32, test.Tokens)
			for i := range positions {
				positions[i] = test.Start + uint32(i)
			}
			plan := spec.PlanLayer(0, true)
			result, err := executeCompiledLayer(blockDispatchOptions{
				Spec: spec, Weights: weights, Plan: &plan,
				Context: CachedBlockContext{
					Builder: builder, Input: input("input", reference.Value{Shape: shape, Data: test.Input}),
					Positions: positions, Recurrent: true, Sequences: 1,
					PastKey: input("conv", conv), PastValue: input("state", state),
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			if op, found := hostBridgeOperation(result.Output, result.Key, result.Value); found {
				t.Fatalf("the gated block holds %s, which the device runs through the host bridge", op)
			}
			values, err := reference.Execute([]*tensor.Tensor{result.Output, result.Key, result.Value}, feeds)
			if err != nil {
				t.Fatal(err)
			}
			for _, check := range []struct {
				name string
				node *tensor.Tensor
				want []float32
			}{
				{"output", result.Output, test.Output}, {"convolution", result.Key, test.Conv}, {"recurrent", result.Value, test.State},
			} {
				got := values[check.node].Data
				if len(got) != len(check.want) {
					t.Fatalf("%s length %d, want %d", check.name, len(got), len(check.want))
				}
				for i, want := range check.want {
					difference := math.Abs(float64(got[i] - want))
					if math.IsNaN(difference) || difference > fixture.Tolerance*(1+math.Abs(float64(want))) {
						t.Fatalf("%s[%d] = %.9g, original %.9g", check.name, i, got[i], want)
					}
				}
			}
			conv, state = values[result.Key], values[result.Value]
		})
	}
}
