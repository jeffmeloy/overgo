//go:build windows

package inference

import (
	"bytes"
	"fmt"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/gguf"
	"overgo/internal/modeldevice"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

// TestAPromptReadCopiesTheNextGroupsExpertsAhead reads a 1,024-token prompt through two routed layers whose BF16
// experts lie in host memory: once the read before found experts there, the first layer copies the experts the second
// routed to more slots than the CPU takes at that read before the second's routes are read (prefill-16k/copy-waits), as
// Strata's prompt path streams a chunk's experts from 1,024 tokens, and either way the output is the bits of the same
// read with every expert on the card, where no read takes the halves. Each token routes to two fixed experts, so every
// expert takes as many slots as the next and none is the CPU's.
func TestAPromptReadCopiesTheNextGroupsExpertsAhead(t *testing.T) {
	cudatest.Require(t)
	ctx := t.Context()
	const hidden, intermediate, experts, topK, tokens, layers = 256, 256, 8, 2, 1024, 2
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	cuda := testcheck.Must(executor.NewWithWorker(worker))(t)
	defer cuda.Close()
	value := func(seed, index int) float32 { return float32((index*7+seed*13)%23-11) / 64 }
	// read builds the layers over a residency of limit card bytes, runs them once, and answers the output's bits and the
	// experts the run copied ahead.
	read := func(limit uint64) ([]byte, uint64) {
		t.Helper()
		builder := tensor.NewBuilder()
		builder.SetMulMatCompute(tensor.MulMatComputeNativeTensorCore)
		host := map[*tensor.Tensor]reference.Value{}
		feed := func(name string, seed int, dims ...uint64) *tensor.Tensor {
			node := builder.Input(name, dtype.F32, tensor.MustShape(dims...))
			elements, _ := node.Shape.Elements()
			data := make([]float32, elements)
			for index := range data {
				data[index] = value(seed, index)
			}
			host[node] = reference.Value{Shape: node.Shape, Data: data}
			return node
		}
		var banks []*tensor.Tensor
		var encoded []gguf.TensorData
		bank := func(name string, seed int, dims ...uint64) *tensor.Tensor {
			node := builder.Input(name, dtype.BF16, tensor.MustShape(dims...))
			elements, _ := node.Shape.Elements()
			bits := make([]uint16, elements)
			for index := range bits {
				bits[index] = dtype.Float32ToBF16(value(seed, index) / 16)
			}
			encoded = append(encoded, gguf.TensorData{Name: name, Shape: dims, Type: dtype.BF16, Data: bytes.NewReader(driver.Bytes(bits))})
			banks = append(banks, node)
			return node
		}
		selected := builder.Input("selected", dtype.F32, tensor.MustShape(topK, tokens))
		choices := make([]float32, topK*tokens)
		for index := range choices {
			// The token's slots take experts three apart, three coprime to the experts.
			choices[index] = float32((index/topK + index%topK*3) % experts)
		}
		host[selected] = reference.Value{Shape: selected.Shape, Data: choices}
		output := feed("input", 0, hidden, tokens)
		for layer := range layers {
			name := func(part string) string { return fmt.Sprintf("blk.%d.ffn_%s_exps.weight", layer, part) }
			output = builder.MoEWithOptions(output, feed(fmt.Sprintf("blk.%d.router", layer), layer+1, hidden, experts),
				bank(name("up"), 3*layer+2, hidden, intermediate, experts), bank(name("down"), 3*layer+3, intermediate, hidden, experts),
				tensor.MoEOptions{TopK: topK, NormalizeTopKProb: true, Scale: 1, ExpertIndexDivisor: 1, Routing: tensor.MoERoutingSoftmax,
					Activation: tensor.MoEActivationSiLU, Gate: bank(name("gate"), 3*layer+4, hidden, intermediate, experts),
					SelectedExperts: selected})
		}
		testcheck.NoError(t, builder.Err())
		var file bytes.Buffer
		testcheck.NoError(t, gguf.Write(&file, nil, encoded, gguf.WriteOptions{}))
		parsed := testcheck.Must(gguf.Parse(bytes.NewReader(file.Bytes()), uint64(file.Len()), gguf.DefaultOptions()))(t)
		weights := testcheck.Must(modeldevice.NewDeviceWeights(worker))(t)
		defer func() {
			if err := cuda.SetExpertBanks(nil); err != nil {
				t.Error(err)
			}
			if err := weights.Close(); err != nil {
				t.Error(err)
			}
		}()
		loader := modeldevice.NewLoader(parsed)
		loader.Experts(weights, parsed.Tensors, limit)
		testcheck.NoError(t, loader.Load(ctx, nil))
		testcheck.NoError(t, cuda.SetExpertBanks(weights))
		graph := testcheck.Must(executor.CompileIndexed(output))(t)
		var bindings []tensor.InputBinding[driver.DevicePtr]
		for _, node := range banks {
			_, pointer, err := weights.Input(builder, node.Name)
			testcheck.NoError(t, err)
			bindings = append(bindings, tensor.InputBinding[driver.DevicePtr]{Node: node, Value: pointer})
		}
		testcheck.NoError(t, graph.Inputs.Bind(bindings))
		before := testcheck.Must(cuda.Metrics(ctx))(t)
		result := testcheck.Must(cuda.ExecuteCompiled(ctx, graph.Graph, host, graph.Inputs))(t)
		after := testcheck.Must(cuda.Metrics(ctx))(t)
		return slices.Clone(driver.Bytes(result[output].Data)), after.PrefetchedExperts - before.PrefetchedExperts
	}
	// The card holds every expert under no limit, and none beyond the banks' tables of a pointer an expert.
	const pointerBytes = 8
	card, cardAhead := read(0)
	first, firstAhead := read(layers * 3 * experts * pointerBytes)
	second, secondAhead := read(layers * 3 * experts * pointerBytes)
	if cardAhead != 0 || firstAhead != 0 || secondAhead != experts {
		t.Fatalf("reads copied %d, %d and %d experts ahead, with every expert on the card and twice with every one in host memory, want 0, 0 and the second layer's %d",
			cardAhead, firstAhead, secondAhead, experts)
	}
	if !bytes.Equal(first, card) || !bytes.Equal(second, card) {
		t.Fatal("the experts copied after their routes or ahead of them changed the card read's bits")
	}
}
