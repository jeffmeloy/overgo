//go:build windows

package routedlm

import (
	"context"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/testcheck"
)

func TestSenseNovaGenerationLayerGraph(t *testing.T) {
	cudatest.Require(t)
	cfg := generationGraphConfig()
	rope := testcheck.Must(newRopePlan(cfg.HeadDim, []RopeSection{
		{Width: 4, Theta: 5e6, Axis: AxisTime},
		{Width: 2, Theta: 1e4, Axis: AxisHeight},
		{Width: 2, Theta: 1e4, Axis: AxisWidth},
	}))(t)
	graph := testcheck.Must(buildDeviceGenerationLayer(cfg, rope, 3, 2, 2, 11, dtype.BF16))(t)
	referenceGraph := testcheck.Must(buildDeviceGenerationLayer(cfg, rope, 3, 2, 2, 11, dtype.F32))(t)
	indexed := testcheck.Must(executor.CompileIndexed(graph.Output))(t)
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	cuda := testcheck.Must(executor.NewWithWorker(worker))(t)
	defer cuda.Close()
	feeds := make(map[*tensor.Tensor]driver.DevicePtr)
	referenceFeeds := make(map[*tensor.Tensor]reference.Value)
	var allocations []driver.DevicePtr
	defer func() {
		_ = worker.Do(t.Context(), func(state *device.State) error {
			for _, pointer := range allocations {
				_ = state.Driver.MemFree(pointer)
			}
			return nil
		})
	}()
	productionNodes, referenceNodes := generationGraphNodes(graph), generationGraphNodes(referenceGraph)
	for nodeIndex, node := range productionNodes {
		elements, _ := node.Shape.Elements()
		values := make([]float32, elements)
		for index := range values {
			values[index] = 0.2 * float32(index%17-8) / 17
		}
		var raw []byte
		if node.Type == dtype.BF16 {
			encoded := make([]uint16, len(values))
			for index, value := range values {
				encoded[index] = dtype.Float32ToBF16(value)
			}
			raw = driver.Bytes(encoded)
			for index, value := range values {
				values[index] = dtype.RoundBF16(value)
			}
		} else {
			raw = driver.Bytes(values)
		}
		referenceNode := referenceNodes[nodeIndex]
		referenceFeeds[referenceNode] = reference.Value{Shape: referenceNode.Shape, Data: values}
		err := worker.Do(t.Context(), func(state *device.State) error {
			pointer, err := state.Driver.MemAlloc(uint64(len(raw)))
			if err != nil {
				return err
			}
			allocations = append(allocations, pointer)
			feeds[node] = pointer
			return state.Driver.MemcpyHtoD(pointer, raw)
		})
		testcheck.NoError(t, err)
	}
	for node, pointer := range feeds {
		testcheck.NoError(t, indexed.Inputs.Set(node, pointer))
	}
	result := testcheck.Must(cuda.ExecuteCompiled(context.WithoutCancel(t.Context()), indexed.Graph, nil, indexed.Inputs))(t)
	got := result[graph.Output].Data
	wantResult := testcheck.Must(reference.Execute([]*tensor.Tensor{referenceGraph.Output}, referenceFeeds))(t)
	want := wantResult[referenceGraph.Output].Data
	if len(got) != cfg.HiddenSize*graph.Tokens {
		t.Fatalf("generation output elements=%d", len(got))
	}
	energy := 0.0
	worst := 0.0
	for index, value := range got {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("generation output[%d]=%g", index, value)
		}
		energy += float64(value * value)
		worst = max(worst, math.Abs(float64(value-want[index])))
	}
	if energy == 0 {
		t.Fatal("generation output is zero")
	}
	if worst > 0.002 {
		t.Fatalf("generation CUDA/reference worst delta=%.3e", worst)
	}
	t.Logf("neutral SenseNova generation layer: prefix=%d image=%dx%d hidden=%d rms=%.6f CUDA/reference=%.3e",
		graph.PrefixTokens, 2, 2, cfg.HiddenSize, math.Sqrt(energy/float64(len(got))), worst)
}
