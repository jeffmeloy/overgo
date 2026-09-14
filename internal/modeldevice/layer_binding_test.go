package modeldevice

import (
	"testing"

	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestBindDeviceLayerGraphInputsPreservesStorageTypes(t *testing.T) {
	const (
		hiddenWidth   = 32
		feedForwardUp = 64
		normPointer   = driver.DevicePtr(11)
		gatePointer   = driver.DevicePtr(22)
	)
	info := func(name string, storage dtype.Type, dimensions ...uint64) *gguf.TensorInfo {
		result := gguf.TensorInfo{Name: name, Type: storage, Dimensions: uint32(len(dimensions))}
		copy(result.Shape[:], dimensions)
		return &result
	}
	layer := model.LayerWeights{
		AttentionNorm:   info("blk.0.attn_norm.weight", dtype.F32, hiddenWidth),
		FeedForwardGate: info("blk.0.ffn_gate.weight", dtype.Q8_0, hiddenWidth, feedForwardUp),
	}
	pointers := map[string]driver.DevicePtr{
		layer.AttentionNorm.Name:   normPointer,
		layer.FeedForwardGate.Name: gatePointer,
	}
	builder := tensor.NewBuilder()
	graph, feeds, err := BindDeviceLayerGraphInputs(builder, layer, func(
		builder *tensor.Builder,
		info gguf.TensorInfo,
	) (*tensor.Tensor, driver.DevicePtr, error) {
		shape, shapeErr := tensor.NewShape(info.Shape[:info.Dimensions]...)
		if shapeErr != nil {
			return nil, 0, shapeErr
		}
		return builder.Input(info.Name, info.Type, shape), pointers[info.Name], builder.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	if graph.AttentionNorm.Type != dtype.F32 || graph.FeedForwardGate.Type != dtype.Q8_0 {
		t.Fatalf("storage types = %s/%s, want F32/Q8_0", graph.AttentionNorm.Type, graph.FeedForwardGate.Type)
	}
	if len(feeds) != 2 ||
		feeds[0].Node != graph.AttentionNorm || feeds[0].Value != normPointer ||
		feeds[1].Node != graph.FeedForwardGate || feeds[1].Value != gatePointer {
		t.Fatalf("device feeds = %v", feeds)
	}
}
