package inference

import (
	"context"

	"llamacpp2go/internal/cuda/driver"
	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
	"llamacpp2go/internal/tensor/reference"
)

// inferenceGraphRuntime: host/device graph bindings
type inferenceGraphRuntime struct {
	runner      *Runner
	ctx         context.Context
	builder     *tensor.Builder
	hostFeeds   map[*tensor.Tensor]reference.Value
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr
}

func (r *Runner) newInferenceGraphRuntime(ctx context.Context) *inferenceGraphRuntime {
	return &inferenceGraphRuntime{
		runner: r, ctx: ctx, builder: r.newGraphBuilder(),
		hostFeeds:   make(map[*tensor.Tensor]reference.Value),
		deviceFeeds: make(map[*tensor.Tensor]driver.DevicePtr),
	}
}

func (runtime *inferenceGraphRuntime) input(name string, value reference.Value) *tensor.Tensor {
	node := runtime.builder.Input(name, dtype.F32, value.Shape)
	runtime.hostFeeds[node] = value
	return node
}

func (runtime *inferenceGraphRuntime) weight(info gguf.TensorInfo) (*tensor.Tensor, error) {
	node, pointer, err := runtime.runner.deviceOrHostTensor(
		runtime.ctx, runtime.builder, info, runtime.hostFeeds,
	)
	if err != nil {
		return nil, err
	}
	if pointer != 0 {
		runtime.deviceFeeds[node] = pointer
	}
	return node, nil
}

func (runtime *inferenceGraphRuntime) layer(
	layer model.LayerWeights,
	prefix string,
) (model.LayerGraphWeights, error) {
	weights, feeds, err := runtime.runner.mtpLayerInputs(
		runtime.ctx, runtime.builder, runtime.hostFeeds, layer, prefix,
	)
	if err != nil {
		return model.LayerGraphWeights{}, err
	}
	for node, pointer := range feeds {
		runtime.deviceFeeds[node] = pointer
	}
	return weights, nil
}

func (runtime *inferenceGraphRuntime) execute(outputs ...*tensor.Tensor) (map[*tensor.Tensor]reference.Value, error) {
	if runtime.runner.hasPreloadedWeights() {
		return runtime.runner.cuda.ExecuteWithDeviceFeeds(
			runtime.ctx, outputs, runtime.hostFeeds, runtime.deviceFeeds,
		)
	}
	return runtime.runner.cuda.Execute(runtime.ctx, outputs, runtime.hostFeeds)
}
