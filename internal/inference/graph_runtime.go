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

func (r *Runner) deviceOrHostTensor(
	ctx context.Context,
	builder *tensor.Builder,
	info gguf.TensorInfo,
	hostFeeds map[*tensor.Tensor]reference.Value,
) (*tensor.Tensor, driver.DevicePtr, error) {
	if r.hasPreloadedWeights() {
		return r.deviceInput(builder, info)
	}
	value, err := model.LoadHostTensor(ctx, r.file, info)
	if err != nil {
		return nil, 0, err
	}
	item := builder.Input(info.Name, dtype.F32, value.Shape)
	hostFeeds[item] = value
	return item, 0, nil
}

func (runtime *inferenceGraphRuntime) layer(
	layer model.LayerWeights,
	prefix string,
) (model.LayerGraphWeights, error) {
	return runtime.layerWithHost(layer, prefix, nil)
}

func (runtime *inferenceGraphRuntime) layerWithHost(
	layer model.LayerWeights,
	prefix string,
	hostLayer *model.HostLayer,
) (model.LayerGraphWeights, error) {
	if hostLayer != nil && !runtime.runner.hasPreloadedWeights() {
		weights, feeds, err := hostLayer.GraphInputs(runtime.builder, prefix)
		if err != nil {
			return model.LayerGraphWeights{}, err
		}
		for node, value := range feeds {
			runtime.hostFeeds[node] = value
		}
		return weights, nil
	}
	weights, feeds, err := runtime.runner.layerGraphInputs(
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

func (r *Runner) layerGraphInputs(
	ctx context.Context,
	builder *tensor.Builder,
	hostFeeds map[*tensor.Tensor]reference.Value,
	layer model.LayerWeights,
	prefix string,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	if r.hasPreloadedWeights() {
		return r.layerDeviceInputs(builder, layer)
	}
	hostLayer, err := model.LoadHostLayer(ctx, r.file, layer)
	if err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	graph, feeds, err := hostLayer.GraphInputs(builder, prefix)
	if err != nil {
		return model.LayerGraphWeights{}, nil, err
	}
	for node, value := range feeds {
		hostFeeds[node] = value
	}
	return graph, map[*tensor.Tensor]driver.DevicePtr{}, nil
}

func (runtime *inferenceGraphRuntime) execute(outputs ...*tensor.Tensor) (map[*tensor.Tensor]reference.Value, error) {
	if runtime.runner.hasPreloadedWeights() {
		return runtime.runner.cuda.ExecuteWithDeviceFeeds(
			runtime.ctx, outputs, runtime.hostFeeds, runtime.deviceFeeds,
		)
	}
	return runtime.runner.cuda.Execute(runtime.ctx, outputs, runtime.hostFeeds)
}
