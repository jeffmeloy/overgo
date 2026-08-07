package inference

import (
	"context"

	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/graphruntime"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// inferenceGraphRuntime: host/device graph bindings
type inferenceGraphRuntime struct {
	runner  *Runner
	ctx     context.Context
	builder *tensor.Builder
	feeds   *graphruntime.Feeds
}

func (r *Runner) newInferenceGraphRuntime(ctx context.Context) *inferenceGraphRuntime {
	return &inferenceGraphRuntime{
		runner: r, ctx: ctx, builder: r.newGraphBuilder(), feeds: graphruntime.NewFeeds(),
	}
}

func (runtime *inferenceGraphRuntime) input(name string, value reference.Value) *tensor.Tensor {
	return runtime.feeds.Input(runtime.builder, name, value)
}

func (runtime *inferenceGraphRuntime) addHostFeeds(feeds map[*tensor.Tensor]reference.Value) {
	runtime.feeds.AddHost(feeds)
}

func (runtime *inferenceGraphRuntime) addDeviceFeeds(feeds map[*tensor.Tensor]driver.DevicePtr) {
	runtime.feeds.AddDevice(feeds)
}

func (runtime *inferenceGraphRuntime) weight(info gguf.TensorInfo) (*tensor.Tensor, error) {
	node, pointer, err := runtime.runner.deviceOrHostTensor(
		runtime.ctx, runtime.builder, info, runtime.feeds.Host,
	)
	if err != nil {
		return nil, err
	}
	if pointer != 0 {
		runtime.feeds.Device[node] = pointer
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
	value, err := r.hostTensor(ctx, info)
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
			runtime.feeds.Host[node] = value
		}
		return weights, nil
	}
	weights, feeds, err := runtime.runner.layerGraphInputs(
		runtime.ctx, runtime.builder, runtime.feeds.Host, layer, prefix,
	)
	if err != nil {
		return model.LayerGraphWeights{}, err
	}
	for node, pointer := range feeds {
		runtime.feeds.Device[node] = pointer
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
	hostLayer, err := r.hostLayer(ctx, prefix, layer)
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

func (r *Runner) hostTensor(ctx context.Context, info gguf.TensorInfo) (reference.Value, error) {
	if r.hostWeights != nil {
		return r.hostWeights.Load(ctx, r.file, info)
	}
	return model.LoadHostTensor(ctx, r.file, info)
}

func (r *Runner) hostLayer(ctx context.Context, key string, info model.LayerWeights) (model.HostLayer, error) {
	if r.hostWeights != nil {
		return r.hostWeights.LoadLayer(ctx, r.file, key, info)
	}
	return model.LoadHostLayer(ctx, r.file, info)
}

func (runtime *inferenceGraphRuntime) execute(outputs ...*tensor.Tensor) (map[*tensor.Tensor]reference.Value, error) {
	return runtime.feeds.Execute(
		outputs, runtime.runner.hasPreloadedWeights(),
		func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
			return runtime.runner.cuda.Execute(runtime.ctx, outputs, feeds)
		},
		func(outputs []*tensor.Tensor, host map[*tensor.Tensor]reference.Value, device map[*tensor.Tensor]driver.DevicePtr) (map[*tensor.Tensor]reference.Value, error) {
			return runtime.runner.cuda.ExecuteWithDeviceFeeds(runtime.ctx, outputs, host, device)
		},
	)
}
