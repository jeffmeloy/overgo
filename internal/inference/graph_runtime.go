package inference

import (
	"context"

	"overgo/internal/gguf"
	"overgo/internal/graphruntime"
	"overgo/internal/model"
	"overgo/internal/modeldevice"
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
	host    map[*tensor.Tensor]reference.Value
}

func (r *Runner) newInferenceGraphRuntime(ctx context.Context) *inferenceGraphRuntime {
	return &inferenceGraphRuntime{
		runner: r, ctx: ctx, builder: r.newGraphBuilder(), feeds: graphruntime.NewFeeds(ctx, r.cuda),
		host: make(map[*tensor.Tensor]reference.Value),
	}
}

func (runtime *inferenceGraphRuntime) input(name string, value reference.Value) *tensor.Tensor {
	return runtime.feeds.Input(runtime.builder, name, value)
}

func (runtime *inferenceGraphRuntime) weight(info gguf.TensorInfo) (*tensor.Tensor, error) {
	if runtime.runner.hasPreloadedWeights() {
		node, pointer, err := runtime.runner.deviceInput(runtime.builder, info)
		if err != nil {
			return nil, err
		}
		runtime.feeds.SetDevice(node, pointer)
		return node, nil
	}
	value, err := runtime.runner.hostTensor(runtime.ctx, info)
	if err != nil {
		return nil, err
	}
	node := runtime.builder.Input(info.Name, dtype.F32, value.Shape)
	runtime.feeds.SetHost(node, value)
	return node, nil
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
	if runtime.runner.hasPreloadedWeights() {
		weights, feeds, err := modeldevice.BindDeviceLayerGraphInputs(
			runtime.builder, layer, runtime.runner.deviceInput,
		)
		if err == nil {
			runtime.feeds.AddDevice(feeds)
		}
		return weights, err
	}
	if hostLayer == nil {
		loaded, err := runtime.runner.hostLayer(runtime.ctx, prefix, layer)
		if err != nil {
			return model.LayerGraphWeights{}, err
		}
		hostLayer = &loaded
	}
	weights, feeds, err := hostLayer.GraphInputs(runtime.builder, prefix)
	if err == nil {
		runtime.feeds.AddHost(feeds)
	}
	return weights, err
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
	for node, value := range runtime.host {
		runtime.feeds.SetHost(node, value)
	}
	return runtime.feeds.Execute(outputs...)
}
