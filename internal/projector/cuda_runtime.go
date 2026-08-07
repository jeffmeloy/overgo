package projector

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/cuda/executor"
	"overgo/internal/gguf"
	"overgo/internal/graphruntime"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

type projectorCUDA struct {
	worker   *device.Worker
	executor *executor.Executor
	weights  *model.DeviceF32Weights
}

func openProjectorCUDA(
	ctx context.Context,
	file *gguf.File,
	required, optional []string,
	ordinal int,
) (*projectorCUDA, error) {
	worker, err := device.New(ordinal)
	if err != nil {
		return nil, err
	}
	state := &projectorCUDA{worker: worker}
	fail := func(cause error) (*projectorCUDA, error) {
		_ = state.Close()
		return nil, cause
	}
	state.executor, err = executor.NewWithWorker(worker)
	if err != nil {
		return fail(err)
	}
	state.weights, err = model.NewDeviceF32Weights(worker)
	if err != nil {
		return fail(err)
	}
	infos, err := projectorTensorInfos(file, required, optional)
	if err != nil {
		return fail(err)
	}
	if err := state.weights.Load(ctx, file, infos); err != nil {
		return fail(err)
	}
	return state, nil
}

func projectorTensorInfos(file *gguf.File, required, optional []string) ([]gguf.TensorInfo, error) {
	infos := make([]gguf.TensorInfo, 0, len(required)+len(optional))
	included := make(map[string]struct{}, len(required)+len(optional))
	for _, name := range required {
		if _, ok := included[name]; ok {
			continue
		}
		info, ok := file.Tensor(name)
		if !ok {
			return nil, fmt.Errorf("tensor %q is unavailable", name)
		}
		infos = append(infos, info)
		included[name] = struct{}{}
	}
	for _, name := range optional {
		if _, ok := included[name]; ok {
			continue
		}
		if info, ok := file.Tensor(name); ok {
			infos = append(infos, info)
			included[name] = struct{}{}
		}
	}
	return infos, nil
}

func (c *projectorCUDA) Close() error {
	if c == nil {
		return nil
	}
	var closeErrors []error
	if c.weights != nil {
		closeErrors = append(closeErrors, c.weights.Close())
		c.weights = nil
	}
	if c.executor != nil {
		closeErrors = append(closeErrors, c.executor.Close())
		c.executor = nil
	}
	if c.worker != nil {
		closeErrors = append(closeErrors, c.worker.Close())
		c.worker = nil
	}
	return errors.Join(closeErrors...)
}

type projectorCUDAWeights struct {
	runtime *projectorCUDA
	builder *tensor.Builder
	feeds   map[*tensor.Tensor]driver.DevicePtr
	err     error
}

func (c *projectorCUDA) bindWeights(builder *tensor.Builder) *projectorCUDAWeights {
	return &projectorCUDAWeights{
		runtime: c,
		builder: builder,
		feeds:   make(map[*tensor.Tensor]driver.DevicePtr),
	}
}

func (b *projectorCUDAWeights) weight(name string) *tensor.Tensor {
	if b.err != nil {
		return nil
	}
	node, pointer, err := b.runtime.weights.Input(b.builder, name)
	if err != nil {
		b.err = err
		return nil
	}
	b.feeds[node] = pointer
	return node
}

func (b *projectorCUDAWeights) result() (map[*tensor.Tensor]driver.DevicePtr, error) {
	return b.feeds, b.err
}

type projectorGraphRuntime struct {
	ctx     context.Context
	file    *gguf.File
	cuda    *projectorCUDA
	builder *tensor.Builder
	feeds   *graphruntime.Feeds
	// hostFeeds: graph-local inputs
	hostFeeds map[*tensor.Tensor]reference.Value
	binding   *projectorCUDAWeights
	err       error
}

func newProjectorGraphRuntime(
	ctx context.Context,
	file *gguf.File,
	cuda *projectorCUDA,
	builder *tensor.Builder,
) *projectorGraphRuntime {
	feeds := graphruntime.NewFeeds()
	runtime := &projectorGraphRuntime{
		ctx: ctx, file: file, cuda: cuda, builder: builder, feeds: feeds, hostFeeds: feeds.Host,
	}
	if cuda != nil {
		runtime.binding = cuda.bindWeights(builder)
	}
	return runtime
}

func (runtime *projectorGraphRuntime) weight(name string) *tensor.Tensor {
	if runtime.err != nil {
		return nil
	}
	if runtime.binding != nil {
		return runtime.binding.weight(name)
	}
	info, ok := runtime.file.Tensor(name)
	if !ok {
		runtime.err = fmt.Errorf("tensor %q is unavailable", name)
		return nil
	}
	node := runtime.builder.Input(name, dtype.F32, tensorInfoShape(info))
	value, err := model.LoadHostTensor(runtime.ctx, runtime.file, info)
	if err != nil {
		runtime.err = err
		return nil
	}
	runtime.feeds.Host[node] = value
	return node
}

func (runtime *projectorGraphRuntime) addOptionalBias(
	input *tensor.Tensor,
	name string,
) *tensor.Tensor {
	if !hasTensor(runtime.file, name) {
		return input
	}
	return runtime.builder.Add(input, runtime.weight(name))
}

func (runtime *projectorGraphRuntime) linear(
	input *tensor.Tensor,
	prefix string,
) *tensor.Tensor {
	return runtime.builder.Add(
		runtime.builder.MulMat(runtime.weight(prefix+".weight"), input),
		runtime.weight(prefix+".bias"),
	)
}

func (runtime *projectorGraphRuntime) affineNorm(
	input *tensor.Tensor,
	prefix string,
	epsilon float32,
) *tensor.Tensor {
	return runtime.builder.AffineLayerNorm(
		input, runtime.weight(prefix+".weight"), runtime.weight(prefix+".bias"), epsilon,
	)
}

func (runtime *projectorGraphRuntime) weightedRMSNorm(
	input *tensor.Tensor,
	prefix string,
	epsilon float32,
) *tensor.Tensor {
	output := runtime.builder.WeightedRMSNorm(input, runtime.weight(prefix+".weight"), epsilon)
	return runtime.addOptionalBias(output, prefix+".bias")
}

func (runtime *projectorGraphRuntime) execute(outputs ...*tensor.Tensor) (map[*tensor.Tensor]reference.Value, error) {
	if runtime.err != nil {
		return nil, runtime.err
	}
	if err := runtime.builder.Err(); err != nil {
		return nil, err
	}
	device := runtime.binding != nil
	if device {
		deviceFeeds, err := runtime.binding.result()
		if err != nil {
			return nil, err
		}
		runtime.feeds.AddDevice(deviceFeeds)
	}
	return runtime.feeds.Execute(
		outputs, device,
		func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error) {
			return reference.Execute(outputs, feeds)
		},
		func(outputs []*tensor.Tensor, host map[*tensor.Tensor]reference.Value, device map[*tensor.Tensor]driver.DevicePtr) (map[*tensor.Tensor]reference.Value, error) {
			return runtime.cuda.executor.ExecuteWithDeviceFeeds(runtime.ctx, outputs, host, device)
		},
	)
}
