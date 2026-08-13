package inference

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

func (r *Runner) hasPreloadedWeights() bool {
	return r != nil && r.deviceWeights != nil
}

func (r *Runner) deviceInput(
	builder *tensor.Builder,
	info gguf.TensorInfo,
) (*tensor.Tensor, driver.DevicePtr, error) {
	if r.rawWeights != nil {
		if _, ok := r.rawWeights.Lookup(info.Name); ok {
			return r.rawWeights.Input(builder, info.Name)
		}
	}
	if r.deviceWeights != nil {
		return r.deviceWeights.Input(builder, info.Name)
	}
	return nil, 0, fmt.Errorf("inference: device tensor %q is not preloaded", info.Name)
}

func (r *Runner) decodeDeviceInput(
	builder *tensor.Builder,
	info gguf.TensorInfo,
) (*tensor.Tensor, driver.DevicePtr, error) {
	if r.decodeWeights != nil {
		if _, ok := r.decodeWeights.Lookup(info.Name); ok {
			return r.decodeWeights.Input(builder, info.Name)
		}
	}
	return r.deviceInput(builder, info)
}

func (r *Runner) wavTokenizerGraphInputs(
	ctx context.Context,
	builder *tensor.Builder,
) (model.SequenceOutputGraphWeights, map[*tensor.Tensor]reference.Value, map[*tensor.Tensor]driver.DevicePtr, error) {
	var result model.SequenceOutputGraphWeights
	hostFeeds := make(map[*tensor.Tensor]reference.Value)
	deviceFeeds := make(map[*tensor.Tensor]driver.DevicePtr)
	if r.weights.AudioDecoder == nil {
		return result, nil, nil, errors.New("inference: AudioDecoder weights are missing")
	}
	input := func(info gguf.TensorInfo) (*tensor.Tensor, error) {
		if r.hasPreloadedWeights() {
			node, pointer, err := r.deviceInput(builder, info)
			if err != nil {
				return nil, err
			}
			deviceFeeds[node] = pointer
			return node, nil
		}
		value, err := r.hostTensor(ctx, info)
		if err != nil {
			return nil, err
		}
		node := builder.Input(info.Name, dtype.F32, value.Shape)
		hostFeeds[node] = value
		return node, nil
	}
	assign := func(destination **tensor.Tensor, info gguf.TensorInfo) error {
		item, err := input(info)
		if err == nil {
			*destination = item
		}
		return err
	}
	info := r.weights.AudioDecoder
	for _, item := range []struct {
		destination **tensor.Tensor
		info        gguf.TensorInfo
	}{
		{&result.InputConv, info.InputConv}, {&result.InputConvBias, info.InputConvBias},
		{&result.TokenNorm, info.TokenNorm}, {&result.TokenNormBias, info.TokenNormBias},
		{&result.OutputNorm, info.OutputNorm}, {&result.OutputNormBias, info.OutputNormBias},
		{&result.Output, info.Output}, {&result.OutputBias, info.OutputBias},
	} {
		if err := assign(item.destination, item.info); err != nil {
			return result, nil, nil, err
		}
	}
	result.Residual = make([]model.SequenceResidualGraphWeights, len(info.PosNet))
	for block := range info.PosNet {
		source, destination := &info.PosNet[block], &result.Residual[block]
		for _, item := range []struct {
			destination **tensor.Tensor
			info        gguf.TensorInfo
		}{
			{&destination.Norm1, source.Norm1}, {&destination.Norm1Bias, source.Norm1Bias},
			{&destination.Conv1, source.Conv1}, {&destination.Conv1Bias, source.Conv1Bias},
			{&destination.Norm2, source.Norm2}, {&destination.Norm2Bias, source.Norm2Bias},
			{&destination.Conv2, source.Conv2}, {&destination.Conv2Bias, source.Conv2Bias},
			{&destination.AttentionNorm, source.AttentionNorm}, {&destination.AttentionNormBias, source.AttentionNormBias},
			{&destination.AttentionQ, source.AttentionQ}, {&destination.AttentionQBias, source.AttentionQBias},
			{&destination.AttentionK, source.AttentionK}, {&destination.AttentionKBias, source.AttentionKBias},
			{&destination.AttentionV, source.AttentionV}, {&destination.AttentionVBias, source.AttentionVBias},
			{&destination.AttentionOutput, source.AttentionOutput}, {&destination.AttentionOutBias, source.AttentionOutBias},
		} {
			if item.info.Name != "" {
				if err := assign(item.destination, item.info); err != nil {
					return result, nil, nil, err
				}
			}
		}
	}
	result.Convolution = make([]model.SequenceConvGraphWeights, len(info.ConvNext))
	for block := range info.ConvNext {
		source, destination := &info.ConvNext[block], &result.Convolution[block]
		for _, item := range []struct {
			destination **tensor.Tensor
			info        gguf.TensorInfo
		}{
			{&destination.Depthwise, source.Depthwise}, {&destination.DepthwiseBias, source.DepthwiseBias},
			{&destination.Norm, source.Norm}, {&destination.NormBias, source.NormBias},
			{&destination.Pointwise1, source.Pointwise1}, {&destination.Pointwise1Bias, source.Pointwise1Bias},
			{&destination.Pointwise2, source.Pointwise2}, {&destination.Pointwise2Bias, source.Pointwise2Bias},
			{&destination.Gamma, source.Gamma},
		} {
			if err := assign(item.destination, item.info); err != nil {
				return result, nil, nil, err
			}
		}
	}
	return result, hostFeeds, deviceFeeds, nil
}

func (r *Runner) applyDeviceOutputNorm(
	builder *tensor.Builder,
	input *tensor.Tensor,
	deviceFeeds map[*tensor.Tensor]driver.DevicePtr,
) (*tensor.Tensor, error) {
	return r.buildOutputNorm(builder, input, func(info gguf.TensorInfo) (*tensor.Tensor, error) {
		node, pointer, err := r.deviceInput(builder, info)
		if err == nil {
			deviceFeeds[node] = pointer
		}
		return node, err
	})
}

func (r *Runner) buildOutputNorm(
	builder *tensor.Builder,
	input *tensor.Tensor,
	bind func(gguf.TensorInfo) (*tensor.Tensor, error),
) (*tensor.Tensor, error) {
	if r.program.Model.Terminal().Normalization == model.OutputNormAbsent {
		return input, nil
	}
	if r.spec.UsesUnweightedLayerNorm() {
		return builder.LayerNorm(input, r.spec.LayerNormEpsilon), builder.Err()
	}
	if r.spec.UsesUnweightedRMSNorm() {
		return builder.RMSNorm(input, r.spec.RMSNormEpsilon), builder.Err()
	}
	weight, err := bind(r.weights.OutputNorm)
	if err != nil {
		return nil, err
	}
	var bias *tensor.Tensor
	if r.weights.OutputNormBias != nil {
		bias, err = bind(*r.weights.OutputNormBias)
		if err != nil {
			return nil, err
		}
	}
	return model.ApplyNormalization(builder, input, weight, bias, r.spec), builder.Err()
}

func (r *Runner) layerDeviceInputs(
	builder *tensor.Builder,
	info model.LayerWeights,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	if r.deviceWeights == nil {
		return model.LayerGraphWeights{}, nil, errors.New("inference: device weights are unavailable")
	}
	return model.BindDeviceLayerGraphInputs(builder, info, r.deviceInput)
}

func (r *Runner) layerDecodeDeviceInputs(
	builder *tensor.Builder,
	info model.LayerWeights,
) (model.LayerGraphWeights, map[*tensor.Tensor]driver.DevicePtr, error) {
	return model.BindDeviceLayerGraphInputs(builder, info, r.decodeDeviceInput)
}
