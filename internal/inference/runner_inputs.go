package inference

import (
	"fmt"

	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/tensor"
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
	normalization := r.program.Model.Normalization()
	if normalization.Operation == model.NormalizationUnweightedLayer {
		return builder.LayerNorm(input, r.spec.LayerNormEpsilon), builder.Err()
	}
	if normalization.Operation == model.NormalizationUnweightedRMS {
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
	return r.program.Model.Normalization().Apply(builder, input, weight, bias), builder.Err()
}
