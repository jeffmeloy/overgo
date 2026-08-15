package latentvideo

import (
	"slices"
	"testing"

	"overgo/internal/pytorchzip"
)

func contiguousTensorMeta(name string, shape ...int64) pytorchzip.TensorMeta {
	numel := int64(1)
	stride := make([]int64, len(shape))
	for axis := len(shape) - 1; axis >= 0; axis-- {
		stride[axis] = numel
		numel *= shape[axis]
	}
	return pytorchzip.TensorMeta{
		Name: name, DType: "FloatStorage", StorageKey: name, StorageSize: numel,
		Shape: shape, Stride: stride, Numel: numel,
	}
}

func TestVAEEncoderPlanCompilesInventory(t *testing.T) {
	const (
		inputChannels   = 3
		featureChannels = 2
		qkvChannels     = 3 * featureChannels
		momentChannels  = 2 * featureChannels
		kernelExtent    = 3
		pointExtent     = 1
	)
	meta := contiguousTensorMeta
	inventory := []pytorchzip.TensorMeta{
		meta("encoder.conv1.weight", featureChannels, inputChannels, kernelExtent, kernelExtent, kernelExtent), meta("encoder.conv1.bias", featureChannels),
		meta("encoder.downsamples.0.residual.0.gamma", featureChannels), meta("encoder.downsamples.0.residual.2.weight", featureChannels, featureChannels, kernelExtent, kernelExtent, kernelExtent),
		meta("encoder.downsamples.0.residual.2.bias", featureChannels), meta("encoder.downsamples.0.residual.3.gamma", featureChannels),
		meta("encoder.downsamples.0.residual.6.weight", featureChannels, featureChannels, kernelExtent, kernelExtent, kernelExtent), meta("encoder.downsamples.0.residual.6.bias", featureChannels),
		meta("encoder.middle.0.norm.gamma", featureChannels), meta("encoder.middle.0.to_qkv.weight", qkvChannels, featureChannels, pointExtent, pointExtent),
		meta("encoder.middle.0.to_qkv.bias", qkvChannels), meta("encoder.middle.0.proj.weight", featureChannels, featureChannels, pointExtent, pointExtent), meta("encoder.middle.0.proj.bias", featureChannels),
		meta("encoder.head.0.gamma", featureChannels), meta("encoder.head.2.weight", momentChannels, featureChannels, kernelExtent, kernelExtent, kernelExtent), meta("encoder.head.2.bias", momentChannels),
		meta("conv1.weight", momentChannels, momentChannels, pointExtent, pointExtent, pointExtent), meta("conv1.bias", momentChannels),
	}
	plan, err := CompileVAEEncoderPlan(inventory)
	if err != nil {
		t.Fatal(err)
	}
	wantPrefixes := []string{"encoder.conv1", "encoder.downsamples.0", "encoder.middle.0", "encoder.head", "conv1"}
	if plan.InputChannels != inputChannels || plan.LatentChannels != featureChannels || plan.MomentChannels != momentChannels ||
		plan.UsedTensorCount != len(inventory) || plan.UsedWeightBytes <= 0 ||
		!slices.Equal(plan.OpPrefixes(), wantPrefixes) {
		t.Fatalf("encoder plan = %+v prefixes=%v", plan, plan.OpPrefixes())
	}
}
