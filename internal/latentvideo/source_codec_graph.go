package latentvideo

import (
	"fmt"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/pytorchzip"
	"overgo/internal/tensor"
)

// VAEEncoderPlan: checkpoint-derived causal source encoder graph.
type VAEEncoderPlan struct {
	vaePlanCore
	InputChannels  int
	LatentChannels int
	MomentChannels int
}

// CompileVAEEncoderPlan derives the Wan/LiveEdit encoder from artifact tensors.
func CompileVAEEncoderPlan(metas []pytorchzip.TensorMeta) (VAEEncoderPlan, error) {
	compiler := newVAEPlanCompiler("vae encoder", metas)
	conv3 := func(name string) (out, in, kt, kh, kw int, err error) {
		metadata, err := compiler.tensorMetadata(name)
		if err != nil {
			return 0, 0, 0, 0, 0, err
		}
		return metadata.conv3d("encoder")
	}
	conv2 := func(name string) (out, in, kh, kw int, err error) {
		metadata, err := compiler.tensorMetadata(name)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		return metadata.conv2d("encoder")
	}

	plan := VAEEncoderPlan{vaePlanCore: vaePlanCore{Stride: [tensor.TripleExtent]int{
		tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent,
	}}}
	out, in, kt, kh, kw, err := conv3("encoder.conv1.weight")
	if err != nil || !checked.PositiveInts(in, out) || !media.Convolution3DMatches(out, in, kt, kh, kw, out, in, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent) {
		return plan, fmt.Errorf("vae encoder: invalid encoder.conv1.weight: %w", err)
	}
	if err := compiler.vector("encoder.conv1.bias", out); err != nil {
		return plan, err
	}
	plan.InputChannels = in
	compiler.add(media.CodecConvolution, "encoder.conv1", in, out, "encoder.conv1.weight", "encoder.conv1.bias")
	channels := out

	compileResidual := func(prefix string) error {
		residualOut, residualIn, kt, kh, kw, err := conv3(prefix + ".residual.2.weight")
		if err != nil || !checked.PositiveInts(residualOut) || !media.Convolution3DMatches(residualOut, residualIn, kt, kh, kw, residualOut, channels, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent) {
			return fmt.Errorf("vae encoder: invalid %s first residual: %w", prefix, err)
		}
		secondOut, secondIn, kt, kh, kw, err := conv3(prefix + ".residual.6.weight")
		if err != nil || !media.Convolution3DMatches(secondOut, secondIn, kt, kh, kw, residualOut, residualOut, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent) {
			return fmt.Errorf("vae encoder: invalid %s second residual: %w", prefix, err)
		}
		for _, item := range []struct {
			name string
			size int
		}{
			{prefix + ".residual.0.gamma", residualIn}, {prefix + ".residual.2.bias", residualOut},
			{prefix + ".residual.3.gamma", residualOut}, {prefix + ".residual.6.bias", residualOut},
		} {
			if err := compiler.vector(item.name, item.size); err != nil {
				return err
			}
		}
		tensors := []string{
			prefix + ".residual.0.gamma", prefix + ".residual.2.weight", prefix + ".residual.2.bias",
			prefix + ".residual.3.gamma", prefix + ".residual.6.weight", prefix + ".residual.6.bias",
		}
		shortcut := compiler.has(prefix + ".shortcut.weight")
		if shortcut != (residualIn != residualOut) {
			return fmt.Errorf("vae encoder: %s shortcut=%t channels=%d->%d", prefix, shortcut, residualIn, residualOut)
		}
		if shortcut {
			shortcutOut, shortcutIn, kt, kh, kw, err := conv3(prefix + ".shortcut.weight")
			if err != nil || !media.Convolution3DMatches(shortcutOut, shortcutIn, kt, kh, kw, residualOut, residualIn, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent) {
				return fmt.Errorf("vae encoder: invalid %s shortcut: %w", prefix, err)
			}
			if err := compiler.vector(prefix+".shortcut.bias", residualOut); err != nil {
				return err
			}
			tensors = append(tensors, prefix+".shortcut.weight", prefix+".shortcut.bias")
		}
		compiler.add(media.CodecResidual, prefix, residualIn, residualOut, tensors...)
		channels = residualOut
		return nil
	}
	compileAttention := func(prefix string) error {
		qkvChannels, ok := checked.MulInt(tensor.TripleExtent, channels)
		if !ok {
			return fmt.Errorf("vae encoder: %s qkv channels overflow", prefix)
		}
		qkvOut, qkvIn, kh, kw, err := conv2(prefix + ".to_qkv.weight")
		if err != nil || !media.Convolution2DMatches(qkvOut, qkvIn, kh, kw, qkvChannels, channels, tensor.SingletonExtent, tensor.SingletonExtent) {
			return fmt.Errorf("vae encoder: invalid %s qkv: %w", prefix, err)
		}
		projOut, projIn, kh, kw, err := conv2(prefix + ".proj.weight")
		if err != nil || !media.Convolution2DMatches(projOut, projIn, kh, kw, channels, channels, tensor.SingletonExtent, tensor.SingletonExtent) {
			return fmt.Errorf("vae encoder: invalid %s projection: %w", prefix, err)
		}
		for _, item := range []struct {
			name string
			size int
		}{{prefix + ".norm.gamma", channels}, {prefix + ".to_qkv.bias", qkvChannels}, {prefix + ".proj.bias", channels}} {
			if err := compiler.vector(item.name, item.size); err != nil {
				return err
			}
		}
		compiler.add(media.CodecAttention, prefix, channels, channels,
			prefix+".norm.gamma", prefix+".to_qkv.weight", prefix+".to_qkv.bias", prefix+".proj.weight", prefix+".proj.bias")
		return nil
	}
	compileDownsample := func(prefix string) error {
		spatialOut, spatialIn, kh, kw, err := conv2(prefix + ".resample.1.weight")
		if err != nil || !media.Convolution2DMatches(spatialOut, spatialIn, kh, kw, channels, channels, tensor.TripleExtent, tensor.TripleExtent) {
			return fmt.Errorf("vae encoder: invalid %s spatial downsample: %w", prefix, err)
		}
		if err := compiler.vector(prefix+".resample.1.bias", channels); err != nil {
			return err
		}
		if !compiler.has(prefix + ".time_conv.weight") {
			operator := media.CodecDownsampleSpatial
			compiler.add(operator, prefix, channels, channels, prefix+".resample.1.weight", prefix+".resample.1.bias")
			plan.Stride[tensor.SingletonExtent] *= operator.SpatialScale()
			plan.Stride[tensor.PairedExtent] *= operator.SpatialScale()
			return nil
		}
		timeOut, timeIn, kt, kh, kw, err := conv3(prefix + ".time_conv.weight")
		if err != nil || !media.Convolution3DMatches(timeOut, timeIn, kt, kh, kw, channels, channels, tensor.TripleExtent, tensor.SingletonExtent, tensor.SingletonExtent) {
			return fmt.Errorf("vae encoder: invalid %s temporal downsample: %w", prefix, err)
		}
		if err := compiler.vector(prefix+".time_conv.bias", channels); err != nil {
			return err
		}
		operator := media.CodecDownsampleSpatiotemporal
		compiler.add(operator, prefix, channels, channels,
			prefix+".resample.1.weight", prefix+".resample.1.bias", prefix+".time_conv.weight", prefix+".time_conv.bias")
		plan.Stride[tensor.FirstOffset] *= operator.TemporalScale()
		plan.Stride[tensor.SingletonExtent] *= operator.SpatialScale()
		plan.Stride[tensor.PairedExtent] *= operator.SpatialScale()
		return nil
	}

	for index := tensor.FirstOffset; ; index++ {
		prefix := fmt.Sprintf("encoder.downsamples.%d", index)
		residual := compiler.has(prefix + ".residual.2.weight")
		downsample := compiler.has(prefix + ".resample.1.weight")
		switch {
		case residual:
			err = compileResidual(prefix)
		case downsample:
			err = compileDownsample(prefix)
		default:
			if checked.Equal(index, tensor.FirstOffset) {
				return plan, fmt.Errorf("vae encoder: no downsample graph")
			}
			goto middle
		}
		if err != nil {
			return plan, err
		}
	}

middle:
	for index := tensor.FirstOffset; ; index++ {
		prefix := fmt.Sprintf("encoder.middle.%d", index)
		residual := compiler.has(prefix + ".residual.2.weight")
		attention := compiler.has(prefix + ".to_qkv.weight")
		switch {
		case residual:
			err = compileResidual(prefix)
		case attention:
			err = compileAttention(prefix)
		default:
			if checked.Equal(index, tensor.FirstOffset) {
				return plan, fmt.Errorf("vae encoder: no middle graph")
			}
			goto head
		}
		if err != nil {
			return plan, err
		}
	}

head:
	headOut, headIn, kt, kh, kw, err := conv3("encoder.head.2.weight")
	if err != nil || !checked.EvenInt(headOut) || !media.Convolution3DMatches(headOut, headIn, kt, kh, kw, headOut, channels, tensor.TripleExtent, tensor.TripleExtent, tensor.TripleExtent) {
		return plan, fmt.Errorf("vae encoder: invalid head: %w", err)
	}
	if err := compiler.vector("encoder.head.0.gamma", channels); err != nil {
		return plan, err
	}
	if err := compiler.vector("encoder.head.2.bias", headOut); err != nil {
		return plan, err
	}
	compiler.add(media.CodecHead, "encoder.head", channels, headOut, "encoder.head.0.gamma", "encoder.head.2.weight", "encoder.head.2.bias")
	pointOut, pointIn, kt, kh, kw, err := conv3("conv1.weight")
	if err != nil || !media.Convolution3DMatches(pointOut, pointIn, kt, kh, kw, headOut, headOut, tensor.SingletonExtent, tensor.SingletonExtent, tensor.SingletonExtent) {
		return plan, fmt.Errorf("vae encoder: invalid moment pointwise: %w", err)
	}
	if err := compiler.vector("conv1.bias", pointOut); err != nil {
		return plan, err
	}
	compiler.add(media.CodecPointwise, "conv1", pointIn, pointOut, "conv1.weight", "conv1.bias")
	latentChannels, ok := checked.DivExactInt(pointOut, tensor.PairedExtent)
	if !ok {
		return plan, fmt.Errorf("vae encoder: moment channels do not split into paired statistics")
	}
	plan.MomentChannels, plan.LatentChannels = pointOut, latentChannels
	plan.CodecProgram.Operations = compiler.ops
	if err := plan.CodecProgram.Validate("vae encoder"); err != nil {
		return plan, err
	}
	stats, err := compiler.finish(func(name string) bool {
		return strings.HasPrefix(name, "encoder.") || strings.HasPrefix(name, "conv1.")
	})
	if err != nil {
		return plan, err
	}
	plan.UsedTensorCount = stats.tensors
	plan.UsedWeightBytes = stats.weightBytes
	plan.LargestOpWeightBytes = stats.largestOpBytes
	return plan, nil
}
