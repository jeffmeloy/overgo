package latentvideo

import (
	"fmt"
	"sort"
	"strings"

	"overgo/internal/pytorchzip"
)

// VAEEncoderPlan: checkpoint-derived causal source encoder graph.
type VAEEncoderPlan struct {
	InputChannels        int
	LatentChannels       int
	MomentChannels       int
	Stride               [3]int
	UsedTensorCount      int
	UsedWeightBytes      int64
	LargestOpWeightBytes int64
	ops                  []vaeDecoderOp
}

func (p VAEEncoderPlan) Ops() int { return len(p.ops) }

func (p VAEEncoderPlan) OpPrefixes() []string {
	prefixes := make([]string, len(p.ops))
	for index, op := range p.ops {
		prefixes[index] = op.prefix
	}
	return prefixes
}

// CompileVAEEncoderPlan derives the Wan/LiveEdit encoder from artifact tensors.
func CompileVAEEncoderPlan(metas []pytorchzip.TensorMeta) (VAEEncoderPlan, error) {
	byName := make(map[string]pytorchzip.TensorMeta, len(metas))
	for _, meta := range metas {
		byName[meta.Name] = meta
	}
	require := func(name string) (pytorchzip.TensorMeta, error) {
		meta, ok := byName[name]
		if !ok {
			return pytorchzip.TensorMeta{}, fmt.Errorf("vae encoder: missing tensor %s", name)
		}
		return meta, nil
	}
	vector := func(name string, elements int) error {
		meta, err := require(name)
		if err != nil {
			return err
		}
		if meta.Numel != int64(elements) {
			return fmt.Errorf("vae encoder: %s elements=%d want=%d", name, meta.Numel, elements)
		}
		return nil
	}
	conv3 := func(name string) (out, in, kt, kh, kw int, err error) {
		meta, err := require(name)
		if err != nil {
			return 0, 0, 0, 0, 0, err
		}
		return (vaeTensorShape{name: name, shape: meta.Shape}).conv3d("encoder")
	}
	conv2 := func(name string) (out, in, kh, kw int, err error) {
		meta, err := require(name)
		if err != nil {
			return 0, 0, 0, 0, err
		}
		return (vaeTensorShape{name: name, shape: meta.Shape}).conv2d("encoder")
	}

	plan := VAEEncoderPlan{Stride: [3]int{1, 1, 1}}
	var names []string
	add := func(kind vaeOpKind, prefix string, in, out int, tensors ...string) {
		plan.ops = append(plan.ops, vaeDecoderOp{kind: kind, prefix: prefix, cIn: in, cOut: out})
		names = append(names, tensors...)
	}
	out, in, kt, kh, kw, err := conv3("encoder.conv1.weight")
	if err != nil || in <= 0 || out <= 0 || kt != 3 || kh != 3 || kw != 3 {
		return plan, fmt.Errorf("vae encoder: invalid encoder.conv1.weight: %w", err)
	}
	if err := vector("encoder.conv1.bias", out); err != nil {
		return plan, err
	}
	plan.InputChannels = in
	add(vaeOpConv, "encoder.conv1", in, out, "encoder.conv1.weight", "encoder.conv1.bias")
	channels := out

	compileResidual := func(prefix string) error {
		residualOut, residualIn, kt, kh, kw, err := conv3(prefix + ".residual.2.weight")
		if err != nil || residualIn != channels || kt != 3 || kh != 3 || kw != 3 {
			return fmt.Errorf("vae encoder: invalid %s first residual: %w", prefix, err)
		}
		secondOut, secondIn, kt, kh, kw, err := conv3(prefix + ".residual.6.weight")
		if err != nil || secondIn != residualOut || secondOut != residualOut || kt != 3 || kh != 3 || kw != 3 {
			return fmt.Errorf("vae encoder: invalid %s second residual: %w", prefix, err)
		}
		for _, item := range []struct {
			name string
			size int
		}{
			{prefix + ".residual.0.gamma", residualIn}, {prefix + ".residual.2.bias", residualOut},
			{prefix + ".residual.3.gamma", residualOut}, {prefix + ".residual.6.bias", residualOut},
		} {
			if err := vector(item.name, item.size); err != nil {
				return err
			}
		}
		tensors := []string{
			prefix + ".residual.0.gamma", prefix + ".residual.2.weight", prefix + ".residual.2.bias",
			prefix + ".residual.3.gamma", prefix + ".residual.6.weight", prefix + ".residual.6.bias",
		}
		_, shortcut := byName[prefix+".shortcut.weight"]
		if shortcut != (residualIn != residualOut) {
			return fmt.Errorf("vae encoder: %s shortcut=%t channels=%d->%d", prefix, shortcut, residualIn, residualOut)
		}
		if shortcut {
			shortcutOut, shortcutIn, kt, kh, kw, err := conv3(prefix + ".shortcut.weight")
			if err != nil || shortcutIn != residualIn || shortcutOut != residualOut || kt != 1 || kh != 1 || kw != 1 {
				return fmt.Errorf("vae encoder: invalid %s shortcut: %w", prefix, err)
			}
			if err := vector(prefix+".shortcut.bias", residualOut); err != nil {
				return err
			}
			tensors = append(tensors, prefix+".shortcut.weight", prefix+".shortcut.bias")
		}
		add(vaeOpResidual, prefix, residualIn, residualOut, tensors...)
		channels = residualOut
		return nil
	}
	compileAttention := func(prefix string) error {
		qkvOut, qkvIn, kh, kw, err := conv2(prefix + ".to_qkv.weight")
		if err != nil || qkvIn != channels || qkvOut != 3*channels || kh != 1 || kw != 1 {
			return fmt.Errorf("vae encoder: invalid %s qkv: %w", prefix, err)
		}
		projOut, projIn, kh, kw, err := conv2(prefix + ".proj.weight")
		if err != nil || projIn != channels || projOut != channels || kh != 1 || kw != 1 {
			return fmt.Errorf("vae encoder: invalid %s projection: %w", prefix, err)
		}
		for _, item := range []struct {
			name string
			size int
		}{{prefix + ".norm.gamma", channels}, {prefix + ".to_qkv.bias", 3 * channels}, {prefix + ".proj.bias", channels}} {
			if err := vector(item.name, item.size); err != nil {
				return err
			}
		}
		add(vaeOpAttention, prefix, channels, channels,
			prefix+".norm.gamma", prefix+".to_qkv.weight", prefix+".to_qkv.bias", prefix+".proj.weight", prefix+".proj.bias")
		return nil
	}
	compileDownsample := func(prefix string) error {
		spatialOut, spatialIn, kh, kw, err := conv2(prefix + ".resample.1.weight")
		if err != nil || spatialIn != channels || spatialOut != channels || kh != 3 || kw != 3 {
			return fmt.Errorf("vae encoder: invalid %s spatial downsample: %w", prefix, err)
		}
		if err := vector(prefix+".resample.1.bias", channels); err != nil {
			return err
		}
		if _, temporal := byName[prefix+".time_conv.weight"]; !temporal {
			add(vaeOpDownsample2D, prefix, channels, channels, prefix+".resample.1.weight", prefix+".resample.1.bias")
			plan.Stride[1] *= vaeSpatialScale
			plan.Stride[2] *= vaeSpatialScale
			return nil
		}
		timeOut, timeIn, kt, kh, kw, err := conv3(prefix + ".time_conv.weight")
		if err != nil || timeIn != channels || timeOut != channels || kt != 3 || kh != 1 || kw != 1 {
			return fmt.Errorf("vae encoder: invalid %s temporal downsample: %w", prefix, err)
		}
		if err := vector(prefix+".time_conv.bias", channels); err != nil {
			return err
		}
		add(vaeOpDownsample3D, prefix, channels, channels,
			prefix+".resample.1.weight", prefix+".resample.1.bias", prefix+".time_conv.weight", prefix+".time_conv.bias")
		plan.Stride[0] *= vaeSpatialScale
		plan.Stride[1] *= vaeSpatialScale
		plan.Stride[2] *= vaeSpatialScale
		return nil
	}

	for index := 0; ; index++ {
		prefix := fmt.Sprintf("encoder.downsamples.%d", index)
		_, residual := byName[prefix+".residual.2.weight"]
		_, downsample := byName[prefix+".resample.1.weight"]
		switch {
		case residual:
			err = compileResidual(prefix)
		case downsample:
			err = compileDownsample(prefix)
		default:
			if index == 0 {
				return plan, fmt.Errorf("vae encoder: no downsample graph")
			}
			goto middle
		}
		if err != nil {
			return plan, err
		}
	}

middle:
	for index := 0; ; index++ {
		prefix := fmt.Sprintf("encoder.middle.%d", index)
		_, residual := byName[prefix+".residual.2.weight"]
		_, attention := byName[prefix+".to_qkv.weight"]
		switch {
		case residual:
			err = compileResidual(prefix)
		case attention:
			err = compileAttention(prefix)
		default:
			if index == 0 {
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
	if err != nil || headIn != channels || headOut <= 0 || headOut%2 != 0 || kt != 3 || kh != 3 || kw != 3 {
		return plan, fmt.Errorf("vae encoder: invalid head: %w", err)
	}
	if err := vector("encoder.head.0.gamma", channels); err != nil {
		return plan, err
	}
	if err := vector("encoder.head.2.bias", headOut); err != nil {
		return plan, err
	}
	add(vaeOpHead, "encoder.head", channels, headOut, "encoder.head.0.gamma", "encoder.head.2.weight", "encoder.head.2.bias")
	pointOut, pointIn, kt, kh, kw, err := conv3("conv1.weight")
	if err != nil || pointIn != headOut || pointOut != headOut || kt != 1 || kh != 1 || kw != 1 {
		return plan, fmt.Errorf("vae encoder: invalid moment pointwise: %w", err)
	}
	if err := vector("conv1.bias", pointOut); err != nil {
		return plan, err
	}
	add(vaeOpPointwise, "conv1", pointIn, pointOut, "conv1.weight", "conv1.bias")
	plan.MomentChannels, plan.LatentChannels = pointOut, pointOut/2

	consumed := make(map[string]bool, len(names))
	for _, name := range names {
		consumed[name] = true
	}
	var unexpected []string
	for _, meta := range metas {
		if (strings.HasPrefix(meta.Name, "encoder.") || strings.HasPrefix(meta.Name, "conv1.")) && !consumed[meta.Name] {
			unexpected = append(unexpected, meta.Name)
		}
	}
	if len(unexpected) > 0 {
		sort.Strings(unexpected)
		return plan, fmt.Errorf("vae encoder: unconsumed tensors %v", unexpected)
	}
	bindings, err := pytorchzip.CompileBindings(metas, names)
	if err != nil {
		return plan, err
	}
	offset := 0
	for index := range plan.ops {
		op := &plan.ops[index]
		count := vaeOpTensorCount(*op)
		op.bindings = bindings[offset : offset+count]
		offset += count
		for _, binding := range op.bindings {
			bytes, err := pytorchzip.TensorMetaBytes(binding.Meta)
			if err != nil {
				return plan, err
			}
			op.weightBytes += bytes
		}
		plan.UsedTensorCount += count
		plan.UsedWeightBytes += op.weightBytes
		plan.LargestOpWeightBytes = max(plan.LargestOpWeightBytes, op.weightBytes)
	}
	if offset != len(bindings) {
		return plan, fmt.Errorf("vae encoder: binding partition consumed %d of %d", offset, len(bindings))
	}
	return plan, nil
}
