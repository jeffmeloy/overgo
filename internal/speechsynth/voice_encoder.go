package speechsynth

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensorcatalog"
)

// codecEncoder maps mono PCM to the speaker projection input of the flow transformer.
// The projection consumes unquantized Mimi embeddings and never generation latents.
type codecEncoder struct {
	transform           *codecTransformer
	convs               []codecConv
	resnets             [][tensor.PairedExtent]codecConv
	down                codecConv
	speaker             []float32
	bos                 []float32
	width, frameSamples int
}

func loadCodecEncoder(read func(string, ...int) ([]float32, error), shapes map[string][]int, config artifactConfig, width int, normalization media.NormalizationProgram) (*codecEncoder, error) {
	sn := config.Mimi.Seanet
	if len(sn.Ratios) == 0 || !checked.PositiveInts(sn.Dimension, sn.NFilters, sn.KernelSize, sn.LastKernelSize, sn.Compress, sn.ResidualKernelSize, width) || sn.Channels != 1 {
		return nil, errors.New("speechsynth: voice encoder requires a complete mono SEANet configuration")
	}
	const prefix = "mimi.encoder.model."
	encoder := &codecEncoder{width: width, frameSamples: 1}
	add := func(index, input, output, kernel, stride int) error {
		weights, err := read(fmt.Sprintf("%s%d.conv.weight", prefix, index), output, input, kernel)
		if err != nil {
			return err
		}
		bias, err := read(fmt.Sprintf("%s%d.conv.bias", prefix, index), output)
		if err != nil {
			return err
		}
		encoder.convs = append(encoder.convs, codecConv{w: weights, b: bias, cIn: input, cOut: output, k: kernel, strd: stride})
		return nil
	}
	if err := add(0, 1, sn.NFilters, sn.KernelSize, 1); err != nil {
		return nil, err
	}
	// A zero input projection carries no recording information even if placeholder tensors exist.
	if !slices.ContainsFunc(encoder.convs[0].w, func(value float32) bool { return value != 0 }) {
		return nil, errors.New("speechsynth: voice encoder input weights are disabled")
	}
	channels := sn.NFilters
	for index := range sn.Ratios {
		stride := sn.Ratios[len(sn.Ratios)-1-index]
		if stride <= 0 || channels%sn.Compress != 0 {
			return nil, errors.New("speechsynth: invalid encoder stride or residual width")
		}
		product, ok := checked.MulInt(encoder.frameSamples, stride)
		if !ok {
			return nil, errors.New("speechsynth: encoder stride overflow")
		}
		encoder.frameSamples = product
		block, err := loadCodecResidual(read, prefix, tensor.TripleExtent*index+1, channels, sn)
		if err != nil {
			return nil, err
		}
		encoder.resnets = append(encoder.resnets, block)
		if err := add(tensor.TripleExtent*index+tensor.TripleExtent, channels, tensor.PairedExtent*channels, tensor.PairedExtent*stride, stride); err != nil {
			return nil, err
		}
		channels *= tensor.PairedExtent
	}
	if err := add(tensor.TripleExtent*len(sn.Ratios)+tensor.PairedExtent, channels, sn.Dimension, sn.LastKernelSize, 1); err != nil {
		return nil, err
	}
	var err error
	encoder.transform, err = loadCodecTransformer(read, shapes, config, "mimi.encoder_transformer.transformer.layers.", sn.Dimension, normalization)
	if err != nil {
		return nil, err
	}
	const downName = "mimi.downsample.conv.conv.weight"
	dimensions, err := tensorcatalog.Dimensions(shapes, downName, tensor.TripleExtent)
	if err != nil {
		return nil, err
	}
	if dimensions[1] != sn.Dimension || !checked.EvenInt(dimensions[tensor.PairedExtent]) {
		return nil, errors.New("speechsynth: invalid voice downsampling geometry")
	}
	down := codecConv{cIn: sn.Dimension, cOut: dimensions[0], k: dimensions[tensor.PairedExtent], strd: dimensions[tensor.PairedExtent] / tensor.PairedExtent}
	if down.strd <= 0 {
		return nil, errors.New("speechsynth: voice downsample stride must be positive")
	}
	if down.w, err = read(downName, dimensions...); err != nil {
		return nil, err
	}
	encoder.down = down
	product, ok := checked.MulInt(encoder.frameSamples, down.strd)
	if !ok {
		return nil, errors.New("speechsynth: voice frame geometry overflow")
	}
	encoder.frameSamples = product
	if float64(encoder.frameSamples)*config.Mimi.FrameRate != float64(config.Mimi.SampleRate) {
		return nil, errors.New("speechsynth: voice frame geometry differs from codec rates")
	}
	encoder.speaker, err = read("flow_lm.speaker_proj_weight", width, down.cOut)
	if err == nil && config.FlowLM.InsertBOSBeforeVoice {
		encoder.bos, err = read("flow_lm.bos_before_voice", 1, 1, width)
	}
	return encoder, err
}

func (e *codecEncoder) encode(ctx context.Context, mono []float32) ([]float32, int, error) {
	if ctx == nil || e == nil || len(mono) == 0 {
		return nil, 0, errors.New("speechsynth: voice encoding requires context and mono samples")
	}
	for _, value := range mono {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, 0, errors.New("speechsynth: voice recording contains nonfinite samples")
		}
	}
	padding := (e.frameSamples - len(mono)%e.frameSamples) % e.frameSamples
	x := make([]float32, len(mono)+padding)
	copy(x, mono)
	frames := len(x)
	apply := func(cv codecConv, input []float32) []float32 {
		output := hostmath.CausalConv1dSuffix(input, cv.cIn, frames, cv.w, cv.b, cv.cOut, cv.k, cv.strd, 0)
		frames /= cv.strd
		return output
	}
	if err := context.Cause(ctx); err != nil {
		return nil, 0, err
	}
	x = apply(e.convs[0], x)
	for index, block := range e.resnets {
		if err := context.Cause(ctx); err != nil {
			return nil, 0, err
		}
		residual := slices.Clone(x)
		for _, cv := range block {
			hostmath.ELUInPlace(residual)
			residual = apply(cv, residual)
		}
		for i := range x {
			x[i] += residual[i]
		}
		hostmath.ELUInPlace(x)
		x = apply(e.convs[index+1], x)
	}
	hostmath.ELUInPlace(x)
	x = apply(e.convs[len(e.convs)-1], x)
	if err := newCodecTransformState(e.transform).transform(ctx, x, frames); err != nil {
		return nil, 0, err
	}
	// PocketTTS 2.1.0 ConvDownsample1d replicates its first input on the causal left boundary.
	// An explicit prefix reuses the same convolution arithmetic and discards its artificial output.
	prefix := e.down.k - e.down.strd
	padded := make([]float32, e.down.cIn*(frames+prefix))
	for channel := range e.down.cIn {
		row := padded[channel*(frames+prefix) : (channel+1)*(frames+prefix)]
		for i := range prefix {
			row[i] = x[channel*frames]
		}
		copy(row[prefix:], x[channel*frames:(channel+1)*frames])
	}
	if err := context.Cause(ctx); err != nil {
		return nil, 0, err
	}
	x = hostmath.CausalConv1dSuffix(padded, e.down.cIn, frames+prefix, e.down.w, nil, e.down.cOut, e.down.k, e.down.strd, prefix/e.down.strd)
	frames /= e.down.strd
	rows := make([]float32, frames*e.down.cOut)
	for frame := range frames {
		for channel := range e.down.cOut {
			rows[frame*e.down.cOut+channel] = x[channel*frames+frame]
		}
	}
	conditioning := make([]float32, len(e.bos)+frames*e.width)
	copy(conditioning, e.bos)
	hostmath.Linear(conditioning[len(e.bos):], rows, e.speaker, frames, e.down.cOut, e.width)
	return conditioning, padding, context.Cause(ctx)
}
