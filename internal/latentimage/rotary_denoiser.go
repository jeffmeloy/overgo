package latentimage

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// rotaryCoordinateAxes follows the pinned image and text coordinate contract.
// Reconsider four coordinates only when admitting a different positional scheme.
const rotaryCoordinateAxes = 4

// rotaryTimestepBase belongs to the pinned F64 timestep formula independently of RoPE.
// Reconsider this value only with a different timestep source contract.
const rotaryTimestepBase = 10000

type rotaryDenoiserProgram struct {
	latent, context, embedding, rope, unit *tensor.Tensor
	output                                 *tensor.Tensor
	boundaries                             map[string]*tensor.Tensor
	weightInputs                           tensor.WeightBindings
	spec                                   TransformerSpec
	height, width, text                    int
}

func compileRotaryDenoiserProgram(t TransformerSpec, epsilon float32, height, width, text int, skip []int, storage dtype.Type) (*rotaryDenoiserProgram, error) {
	if !checked.PositiveInts(height, width, text, t.InChannels, t.TextHidden, t.RefinerTextBlocks, t.Layers) || !checked.EvenInt(t.TimestepEmbed) || t.HeadDim%(rotaryCoordinateAxes*tensor.PairedExtent) != 0 || !checked.PositiveFinite64(t.RopeTheta) || t.ModFieldsOr6() != jointModulationFields {
		return nil, errors.New("rotary denoiser: invalid geometry, frequency or modulation")
	}
	image, ok := checked.MulInt(height, width)
	if !ok {
		return nil, errors.New("rotary denoiser: spatial size overflows")
	}
	total, ok := checked.AddInt(image, text)
	if !ok {
		return nil, errors.New("rotary denoiser: sequence size overflows")
	}
	for _, index := range skip {
		if !checked.ValidIndex(index, t.Layers) {
			return nil, fmt.Errorf("rotary denoiser: skipped block %d outside %d layers", index, t.Layers)
		}
	}
	p := &rotaryDenoiserProgram{spec: t, height: height, width: width, text: text, boundaries: map[string]*tensor.Tensor{}}
	g, err := newRotaryGraph(t, epsilon, total, storage, &p.weightInputs)
	if err != nil {
		return nil, err
	}
	b := g.b
	p.latent = b.Input("latent", dtype.F32, tensor.MustShape(uint64(image), uint64(t.InChannels)))
	p.context = b.Input("context", dtype.F32, tensor.MustShape(uint64(t.TextHidden), uint64(text)))
	p.embedding = b.Input("timestep_embedding", dtype.F32, tensor.MustShape(uint64(t.TimestepEmbed), tensor.SingletonExtent))
	p.unit = b.Input("unit", dtype.F32, tensor.MustShape(g.width))
	p.rope = g.rotaryInput("rotary")
	time := g.linear("time_embedding.0", g.round(p.embedding), uint64(t.TimestepEmbed), g.width, true)
	time = g.linear("time_embedding.2", g.round(b.SiLUNative(time)), g.width, g.width, true)
	mod := b.Reshape(g.linear("shared_block_modulation", time, g.width, jointModulationFields*g.width, true), jointModulationFields*g.width)
	head := b.Reshape(g.linear("video_head_modulation", time, g.width, tensor.PairedExtent*g.width, true), tensor.PairedExtent*g.width)
	p.boundaries["timestep_embedding"], p.boundaries["modulation"], p.boundaries["head_modulation"] = time, mod, head
	video := g.linear("video_embedder", g.round(b.Transpose2D(p.latent)), uint64(t.InChannels), g.width, true)
	p.boundaries["video"] = video
	textGraph := *g
	textGraph.sequence = uint64(text)
	textGraph.sine = b.FlatSlice(g.sine, uint64(image)*g.headWidth, g.headWidth, tensor.SingletonExtent, uint64(text))
	textGraph.cosine = b.FlatSlice(g.cosine, uint64(image)*g.headWidth, g.headWidth, tensor.SingletonExtent, uint64(text))
	refined, _ := textGraph.refine(t, p.context)
	p.boundaries["text_refinement"] = refined
	hidden := b.Concat(video, refined, tensor.SingletonExtent)
	for index := range t.Layers {
		if slices.Contains(skip, index) {
			continue
		}
		hidden = g.joint(hidden, mod, p.unit, index)
		if index == 0 || index == t.Layers/tensor.PairedExtent || index == t.Layers-1 {
			p.boundaries[fmt.Sprintf("block.%d", index)] = hidden
		}
	}
	hidden = b.FlatSlice(hidden, tensor.FirstOffset, g.width, uint64(image))
	hidden = g.norm("final_norm_video", hidden, g.width)
	hidden = g.modulate(hidden, b.FlatSlice(head, tensor.FirstOffset, g.width), b.FlatSlice(head, g.width, g.width), p.unit)
	p.output = b.Transpose2D(g.linear("final_linear_video", hidden, g.width, uint64(t.InChannels), false))
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("rotary denoiser: %w", err)
	}
	return p, nil
}

func (p *rotaryDenoiserProgram) feeds(latent, context []float32, timestep float32) (map[*tensor.Tensor]reference.Value, error) {
	if err := checked.Length(latent, p.spec.InChannels, p.height, p.width); err != nil {
		return nil, err
	}
	if err := checked.Length(context, p.spec.TextHidden, p.text); err != nil {
		return nil, err
	}
	embedding, err := (media.SinusoidalProgram{Dimensions: p.spec.TimestepEmbed, FrequencyBase: rotaryTimestepBase, InputScale: float64(tensor.UnitScale)}).Encode32(float64(timestep))
	if err != nil {
		return nil, err
	}
	rope, err := coordinateRotary(p.spec.HeadDim, p.height, p.width, p.text, p.spec.RopeTheta)
	if err != nil {
		return nil, err
	}
	unit := make([]float32, p.spec.Hidden)
	for i := range unit {
		unit[i] = tensor.UnitScale
	}
	return map[*tensor.Tensor]reference.Value{
		p.latent: {Shape: p.latent.Shape, Data: latent}, p.context: {Shape: p.context.Shape, Data: context},
		p.embedding: {Shape: p.embedding.Shape, Data: embedding}, p.rope: {Shape: p.rope.Shape, Data: rope}, p.unit: {Shape: p.unit.Shape, Data: unit},
	}, nil
}

func coordinateRotary(head, height, width, text int, base float64) ([]float32, error) {
	if !checked.PositiveInts(head, height, width, text) || head%(rotaryCoordinateAxes*tensor.PairedExtent) != 0 || !checked.PositiveFinite64(base) {
		return nil, errors.New("coordinate rotary: invalid geometry or frequency base")
	}
	image, ok := checked.MulInt(height, width)
	if !ok {
		return nil, errors.New("coordinate rotary: image size overflows")
	}
	sequence, ok := checked.AddInt(image, text)
	if !ok {
		return nil, errors.New("coordinate rotary: sequence size overflows")
	}
	count, ok := checked.MulInt(sequence, head)
	if !ok {
		return nil, errors.New("coordinate rotary: output size overflows")
	}
	values := make([]float32, count)
	bands := head / (rotaryCoordinateAxes * tensor.PairedExtent)
	frequencies := make([]float32, bands)
	for band := range bands {
		frequencies[band] = tensor.UnitScale / float32(math.Pow(base, float64(float32(band)/float32(bands))))
	}
	for token := range sequence {
		coordinates := [rotaryCoordinateAxes]float32{0, float32(text), float32(token/width) - float32(height-1)/tensor.PairedExtent, float32(token%width) - float32(width-1)/tensor.PairedExtent}
		if token >= image {
			coordinates = [rotaryCoordinateAxes]float32{0, float32(token - image), 0, 0}
		}
		for axis, coordinate := range coordinates {
			for band, frequency := range frequencies {
				phase := coordinate * frequency
				index := token*head + axis*bands + band
				values[index] = float32(math.Sin(float64(phase)))
				values[index+head/tensor.PairedExtent] = float32(math.Cos(float64(phase)))
			}
		}
	}
	return values, nil
}
