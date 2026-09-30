package projector

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/hostmath"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
	"overgo/internal/tensorcatalog"
)

const gemma4UAProjectorType = "gemma4ua"

type gemma4AudioSpec struct {
	SampleRate      int
	SamplesPerToken int
	Hidden          int
	RMSNormEpsilon  float32
}

type gemma4AudioOutput struct {
	Embeddings reference.Value
}

func readGemma4AudioSpec(file *gguf.File) (gemma4AudioSpec, error) {
	if err := validateProjector(
		file, audioProjectorTypeKey, audioEncoderEnabledKey, gemma4UAProjectorType, "audio",
	); err != nil {
		return gemma4AudioSpec{}, err
	}
	inputWidth, err := metadataUint32(file, "clip.audio.embedding_length")
	if err != nil {
		return gemma4AudioSpec{}, err
	}
	hidden, err := metadataUint32(file, "clip.audio.projection_dim")
	if err != nil {
		return gemma4AudioSpec{}, err
	}
	epsilon, err := metadataFloat32(file, "clip.audio.attention.layer_norm_epsilon")
	if err != nil {
		return gemma4AudioSpec{}, err
	}
	sampleRate, err := metadataUint32(file, "clip.audio.sample_rate")
	if err != nil {
		return gemma4AudioSpec{}, err
	}
	inputWidthInt, inputOK := checked.Int(uint64(inputWidth))
	hiddenInt, hiddenOK := checked.Int(uint64(hidden))
	sampleRateInt, rateOK := checked.Int(uint64(sampleRate))
	if !inputOK || !hiddenOK || !rateOK {
		return gemma4AudioSpec{}, errors.New("projector: Gemma 4 audio dimensions exceed native limits")
	}
	spec := gemma4AudioSpec{
		SampleRate: sampleRateInt, SamplesPerToken: inputWidthInt, Hidden: hiddenInt,
		RMSNormEpsilon: epsilon,
	}
	if err := spec.validate(); err != nil {
		return gemma4AudioSpec{}, err
	}
	projection, ok := file.Tensor("mm.a.input_projection.weight")
	if !ok {
		return gemma4AudioSpec{}, errors.New("projector: Gemma 4 audio projection is unavailable")
	}
	if err := tensorcatalog.ValidateInfo(projection, tensorcatalog.Requirement{Shapes: [][]uint64{{
		uint64(spec.SamplesPerToken), uint64(spec.Hidden),
	}}}); err != nil {
		return gemma4AudioSpec{}, fmt.Errorf("projector: Gemma 4 audio projection: %w", err)
	}
	return spec, nil
}

func (s gemma4AudioSpec) validate() error {
	if s.SampleRate <= 0 || s.SamplesPerToken <= 0 || s.Hidden <= 0 || s.RMSNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid Gemma 4 audio metadata: %+v", s)
	}
	return nil
}

func (r *Gemma4Runner) AudioSpec() (gemma4AudioSpec, error) {
	if r == nil || r.file == nil {
		return gemma4AudioSpec{}, errRunnerClosed
	}
	return readGemma4AudioSpec(r.file)
}

func preprocessGemma4Audio(samples []float32, spec gemma4AudioSpec) ([]float32, int, error) {
	if err := spec.validate(); err != nil {
		return nil, 0, err
	}
	if len(samples) == 0 {
		return nil, 0, errors.New("projector: audio is empty")
	}
	for index, sample := range samples {
		if !checked.Finite32(sample) {
			return nil, 0, fmt.Errorf("projector: audio sample %d is not finite", index)
		}
	}
	rows := 1 + (len(samples)-1)/spec.SamplesPerToken
	elements, ok := checked.MulInt(rows, spec.SamplesPerToken)
	if !ok {
		return nil, 0, errors.New("projector: audio frame count exceeds native limits")
	}
	frames := make([]float32, elements)
	copy(frames, samples)
	return frames, rows, nil
}

func (r *Gemma4Runner) EncodeAudio(ctx context.Context, samples []float32) (gemma4AudioOutput, error) {
	spec, err := r.AudioSpec()
	if err != nil {
		return gemma4AudioOutput{}, err
	}
	frames, rows, err := preprocessGemma4Audio(samples, spec)
	if err != nil {
		return gemma4AudioOutput{}, err
	}
	if err := ctx.Err(); err != nil {
		return gemma4AudioOutput{}, err
	}
	if r.cuda != nil {
		return r.encodeAudioCUDA(ctx, frames, rows, spec)
	}
	dtype.RoundBF16Slice(frames)
	normed := make([]float32, len(frames))
	hostmath.RMSNormInto(normed, frames, nil, rows, spec.SamplesPerToken, float64(spec.RMSNormEpsilon))
	dtype.RoundBF16Slice(normed)
	projection, err := r.load(ctx, "mm.a.input_projection.weight")
	if err != nil {
		return gemma4AudioOutput{}, err
	}
	embeddings := hostmath.LinearF64BiasFirstNew(normed, projection.Data, nil, rows, spec.SamplesPerToken, spec.Hidden)
	dtype.RoundBF16Slice(embeddings)
	value, err := reference.NewValue(tensor.MustShape(uint64(spec.Hidden), uint64(rows)), embeddings)
	if err != nil {
		return gemma4AudioOutput{}, err
	}
	return gemma4AudioOutput{Embeddings: value}, nil
}
