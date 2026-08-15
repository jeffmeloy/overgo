package projector

import (
	"context"
	"errors"
	"fmt"
	"math"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

const gemma4UAProjectorType = "gemma4ua"

type Gemma4AudioSpec struct {
	SampleRate      int
	SamplesPerToken int
	Hidden          int
	RMSNormEpsilon  float32
}

type Gemma4AudioOutput struct {
	Embeddings reference.Value
}

func ReadGemma4AudioSpec(file *gguf.File) (Gemma4AudioSpec, error) {
	if err := validateProjector(
		file, "clip.audio.projector_type", "clip.has_audio_encoder", gemma4UAProjectorType, "audio",
	); err != nil {
		return Gemma4AudioSpec{}, err
	}
	inputWidth, err := metadataUint32(file, "clip.audio.embedding_length")
	if err != nil {
		return Gemma4AudioSpec{}, err
	}
	hidden, err := metadataUint32(file, "clip.audio.projection_dim")
	if err != nil {
		return Gemma4AudioSpec{}, err
	}
	epsilon, err := metadataFloat32(file, "clip.audio.attention.layer_norm_epsilon")
	if err != nil {
		return Gemma4AudioSpec{}, err
	}
	maxNativeInt := uint64(^uint(0) >> 1)
	if uint64(inputWidth) > maxNativeInt || uint64(hidden) > maxNativeInt {
		return Gemma4AudioSpec{}, errors.New("projector: Gemma 4 audio dimensions exceed native limits")
	}
	spec := Gemma4AudioSpec{
		SampleRate: 16000, SamplesPerToken: int(inputWidth), Hidden: int(hidden),
		RMSNormEpsilon: epsilon,
	}
	if err := spec.validate(); err != nil {
		return Gemma4AudioSpec{}, err
	}
	projection, ok := file.Tensor("mm.a.input_projection.weight")
	if !ok || projection.Dimensions != 2 ||
		projection.Shape[0] != uint64(spec.SamplesPerToken) || projection.Shape[1] != uint64(spec.Hidden) {
		return Gemma4AudioSpec{}, fmt.Errorf(
			"projector: Gemma 4 audio projection shape is invalid; want [%d %d]",
			spec.SamplesPerToken, spec.Hidden,
		)
	}
	return spec, nil
}

func (s Gemma4AudioSpec) validate() error {
	if s.SampleRate <= 0 || s.SamplesPerToken <= 0 || s.Hidden <= 0 || s.RMSNormEpsilon <= 0 {
		return fmt.Errorf("projector: invalid Gemma 4 audio metadata: %+v", s)
	}
	return nil
}

func (r *Gemma4Runner) AudioSpec() (Gemma4AudioSpec, error) {
	if r == nil || r.file == nil {
		return Gemma4AudioSpec{}, errRunnerClosed
	}
	return ReadGemma4AudioSpec(r.file)
}

func PreprocessGemma4Audio(samples []float32, spec Gemma4AudioSpec) ([]float32, int, error) {
	if err := spec.validate(); err != nil {
		return nil, 0, err
	}
	if len(samples) == 0 {
		return nil, 0, errors.New("projector: audio is empty")
	}
	for index, sample := range samples {
		if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
			return nil, 0, fmt.Errorf("projector: audio sample %d is not finite", index)
		}
	}
	rows := 1 + (len(samples)-1)/spec.SamplesPerToken
	if rows > int(^uint(0)>>1)/spec.SamplesPerToken {
		return nil, 0, errors.New("projector: audio frame count exceeds native limits")
	}
	frames := make([]float32, rows*spec.SamplesPerToken)
	copy(frames, samples)
	return frames, rows, nil
}

func (r *Gemma4Runner) EncodeAudio(ctx context.Context, samples []float32) (Gemma4AudioOutput, error) {
	spec, err := r.AudioSpec()
	if err != nil {
		return Gemma4AudioOutput{}, err
	}
	frames, rows, err := PreprocessGemma4Audio(samples, spec)
	if err != nil {
		return Gemma4AudioOutput{}, err
	}
	if err := ctx.Err(); err != nil {
		return Gemma4AudioOutput{}, err
	}
	if r.cuda != nil {
		return r.encodeAudioCUDA(ctx, frames, rows, spec)
	}
	bf16RoundSlice(frames)
	normed := make([]float32, len(frames))
	rmsNormNoWeight(normed, frames, rows, spec.SamplesPerToken, spec.RMSNormEpsilon)
	bf16RoundSlice(normed)
	projection, err := r.load(ctx, "mm.a.input_projection.weight")
	if err != nil {
		return Gemma4AudioOutput{}, err
	}
	embeddings := linear(normed, projection.Data, nil, rows, spec.SamplesPerToken, spec.Hidden)
	bf16RoundSlice(embeddings)
	value, err := reference.NewValue(tensor.MustShape(uint64(spec.Hidden), uint64(rows)), embeddings)
	if err != nil {
		return Gemma4AudioOutput{}, err
	}
	return Gemma4AudioOutput{Embeddings: value}, nil
}
