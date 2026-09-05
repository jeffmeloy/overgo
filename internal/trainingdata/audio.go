package trainingdata

import (
	"context"
	"errors"

	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/recipecontract"
)

// AudioProcessor decodes one mono WAV or FLAC within the caller's scalar-sample
// budget. It preserves source rate and sample values; it does not normalize.
func AudioProcessor(role ValueRole, maximumSamples uint64) Processor {
	return func(ctx context.Context, record RawRecord) (Example, error) {
		if !validRole(role) {
			return Example{}, errors.New("training data: invalid audio role")
		}
		audio, _, err := media.DecodeAudio(ctx, record.Data, maximumSamples)
		if err != nil {
			return Example{}, err
		}
		sampleRate, fits := checked.Int(audio.Format.SampleRate)
		if audio.Format.Channels != 1 || !fits || len(audio.Samples) == 0 {
			return Example{}, errors.New("training data: audio requires nonempty mono samples and a representable rate")
		}
		return Example{ID: record.ID, Group: record.Group, Values: []Value{{
			Role: role, Modality: recipecontract.ModalityAudio, Encoding: EncodingFloat32LE,
			Shape: []int{len(audio.Samples)}, SampleRate: sampleRate, Data: binaryschema.LittleEndian.Float32s(audio.Samples),
		}}}, nil
	}
}

// Audio decodes one processor-owned sample vector and rate.
func Audio(value Value) ([]float32, int, error) {
	if value.Modality != recipecontract.ModalityAudio || len(value.Shape) != 1 || value.Shape[0] <= 0 || value.SampleRate <= 0 {
		return nil, 0, errors.New("training data: invalid audio value")
	}
	samples, err := Float32(value)
	if err != nil {
		return nil, 0, err
	}
	if len(samples) != value.Shape[0] {
		return nil, 0, errors.New("training data: audio payload differs from shape")
	}
	return samples, value.SampleRate, nil
}
