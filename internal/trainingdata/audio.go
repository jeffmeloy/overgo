package trainingdata

import (
	"context"
	"errors"

	"overgo/internal/media"
	"overgo/internal/recipecontract"
)

// AudioProcessor decodes one mono WAV into normalized samples.
func AudioProcessor(role ValueRole) Processor {
	return func(_ context.Context, record RawRecord) (Example, error) {
		if !validRole(role) {
			return Example{}, errors.New("training data: invalid audio role")
		}
		samples, sampleRate, err := media.DecodeWAV(record.Data)
		if err != nil {
			return Example{}, err
		}
		return Example{ID: record.ID, Group: record.Group, Values: []Value{{
			Role: role, Modality: recipecontract.ModalityAudio, Encoding: EncodingFloat32LE,
			Shape: []int{len(samples)}, SampleRate: sampleRate, Data: EncodeFloat32(samples),
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
