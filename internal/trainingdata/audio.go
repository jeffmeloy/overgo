package trainingdata

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/dataset"
	"overgo/internal/media"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
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
		return decodedAudioExample(record, role, audio)
	}
}

func decodedAudioExample(record RawRecord, role ValueRole, audio media.DecodedAudio) (Example, error) {
	sampleRate, fits := checked.Int(audio.Format.SampleRate)
	if audio.Format.Channels != 1 || !fits || len(audio.Samples) == 0 {
		return Example{}, errors.New("training data: audio requires nonempty mono samples and a representable rate")
	}
	return Example{ID: record.ID, Group: record.Group, Values: []Value{{
		Role: role, Modality: recipecontract.ModalityAudio, Encoding: EncodingFloat32LE,
		Shape: []int{len(audio.Samples)}, SampleRate: sampleRate, Data: binaryschema.LittleEndian.Float32s(audio.Samples),
	}}}, nil
}

// AudioTextProcessor reads row-addressed audio through the shared source owner,
// decodes once and persists signal admission before exposing input and target
// values to the universal trainer. Rejected signals return an error carrying the
// durable decision identity; the stream does not silently skip or advance them.
// The caller owns the source reader's lifetime.
func AudioTextProcessor(repository artifact.Repository, source *dataset.AudioPayloadReader, policy dataset.AudioInspectionPolicy) (Processor, error) {
	if repository == nil || source == nil {
		return nil, errors.New("training data: incomplete audio-text processor")
	}
	if err := policy.Validate(); err != nil {
		return nil, err
	}
	return func(ctx context.Context, record RawRecord) (Example, error) {
		var pair dataset.SpeechRecord
		if err := strictjson.DecodeBytes(record.Data, &pair); err != nil {
			return Example{}, err
		}
		if !utf8.ValidString(pair.Target) || strings.TrimSpace(pair.Target) == "" {
			return Example{}, errors.New("training data: audio transcript is absent or invalid")
		}
		locations, err := repository.Locations(ctx, pair.Origin.Container)
		if err != nil {
			return Example{}, err
		}
		data, err := source.Read(ctx, dataset.AudioPayloadReference{Path: fileLocation(locations), Audio: pair.Audio, Origin: pair.Origin}, policy.MaximumEncodedBytes)
		if err != nil {
			return Example{}, err
		}
		inspection, err := dataset.InspectAudio(ctx, repository, data, pair.Origin, policy)
		if err != nil {
			return Example{}, err
		}
		if inspection.Decision.Outcome != recipecontract.AudioAdmissionAccepted {
			return Example{}, fmt.Errorf("training data: audio admission %s: %s", inspection.Decision.Outcome, inspection.DecisionID)
		}
		example, err := decodedAudioExample(record, RoleInput, media.DecodedAudio{Format: inspection.Signal.Format, Samples: inspection.Samples})
		if err != nil {
			return Example{}, err
		}
		example.Values = append(example.Values, Value{Role: RoleTarget, Modality: recipecontract.ModalityText, Encoding: EncodingUTF8, Data: []byte(pair.Target)})
		return example, nil
	}, nil
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
