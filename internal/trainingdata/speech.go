package trainingdata

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/binaryschema"
	"overgo/internal/checked"
	"overgo/internal/recipecontract"
	"overgo/internal/strictjson"
	"overgo/internal/tensor"
)

// SpeechLatentProfile binds normalized targets to their producer and admission bounds.
// Conditioning identifies target provenance rather than adding voice inputs to the text/BOS trainer.
// Limits bound one decoded example rather than total model or materializer memory.
type SpeechLatentProfile struct {
	Model               artifact.ID `json:"model"`
	Tokenizer           artifact.ID `json:"tokenizer"`
	Codec               artifact.ID `json:"codec"`
	Transform           artifact.ID `json:"transform"`
	Conditioning        artifact.ID `json:"conditioning"`
	Noise               artifact.ID `json:"noise"`
	Width               int         `json:"width"`
	MaximumFrames       int         `json:"maximum_frames"`
	MaximumTextBytes    int         `json:"maximum_text_bytes"`
	MaximumEncodedBytes int         `json:"maximum_encoded_bytes"`
}

// SpeechLatentRecord identifies a normalized codec target and its immutable source record.
// Profile binds normalization, conditioning, producer and noise identities without repeating them per frame.
type SpeechLatentRecord struct {
	Profile      artifact.ID `json:"profile"`
	Source       artifact.ID `json:"source"`
	SourceRecord string      `json:"source_record"`
	Text         string      `json:"text"`
	Frames       int         `json:"frames"`
	Seed         int64       `json:"seed"`
	Latents      []float32   `json:"normalized_codec_latents"`
}

// Content identifies the normalized target contract and refuses incomplete provenance or bounds.
func (p SpeechLatentProfile) Content() (artifact.Content, error) {
	if p.Model.Kind() != artifact.KindModel || !p.Tokenizer.Valid() || !p.Codec.Valid() ||
		p.Transform.Kind() != artifact.KindProfile || !p.Conditioning.Valid() || p.Noise.Kind() != artifact.KindProfile ||
		p.Width <= 0 || p.MaximumFrames <= 0 || p.MaximumTextBytes <= 0 || p.MaximumEncodedBytes <= 0 {
		return artifact.Content{}, errors.New("speech latent data: exact producer, normalization, conditioning, noise and positive bounds required")
	}
	return artifact.JSONContent(artifact.JSONContract(artifact.KindProfile, "overgo/speech-normalized-latents/v1"), p)
}

// SpeechLatentProcessor validates source provenance and emits one text/normalized-latent pair.
// Waveforms and speaker-projected conditioning have no accepted record field.
func SpeechLatentProcessor(reader artifact.Reader, profile SpeechLatentProfile) (Processor, error) {
	content, err := profile.Content()
	if err != nil || reader == nil {
		return nil, errors.Join(errors.New("speech latent data: reader and complete profile required"), err)
	}
	return func(ctx context.Context, raw RawRecord) (Example, error) {
		if err := ctx.Err(); err != nil {
			return Example{}, err
		}
		if len(raw.Data) > profile.MaximumEncodedBytes {
			return Example{}, errors.New("speech latent data: encoded record exceeds declared bound")
		}
		var record SpeechLatentRecord
		if err := strictjson.DecodeBytes(raw.Data, &record); err != nil {
			return Example{}, err
		}
		if record.Profile != content.Descriptor.ID || !record.Source.Valid() || strings.TrimSpace(record.SourceRecord) == "" ||
			!utf8.ValidString(record.Text) || strings.TrimSpace(record.Text) == "" || len(record.Text) > profile.MaximumTextBytes ||
			record.Frames <= 0 || record.Frames > profile.MaximumFrames || checked.Length(record.Latents, record.Frames, profile.Width) != nil {
			return Example{}, errors.New("speech latent data: record identity, text or normalized target shape differs from profile")
		}
		if _, found, err := reader.Artifact(ctx, record.Source); err != nil || !found {
			return Example{}, errors.Join(fmt.Errorf("speech latent data: source %s is unavailable", record.Source), err)
		}
		for _, value := range record.Latents {
			if !checked.Finite32(value) {
				return Example{}, errors.New("speech latent data: nonfinite normalized target")
			}
		}
		payload := make([]byte, binaryschema.Uint64Bytes+len(record.Latents)*binaryschema.Uint32Bytes)
		binary.LittleEndian.PutUint64(payload, uint64(record.Seed))
		for index, value := range record.Latents {
			binary.LittleEndian.PutUint32(payload[binaryschema.Uint64Bytes+index*binaryschema.Uint32Bytes:], math.Float32bits(value))
		}
		return Example{ID: raw.ID, Group: raw.Group, Values: []Value{
			{Role: RoleInput, Modality: recipecontract.ModalityText, Encoding: EncodingUTF8, Data: []byte(record.Text)},
			{Role: RoleTarget, Modality: recipecontract.ModalityAudio, Encoding: speechLatentEncoding, Shape: []int{record.Frames, profile.Width}, Data: payload},
		}}, nil
	}, nil
}

// speechLatentEncoding stores an int64 noise seed before frame-major normalized F32 values.
const speechLatentEncoding = "seed-i64-normalized-f32le/v1"

// SpeechLatents decodes the pair produced by SpeechLatentProcessor without interpreting it as PCM.
func SpeechLatents(example Example) (text string, latents []float32, frames int, seed int64, err error) {
	if len(example.Values) != tensor.PairedExtent {
		return "", nil, 0, 0, errors.New("speech latent data: one text input and one latent target required")
	}
	input, target := example.Values[0], example.Values[1]
	if input.Role != RoleInput || input.Modality != recipecontract.ModalityText || input.Encoding != EncodingUTF8 ||
		target.Role != RoleTarget || target.Modality != recipecontract.ModalityAudio || target.Encoding != speechLatentEncoding ||
		len(target.Shape) != tensor.PairedExtent || target.Shape[0] <= 0 || target.Shape[1] <= 0 || len(target.Data) <= binaryschema.Uint64Bytes {
		return "", nil, 0, 0, errors.New("speech latent data: invalid processed pair")
	}
	seed = int64(binary.LittleEndian.Uint64(target.Data))
	target.Data, target.Encoding = target.Data[binaryschema.Uint64Bytes:], EncodingFloat32LE
	latents, err = Float32(target)
	if err == nil {
		err = checked.Length(latents, target.Shape[0], target.Shape[1])
	}
	return string(input.Data), latents, target.Shape[0], seed, err
}
