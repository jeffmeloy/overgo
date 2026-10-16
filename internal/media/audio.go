//overgo:runtime-inputs caller

package media

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

type audioOutputKey struct{}

// WithAudioOutput binds a synchronous provisional-output consumer to an execution.
// The consumer never publishes a completed recording.
func WithAudioOutput(ctx context.Context, emit func(context.Context, DecodedAudio) error) context.Context {
	return context.WithValue(ctx, audioOutputKey{}, emit)
}

// EmitAudio delivers provisional samples before the execution advances.
// A consumer error refuses the execution and its completed output.
// A provisional output requires finite samples in its declared interleaved format.
func EmitAudio(ctx context.Context, chunk DecodedAudio) error {
	if err := context.Cause(ctx); err != nil {
		return err
	}
	if emit, ok := ctx.Value(audioOutputKey{}).(func(context.Context, DecodedAudio) error); ok && emit != nil {
		if err := chunk.Format.Validate(); err != nil {
			return err
		}
		if chunk.Format.Encoding != "pcm-f32le" || len(chunk.Samples) == 0 || uint64(len(chunk.Samples))%uint64(chunk.Format.Channels) != 0 {
			return errors.New("media: provisional audio requires whole nonempty frames in pcm-f32le format")
		}
		for _, sample := range chunk.Samples {
			if math.IsNaN(float64(sample)) || math.IsInf(float64(sample), 0) {
				return errors.New("media: provisional audio requires finite samples")
			}
		}
		return emit(ctx, chunk)
	}
	return nil
}

const (
	// WAVMediaType names RIFF/WAVE audio, the form synthesized speech is published in.
	WAVMediaType          = "audio/wav"
	wavPCM                = 1
	wavIEEEFloat          = 3
	bitsPerByte           = 8
	pcm16Bits             = 16
	float32Bits           = 32
	float64Bits           = 64
	float32Bytes          = float32Bits / bitsPerByte
	pcm16Bytes            = pcm16Bits / bitsPerByte
	pcm16Magnitude        = 1 << (pcm16Bits - 1)
	riffHeaderBytes       = 12
	wavChunkHeaderBytes   = 8
	wavFormatMinimumBytes = 16
)

func DecodeFloat32LE(data []byte) ([]float32, error) {
	if len(data) == 0 {
		return nil, errors.New("media: audio is empty")
	}
	if len(data)%float32Bytes != 0 {
		return nil, fmt.Errorf("media: float32 audio byte count %d is not divisible by %d", len(data), float32Bytes)
	}
	samples := make([]float32, len(data)/float32Bytes)
	for index := range samples {
		samples[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[index*float32Bytes:]))
	}
	return samples, nil
}

// EncodeWAVPCM16 encodes mono samples (clipped to [-1, 1]) as a 16-bit PCM
// RIFF/WAVE byte slice, readable through DecodeAudio.
func EncodeWAVPCM16(samples []float32, sampleRate int) ([]byte, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("media: WAV sample rate %d", sampleRate)
	}
	dataBytes := len(samples) * pcm16Bytes
	out := make([]byte, riffHeaderBytes+wavChunkHeaderBytes+wavFormatMinimumBytes+wavChunkHeaderBytes+dataBytes)
	copy(out[0:4], "RIFF")
	binary.LittleEndian.PutUint32(out[4:8], uint32(len(out)-wavChunkHeaderBytes))
	copy(out[8:riffHeaderBytes], "WAVE")
	fmtStart := riffHeaderBytes
	copy(out[fmtStart:fmtStart+4], "fmt ")
	binary.LittleEndian.PutUint32(out[fmtStart+4:fmtStart+8], wavFormatMinimumBytes)
	body := fmtStart + wavChunkHeaderBytes
	binary.LittleEndian.PutUint16(out[body:body+2], wavPCM)
	binary.LittleEndian.PutUint16(out[body+2:body+4], 1) // mono
	binary.LittleEndian.PutUint32(out[body+4:body+8], uint32(sampleRate))
	binary.LittleEndian.PutUint32(out[body+8:body+12], uint32(sampleRate*pcm16Bytes))
	binary.LittleEndian.PutUint16(out[body+12:body+14], pcm16Bytes)
	binary.LittleEndian.PutUint16(out[body+14:body+16], pcm16Bits)
	dataStart := body + wavFormatMinimumBytes
	copy(out[dataStart:dataStart+4], "data")
	binary.LittleEndian.PutUint32(out[dataStart+4:dataStart+8], uint32(dataBytes))
	payload := out[dataStart+wavChunkHeaderBytes:]
	for index, sample := range samples {
		value := math.Round(float64(sample) * pcm16Magnitude)
		if value > pcm16Magnitude-1 {
			value = pcm16Magnitude - 1
		}
		if value < -pcm16Magnitude {
			value = -pcm16Magnitude
		}
		binary.LittleEndian.PutUint16(payload[index*pcm16Bytes:], uint16(int16(value)))
	}
	return out, nil
}
