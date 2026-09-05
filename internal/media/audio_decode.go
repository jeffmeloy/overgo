package media

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"

	"overgo/internal/recipecontract"
)

var (
	errAudioUnsupported = errors.New("media: unsupported audio encoding")
	errAudioLimit       = errors.New("media: decoded audio exceeds sample budget")
)

// DecodedAudio owns channel-interleaved PCM float32 samples at the source rate.
// Decoding does not clamp, normalize, resample, or suppress non-finite samples.
type DecodedAudio struct {
	Format  recipecontract.AudioFormat
	Samples []float32
}

// DecodeAudio decodes RIFF/WAVE or native FLAC, bounded by maximumSamples scalar
// values. Errors never return a partial signal. The status distinguishes source
// failures from resource refusal; cancellation remains a context error.
func DecodeAudio(ctx context.Context, data []byte, maximumSamples uint64) (DecodedAudio, recipecontract.AudioDecodeStatus, error) {
	if ctx == nil {
		return DecodedAudio{}, recipecontract.AudioDecodeNotAttempted, errors.New("media: nil audio decode context")
	}
	if err := ctx.Err(); err != nil {
		return DecodedAudio{}, recipecontract.AudioDecodeNotAttempted, err
	}
	if maximumSamples == 0 || maximumSamples > uint64(math.MaxInt)/float32Bytes {
		return DecodedAudio{}, recipecontract.AudioDecodeResourceLimit, errAudioLimit
	}
	var audio DecodedAudio
	var err error
	switch {
	case bytes.HasPrefix(data, []byte("RIFF")):
		audio, err = decodeWAV(data, maximumSamples)
	case bytes.HasPrefix(data, []byte("fLaC")):
		audio, err = decodeFLAC(ctx, data, maximumSamples)
	case len(data) == 0:
		err = io.ErrUnexpectedEOF
	default:
		err = errAudioUnsupported
	}
	if contextErr := ctx.Err(); contextErr != nil {
		return DecodedAudio{}, recipecontract.AudioDecodeNotAttempted, contextErr
	}
	if err == nil {
		return audio, recipecontract.AudioDecodeComplete, nil
	}
	status := recipecontract.AudioDecodeCorrupt
	switch {
	case errors.Is(err, errAudioUnsupported):
		status = recipecontract.AudioDecodeUnsupported
	case errors.Is(err, io.EOF), errors.Is(err, io.ErrUnexpectedEOF):
		status = recipecontract.AudioDecodeTruncated
	case errors.Is(err, errAudioLimit):
		status = recipecontract.AudioDecodeResourceLimit
	}
	return DecodedAudio{}, status, err
}
