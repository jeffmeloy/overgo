// Package audiodsp owns CPU audio signal measurement and processing.
package audiodsp

import (
	"context"
	"errors"
	"math"

	"overgo/internal/checked"
	"overgo/internal/media"
	"overgo/internal/recipecontract"
)

// MeasureSignal measures every interleaved scalar, retaining non-finite and
// clipped counts without modifying the signal. RMS and DC use finite samples
// only; policy separately decides whether any non-finite input is admissible.
func MeasureSignal(ctx context.Context, source recipecontract.AudioReference, audio media.DecodedAudio, clipThreshold float64) (recipecontract.DecodedAudioSignalProfile, error) {
	if ctx == nil {
		return recipecontract.DecodedAudioSignalProfile{}, errors.New("audio signal: nil context")
	}
	if err := ctx.Err(); err != nil {
		return recipecontract.DecodedAudioSignalProfile{}, err
	}
	if err := source.Validate(); err != nil {
		return recipecontract.DecodedAudioSignalProfile{}, err
	}
	if err := audio.Format.Validate(); err != nil {
		return recipecontract.DecodedAudioSignalProfile{}, err
	}
	if audio.Format.Encoding != "pcm-f32le" || uint64(len(audio.Samples))%uint64(audio.Format.Channels) != 0 || !checked.PositiveFinite64(clipThreshold) {
		return recipecontract.DecodedAudioSignalProfile{}, errors.New("audio signal: invalid decoded layout or clip threshold")
	}
	profile := recipecontract.DecodedAudioSignalProfile{
		Source: source, SchemaValid: true, DecodeStatus: recipecontract.AudioDecodeComplete,
		Format: audio.Format, SampleCount: uint64(len(audio.Samples)),
		FrameCount: uint64(len(audio.Samples)) / uint64(audio.Format.Channels), ClipThreshold: clipThreshold,
	}
	var sum, squares, sumCorrection, squaresCorrection float64
	for _, scalar := range audio.Samples {
		value := float64(scalar)
		if !checked.Finite64(value) {
			profile.NonFiniteSampleCount++
			continue
		}
		profile.FiniteSampleCount++
		absolute := math.Abs(value)
		profile.PeakAbsolute = max(profile.PeakAbsolute, absolute)
		if absolute >= clipThreshold {
			profile.ClippedSampleCount++
		}
		// Float32 squares and an addressable slice's count cannot overflow a
		// float64 accumulator. Compensated sums limit cancellation error.
		adjusted := value - sumCorrection
		next := sum + adjusted
		sumCorrection, sum = (next-sum)-adjusted, next
		adjusted = value*value - squaresCorrection
		next = squares + adjusted
		squaresCorrection, squares = (next-squares)-adjusted, next
	}
	if profile.FiniteSampleCount != 0 {
		count := float64(profile.FiniteSampleCount)
		// Enforce mathematical bounds after floating-point rounding.
		profile.RootMeanSquare = min(profile.PeakAbsolute, math.Sqrt(squares/count))
		profile.DCOffset = min(profile.RootMeanSquare, max(-profile.RootMeanSquare, sum/count))
	}
	if err := ctx.Err(); err != nil {
		return recipecontract.DecodedAudioSignalProfile{}, err
	}
	return profile, profile.Validate()
}
