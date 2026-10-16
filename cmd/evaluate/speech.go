package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/media"
	"overgo/internal/mediacapability"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/sequencescore"
	"overgo/internal/speechrecognition"
	"overgo/internal/speechsynth"
	"overgo/internal/tokenizer"
)

const (
	// speechJudgeMemoryBytes is the host ceiling the judging recognizer's encoder and
	// frontend are admitted to work within.
	speechJudgeMemoryBytes = 4 << 30
	// defaultSpeechVoice is the voice a round-trip suite reads in unless a run names another.
	defaultSpeechVoice = "alba"
	// defaultSpeechSeed fixes the speech model's randomness in a round-trip suite.
	defaultSpeechSeed = 7
)

// speechRuntime is a speech model and the recognizer that judges it as an
// evaluation runtime: it reads a text aloud, writes down what the judge heard,
// and generates and scores no text.
type speechRuntime struct {
	synthesizer *speechsynth.Synthesizer
	recognizer  *speechrecognition.Recognizer
	judge       string
}

// RoundTrip has the model read the text aloud in the voice, and returns the
// judge's transcript of the recording.
func (r speechRuntime) RoundTrip(ctx context.Context, text, voice string, seed int64) (string, error) {
	audio, err := r.synthesizer.Speak(ctx, speechsynth.DocumentRequest{Text: text, Voice: voice, Seed: seed})
	if err != nil {
		return "", err
	}
	heard, err := resampleTo(ctx, audio.PCM, audio.SampleRate, r.recognizer.SampleRate())
	if err != nil {
		return "", err
	}
	recording, err := media.EncodeWAVPCM16(heard, r.recognizer.SampleRate())
	if err != nil {
		return "", err
	}
	return r.recognizer.Transcribe(ctx, speechrecognition.TranscriptionRequest{
		Audio: recording, MaximumSamples: uint64(len(heard)), MemoryBytes: speechJudgeMemoryBytes,
	})
}

// resampleHalfWidth is how many zero crossings of the low-pass each side of a
// tap's centre the resampler's filter keeps: more is a sharper cut-off.
const resampleHalfWidth = 16

// nyquistFraction is the highest frequency a sample rate can hold, as a fraction of the rate.
const nyquistFraction = 0.5

// fullTurn is the radians in one cycle.
const fullTurn = math.Pi + math.Pi

// resampleTo converts samples from one rate to another through a polyphase
// windowed-sinc filter that cuts off at the lower of the two rates' Nyquist
// frequencies, so a recording is heard by a recognizer at the rate it declares.
func resampleTo(ctx context.Context, samples []float32, from, to int) ([]float32, error) {
	if from <= 0 || to <= 0 {
		return nil, fmt.Errorf("evaluate: resample from %d Hz to %d Hz", from, to)
	}
	if from == to {
		return samples, nil
	}
	divisor, remainder := from, to
	for remainder != 0 {
		divisor, remainder = remainder, divisor%remainder
	}
	up, down := to/divisor, from/divisor
	half := resampleHalfWidth * max(up, down)
	taps := make([]float64, half+half+1)
	cutoff := nyquistFraction / float64(max(up, down))
	// The window reaches its zero one tap past each end of the filter, twice the half-width plus two taps across.
	span := float64(half + half + 1 + 1)
	for index := range taps {
		offset := float64(index - half)
		sinc := cutoff + cutoff
		if offset != 0 {
			sinc = math.Sin(fullTurn*cutoff*offset) / (math.Pi * offset)
		}
		// The Hann window over the filter's support, written as the square of a cosine.
		cosine := math.Cos(math.Pi * offset / span)
		taps[index] = float64(up) * sinc * cosine * cosine
	}
	return media.ResamplePoly(ctx, nil, samples, up, down, taps)
}

// Judge names the recognizer that listens.
func (r speechRuntime) Judge() string { return r.judge }

// Generate refuses: a speech model writes no text.
func (speechRuntime) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("evaluate: a speech model generates no text")
}

// ScoreContinuations refuses: a speech model scores no text.
func (speechRuntime) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("evaluate: a speech model scores no continuation")
}

// openSpeechSession opens the speech model at the request's path under its
// active speech recipe, judged by the recognizer the request names.
func openSpeechSession(ctx context.Context, store *overgodb.Store, value manifest, request modelRequest) (evaluationSession, error) {
	if request.Judge == "" {
		return nil, fmt.Errorf("evaluate: %s is a speech model, which a recognizer judges, and the request names none", request.Path)
	}
	source, err := mediacapability.Catalog[recipe.TaskSpeech].Resolve(request.Path)
	if err != nil {
		return nil, fmt.Errorf("evaluate: %s is no speech model: %w", request.Path, err)
	}
	return openCapabilitySession(ctx, store, value, source, recipe.TaskSpeech, request.Recipe, func(definition recipe.Definition) (evaluation.Runtime, func(context.Context) error, error) {
		synthesizer, err := speechsynth.LoadSynthesizer(request.Path)
		if err != nil {
			return nil, nil, err
		}
		if err := synthesizer.RestoreRecipe(ctx, store, definition); err != nil {
			return nil, nil, err
		}
		recognizer, err := mediacapability.OpenRecognizer(ctx, request.Judge, speechJudgeMemoryBytes)
		return speechRuntime{synthesizer: synthesizer, recognizer: recognizer, judge: filepath.Base(request.Judge)},
			func(context.Context) error { return nil }, err
	})
}

// writeSpeechRoundTrip builds the round-trip suite of LibriSpeech test-clean's
// transcripts and writes it to out.
func writeSpeechRoundTrip(ctx context.Context, parquet, judge, voice, out string, cases int, seed uint64) error {
	suite, err := evaluation.BuildLibriSpeechRoundTrip(ctx, evaluation.SpeechRoundTripSpec{
		Parquet: parquet, Cases: cases, Seed: seed, Voice: voice, SynthesisSeed: defaultSpeechSeed, Judge: filepath.Base(judge),
	})
	if err != nil {
		return err
	}
	return writeSuite(out, suite, len(suite.Cases), "speech round-trip")
}
