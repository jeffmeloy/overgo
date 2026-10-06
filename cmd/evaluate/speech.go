package main

import (
	"context"
	"errors"
	"fmt"
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
	recording, err := media.EncodeWAVPCM16(audio.PCM, audio.SampleRate)
	if err != nil {
		return "", err
	}
	return r.recognizer.Transcribe(ctx, speechrecognition.TranscriptionRequest{
		Audio: recording, MaximumSamples: uint64(len(audio.PCM)), MemoryBytes: speechJudgeMemoryBytes,
	})
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
	return openCapabilitySession(ctx, store, value, source, recipe.TaskSpeech, func() (evaluation.Runtime, func(context.Context) error, error) {
		synthesizer, err := speechsynth.LoadSynthesizer(request.Path)
		if err != nil {
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
