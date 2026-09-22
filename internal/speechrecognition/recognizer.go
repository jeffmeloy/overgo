package speechrecognition

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/audiodsp"
	"overgo/internal/checked"
	"overgo/internal/hfbpe"
	"overgo/internal/media"
	"overgo/internal/safetensors"
)

// Recognizer is one loaded speech model: the encoder its declaration
// describes, the head that projects frames to vocabulary logits, the
// tokenizer that reads those IDs back as text, and the frontend that turns a
// waveform into the features the encoder expects. It is the model-resident
// component a transcription session keeps across requests; nothing here
// names a model family, because the declaration carries every name and
// convention the encoder needs.
type Recognizer struct {
	encoder   *Encoder
	tokenizer *hfbpe.Tokenizer
	frontend  *audiodsp.Frontend
	grouping  audiodsp.GroupedFeatureConfig
	blank     int
}

// RecognizerSpec is what a checkpoint declares beyond its weights: how its
// encoder is bound, how its waveform becomes features, and which vocabulary
// entry the connectionist classifier reads as blank.
type RecognizerSpec struct {
	Declaration Declaration
	Frontend    audiodsp.FrontendConfig
	Grouping    audiodsp.GroupedFeatureConfig
	Blank       int
}

// LoadRecognizer assembles a recognizer from a checkpoint directory and the
// declarations that describe it. The tokenizer is the checkpoint's own.
func LoadRecognizer(ctx context.Context, directory string, spec RecognizerSpec, memoryBytes uint64) (*Recognizer, error) {
	if spec.Blank < 0 {
		return nil, errors.New("speechrecognition: the blank vocabulary entry is not declared")
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	encoder, err := LoadEncoder(ctx, source, spec.Declaration, memoryBytes)
	if err != nil {
		return nil, err
	}
	tokenizer, err := hfbpe.Load(directory)
	if err != nil {
		return nil, fmt.Errorf("speechrecognition: tokenizer: %w", err)
	}
	frontend, err := audiodsp.NewFrontend(spec.Frontend, memoryBytes)
	if err != nil {
		return nil, err
	}
	return &Recognizer{
		encoder: encoder, tokenizer: tokenizer, frontend: frontend,
		grouping: spec.Grouping, blank: spec.Blank,
	}, nil
}

// TranscriptionRequest is one recording to transcribe: the encoded audio and
// the sample bound the decoder is admitted to read, which the caller states
// rather than inheriting from the file.
type TranscriptionRequest struct {
	Audio          []byte `json:"audio"`
	MaximumSamples uint64 `json:"maximum_samples"`
	// MemoryBytes is the host ceiling the encoder and frontend are admitted
	// to work within, which every caller of this runtime states rather than
	// inheriting: an evaluation manifest, a workspace policy, a request.
	MemoryBytes uint64 `json:"memory_bytes"`
}

// ValidateTranscriptionRequest refuses a request that carries no recording or
// no bound, before any model is loaded.
func ValidateTranscriptionRequest(request TranscriptionRequest) error {
	if len(request.Audio) == 0 {
		return errors.New("speechrecognition: transcription requires encoded audio")
	}
	if request.MaximumSamples == 0 {
		return errors.New("speechrecognition: transcription requires a sample bound")
	}
	if request.MemoryBytes == 0 {
		return errors.New("speechrecognition: transcription requires a host memory ceiling")
	}
	return nil
}

// Transcribe decodes the recording, derives its features, encodes them,
// projects the frames to logits, collapses them by the connectionist rule and
// reads the surviving IDs back as text. Each step refuses rather than
// guessing, so an unreadable recording or an empty encode is an error and
// never an empty transcript.
func (r *Recognizer) Transcribe(ctx context.Context, request TranscriptionRequest) (string, error) {
	if r == nil || r.encoder == nil || r.tokenizer == nil || r.frontend == nil {
		return "", errors.New("speechrecognition: incomplete recognizer")
	}
	if err := ValidateTranscriptionRequest(request); err != nil {
		return "", err
	}
	audio, _, err := media.DecodeAudio(ctx, request.Audio, request.MaximumSamples)
	if err != nil {
		return "", err
	}
	rate, ok := checked.Int(audio.Format.SampleRate)
	if !ok || rate == 0 {
		return "", fmt.Errorf("speechrecognition: sample rate %d is out of range", audio.Format.SampleRate)
	}
	var frontendWork audiodsp.Workspace
	features, frames, _, err := r.frontend.ProcessGrouped(ctx, audio.Samples, rate, &frontendWork, r.grouping)
	if err != nil {
		return "", err
	}
	var encoderWork Workspace
	hidden, encoded, err := r.encoder.Encode(ctx, features, frames, &encoderWork, nil)
	if err != nil {
		return "", err
	}
	if encoded == 0 || len(hidden) == 0 {
		return "", errors.New("speechrecognition: the encoder produced no frames")
	}
	logits, err := r.encoder.Project(ctx, hidden, encoded, &encoderWork)
	if err != nil {
		return "", err
	}
	vocabulary := r.encoder.VocabularySize()
	if r.blank >= vocabulary {
		return "", fmt.Errorf("speechrecognition: blank entry %d is beyond the %d-entry vocabulary", r.blank, vocabulary)
	}
	ids, err := GreedyCTC(ctx, make([]int, encoded), make([]int, encoded), logits, encoded, vocabulary, r.blank)
	if err != nil {
		return "", err
	}
	return r.tokenizer.Decode(ids), nil
}
