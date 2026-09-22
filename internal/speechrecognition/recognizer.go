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
	encoder *Encoder
	// transducer is set when the checkpoint declares a recurrent decoder
	// instead of a connectionist classifier. The two read the same acoustic
	// encoder and differ only in how frames become tokens.
	transducer *Transducer
	tokenizer  *hfbpe.Tokenizer
	frontend   *audiodsp.Frontend
	grouping   audiodsp.GroupedFeatureConfig
	blank      int
}

// RecognizerSpec is what a checkpoint declares beyond its weights: how its
// encoder is bound, how its waveform becomes features, and which vocabulary
// entry the connectionist classifier reads as blank.
type RecognizerSpec struct {
	Declaration Declaration
	Frontend    audiodsp.FrontendConfig
	Grouping    audiodsp.GroupedFeatureConfig
	Blank       int
	// Transducer is the recurrent decoder's binding when the checkpoint
	// declares one. Such a checkpoint reads the frontend's own frames and
	// names its blank entry in this binding, so the grouping and the blank
	// above belong to the connectionist form alone.
	Transducer *TransducerBinding
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
	var encoder *Encoder
	var transducer *Transducer
	if spec.Transducer == nil {
		encoder, err = LoadEncoder(ctx, source, spec.Declaration, memoryBytes)
	} else {
		transducer, err = LoadTransducer(ctx, source, spec.Declaration, *spec.Transducer, memoryBytes)
		if err == nil {
			encoder = transducer.encoder
		}
	}
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
		encoder: encoder, transducer: transducer, tokenizer: tokenizer, frontend: frontend,
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
	if r.transducer != nil {
		return r.transcribeRecurrent(ctx, audio.Samples, rate, &frontendWork)
	}
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
	// A persisted transcript never silently drops what the model emitted, so
	// an unknown identifier or invalid text is an error here.
	return r.tokenizer.DecodeStrict(ids)
}

// transcribeRecurrent reads a recording through the recurrent decoder: the
// frontend's own frames, with no stacking or deltas, because the binding's
// subsampling consumes them as the encoder declares. The decoder emits the
// tokens it kept, and an encode that produced no frames is an error rather
// than an empty transcript, as it is on the connectionist path.
func (r *Recognizer) transcribeRecurrent(ctx context.Context, samples []float32, rate int, work *audiodsp.Workspace) (string, error) {
	features, frames, err := r.frontend.Process(ctx, [][]float32{samples}, rate, work, audiodsp.ProcessOptions{})
	if err != nil {
		return "", err
	}
	if frames == 0 {
		return "", errors.New("speechrecognition: the frontend produced no frames")
	}
	var decoderWork TransducerWorkspace
	result, err := r.transducer.Recognize(ctx, features, frames, &decoderWork, nil)
	if err != nil {
		return "", err
	}
	if result.Frames == 0 {
		return "", errors.New("speechrecognition: the encoder produced no frames")
	}
	// The recurrent decoder emits the blank and the control entries its
	// vocabulary marks special; the transcript is what remains once the
	// artifact's own marking is honoured.
	return r.tokenizer.DecodeText(result.Tokens)
}
