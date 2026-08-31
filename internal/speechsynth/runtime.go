package speechsynth

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/modelrecipe"
	"overgo/internal/tensor"
	"overgo/internal/workflowruntime"
)

var audioContract = artifact.JSONContract(artifact.KindFile, "overgo.speech-audio.v1")

type SynthesisRequest struct {
	Text      string `json:"text"`
	MaxFrames int    `json:"max_frames"`
	Seed      int64  `json:"seed"`
	// Voice names an exported artifact voice; the model was trained to
	// speak behind a voice prompt, and generating without one drifts
	// into noise, so a voice is required.
	Voice string `json:"voice"`
	// Temperature scales the flow noise variance (std = sqrt(temp));
	// zero defers to the artifact's declared default.
	Temperature float64 `json:"temperature,omitzero"`
}

type Audio struct {
	PCM        []float32 `json:"pcm"`
	SampleRate int       `json:"sample_rate"`
	Channels   int       `json:"channels"`
}

type generationPlan struct {
	tokens      []int
	maxFrames   int
	seed        int64
	voice       string
	temperature float64
}

func ValidateSynthesisRequest(request SynthesisRequest) error {
	if strings.TrimSpace(request.Text) == "" || request.MaxFrames <= tensor.FirstOffset {
		return errors.New("speechsynth: synthesis requires text and a positive frame limit")
	}
	if strings.TrimSpace(request.Voice) == "" {
		return errors.New("speechsynth: synthesis requires a voice; the model speaks behind a voice prompt")
	}
	if request.Temperature < 0 {
		return errors.New("speechsynth: negative temperature")
	}
	return nil
}

// referenceEOSThreshold is the reference decode API's end-of-speech
// logit threshold; the artifact config declares no override.
const referenceEOSThreshold = -4.0

type Synthesizer struct {
	model     *Model
	tokenizer *Unigram
	directory string
	// defaultTemperature is the artifact-declared flow noise variance
	// used when a request does not state one.
	defaultTemperature float64
	// voices caches loaded voice states by name.
	voices map[string]*VoiceState
}

func LoadSynthesizer(directory string) (*Synthesizer, error) {
	model, err := Load(directory)
	if err != nil {
		return nil, err
	}
	tokenizer, err := LoadTokenizer(filepath.Join(directory, "tokenizer.model"))
	if err != nil {
		return nil, err
	}
	var config struct {
		DefaultTemperature float64 `json:"default_temperature"`
	}
	if err := jsonfile.Decode(filepath.Join(directory, "pockettts_config.json"), &config); err != nil {
		return nil, fmt.Errorf("speechsynth: artifact config: %w", err)
	}
	if config.DefaultTemperature <= 0 {
		return nil, errors.New("speechsynth: artifact declares no default temperature")
	}
	return &Synthesizer{
		model: model, tokenizer: tokenizer, directory: directory,
		defaultTemperature: config.DefaultTemperature, voices: map[string]*VoiceState{},
	}, nil
}

func (s *Synthesizer) complete() bool {
	return s != nil && s.model != nil && s.model.Codec != nil && s.tokenizer != nil
}

func (s *Synthesizer) tokenize(request SynthesisRequest) (generationPlan, error) {
	if !s.complete() {
		return generationPlan{}, errors.New("speechsynth: incomplete synthesizer")
	}
	if err := ValidateSynthesisRequest(request); err != nil {
		return generationPlan{}, err
	}
	tokens, err := s.tokenizer.Encode(request.Text)
	if err != nil {
		return generationPlan{}, err
	}
	temperature := request.Temperature
	if temperature == 0 {
		temperature = s.defaultTemperature
	}
	return generationPlan{
		tokens: tokens, maxFrames: request.MaxFrames, seed: request.Seed,
		voice: strings.TrimSpace(request.Voice), temperature: temperature,
	}, nil
}

// voiceState resolves and caches one named artifact voice.
func (s *Synthesizer) voiceState(name string) (*VoiceState, error) {
	if voice, ok := s.voices[name]; ok {
		return voice, nil
	}
	path, err := ResolveVoicePath(s.directory, name)
	if err != nil {
		return nil, err
	}
	voice, err := LoadVoiceState(path, s.model.Dims)
	if err != nil {
		return nil, err
	}
	s.voices[name] = voice
	return voice, nil
}

func (s *Synthesizer) generate(plan generationPlan) (LatentBatch, error) {
	if !s.complete() || len(plan.tokens) == tensor.FirstOffset || plan.maxFrames <= tensor.FirstOffset ||
		plan.voice == "" || plan.temperature <= 0 {
		return LatentBatch{}, errors.New("speechsynth: invalid generation plan")
	}
	voice, err := s.voiceState(plan.voice)
	if err != nil {
		return LatentBatch{}, err
	}
	random := rand.New(rand.NewSource(plan.seed))
	deviation := math.Sqrt(plan.temperature)
	latents, _, err := s.model.GenerateLatentsWithVoice(voice, plan.tokens, GenerateParams{
		MaxFrames: plan.maxFrames, EOSThreshold: referenceEOSThreshold,
		NoiseAt: func(_ int, values []float32) {
			for index := range values {
				values[index] = float32(deviation * random.NormFloat64())
			}
		},
	})
	if err != nil {
		return LatentBatch{}, err
	}
	return latents, nil
}

func (s *Synthesizer) decode(latents LatentBatch) (Audio, error) {
	if !s.complete() {
		return Audio{}, errors.New("speechsynth: incomplete synthesizer")
	}
	pcm, err := s.model.LatentsToPCM(latents)
	if err != nil {
		return Audio{}, err
	}
	return Audio{PCM: pcm, SampleRate: s.model.Codec.SampleRate, Channels: tensor.SingletonExtent}, nil
}

// RegisterRuntime binds one synthesizer to its recipe stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, synthesizer *Synthesizer) error {
	if synthesizer == nil {
		return errors.New("speechsynth: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONPipeline(
		runtime, modelID, audioContract,
		modelrecipe.ModuleSpeechTokenize, synthesizer.tokenize,
		modelrecipe.ModuleSpeechGenerate, synthesizer.generate,
		modelrecipe.ModuleSpeechDecode, synthesizer.decode,
	)
}
