package speechsynth

import (
	"errors"
	"math"
	"math/rand"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/workflowruntime"
)

var audioContract = artifact.JSONContract(artifact.KindFile, "overgo.speech-audio.v1")

type SynthesisRequest struct {
	Text      string `json:"text"`
	MaxFrames int    `json:"max_frames"`
	Seed      int64  `json:"seed"`
}

type Audio struct {
	PCM        []float32 `json:"pcm"`
	SampleRate int       `json:"sample_rate"`
}

func ValidateSynthesisRequest(request SynthesisRequest) error {
	if strings.TrimSpace(request.Text) == "" || request.MaxFrames <= 0 {
		return errors.New("speechsynth: synthesis requires text and a positive frame limit")
	}
	return nil
}

type Synthesizer struct {
	model     *Model
	tokenizer *Unigram
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
	return &Synthesizer{model: model, tokenizer: tokenizer}, nil
}

func (s *Synthesizer) Synthesize(request SynthesisRequest) (Audio, error) {
	if s == nil || s.model == nil || s.model.Codec == nil || s.tokenizer == nil {
		return Audio{}, errors.New("speechsynth: incomplete synthesizer")
	}
	if err := ValidateSynthesisRequest(request); err != nil {
		return Audio{}, err
	}
	tokens, err := s.tokenizer.Encode(request.Text)
	if err != nil {
		return Audio{}, err
	}
	random := rand.New(rand.NewSource(request.Seed))
	latents, _, err := s.model.GenerateLatents(nil, 0, tokens, GenerateParams{
		MaxFrames: request.MaxFrames, EOSThreshold: math.Inf(1),
		NoiseAt: func(_ int, values []float32) {
			for index := range values {
				values[index] = float32(random.NormFloat64())
			}
		},
	})
	if err != nil {
		return Audio{}, err
	}
	pcm, err := s.model.LatentsToPCM(latents)
	if err != nil {
		return Audio{}, err
	}
	return Audio{PCM: pcm, SampleRate: s.model.Codec.SampleRate}, nil
}

type synthesisEngine interface {
	Synthesize(SynthesisRequest) (Audio, error)
}

// RegisterRuntime binds one synthesizer to its recipe stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, synthesizer *Synthesizer) error {
	if synthesizer == nil {
		return errors.New("speechsynth: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, synthesizer)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, synthesizer synthesisEngine) error {
	if synthesizer == nil {
		return errors.New("speechsynth: incomplete runtime binding")
	}
	return workflowruntime.RegisterJSONStage[SynthesisRequest, Audio](
		runtime, modelrecipe.ModuleSpeechSynthesize, modelID, audioContract,
		func(request SynthesisRequest) (Audio, error) {
			if err := ValidateSynthesisRequest(request); err != nil {
				return Audio{}, err
			}
			return synthesizer.Synthesize(request)
		},
	)
}
