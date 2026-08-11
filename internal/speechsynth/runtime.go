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

type GenerationPlan struct {
	Tokens    []int
	MaxFrames int
	Seed      int64
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

func (s *Synthesizer) complete() bool {
	return s != nil && s.model != nil && s.model.Codec != nil && s.tokenizer != nil
}

func (s *Synthesizer) Tokenize(request SynthesisRequest) (GenerationPlan, error) {
	if !s.complete() {
		return GenerationPlan{}, errors.New("speechsynth: incomplete synthesizer")
	}
	if err := ValidateSynthesisRequest(request); err != nil {
		return GenerationPlan{}, err
	}
	tokens, err := s.tokenizer.Encode(request.Text)
	if err != nil {
		return GenerationPlan{}, err
	}
	return GenerationPlan{Tokens: tokens, MaxFrames: request.MaxFrames, Seed: request.Seed}, nil
}

func (s *Synthesizer) Generate(plan GenerationPlan) (LatentBatch, error) {
	if !s.complete() || len(plan.Tokens) == 0 || plan.MaxFrames <= 0 {
		return LatentBatch{}, errors.New("speechsynth: invalid generation plan")
	}
	random := rand.New(rand.NewSource(plan.Seed))
	latents, _, err := s.model.GenerateLatents(nil, 0, plan.Tokens, GenerateParams{
		MaxFrames: plan.MaxFrames, EOSThreshold: math.Inf(1),
		NoiseAt: func(_ int, values []float32) {
			for index := range values {
				values[index] = float32(random.NormFloat64())
			}
		},
	})
	if err != nil {
		return LatentBatch{}, err
	}
	return latents, nil
}

func (s *Synthesizer) Decode(latents LatentBatch) (Audio, error) {
	if !s.complete() {
		return Audio{}, errors.New("speechsynth: incomplete synthesizer")
	}
	pcm, err := s.model.LatentsToPCM(latents)
	if err != nil {
		return Audio{}, err
	}
	return Audio{PCM: pcm, SampleRate: s.model.Codec.SampleRate}, nil
}

type runtimeStages interface {
	Tokenize(SynthesisRequest) (GenerationPlan, error)
	Generate(GenerationPlan) (LatentBatch, error)
	Decode(LatentBatch) (Audio, error)
}

// RegisterRuntime binds one synthesizer to its recipe stage.
func RegisterRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, synthesizer *Synthesizer) error {
	if synthesizer == nil {
		return errors.New("speechsynth: incomplete runtime binding")
	}
	return registerRuntime(runtime, modelID, synthesizer)
}

func registerRuntime(runtime *workflowruntime.Runtime, modelID artifact.ID, synthesizer runtimeStages) error {
	if synthesizer == nil {
		return errors.New("speechsynth: incomplete runtime binding")
	}
	if err := workflowruntime.RegisterScalarStage(
		runtime, modelrecipe.ModuleSpeechTokenize, modelID, synthesizer.Tokenize, nil,
	); err != nil {
		return err
	}
	if err := workflowruntime.RegisterScalarStage(
		runtime, modelrecipe.ModuleSpeechGenerate, modelID, synthesizer.Generate, nil,
	); err != nil {
		return err
	}
	return workflowruntime.RegisterJSONStage(
		runtime, modelrecipe.ModuleSpeechDecode, modelID, audioContract, synthesizer.Decode,
	)
}
