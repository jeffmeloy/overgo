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
	Channels   int       `json:"channels"`
}

type generationPlan struct {
	tokens    []int
	maxFrames int
	seed      int64
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
	return generationPlan{tokens: tokens, maxFrames: request.MaxFrames, seed: request.Seed}, nil
}

func (s *Synthesizer) generate(plan generationPlan) (LatentBatch, error) {
	if !s.complete() || len(plan.tokens) == 0 || plan.maxFrames <= 0 {
		return LatentBatch{}, errors.New("speechsynth: invalid generation plan")
	}
	random := rand.New(rand.NewSource(plan.seed))
	latents, _, err := s.model.GenerateLatents(nil, 0, plan.tokens, GenerateParams{
		MaxFrames: plan.maxFrames, EOSThreshold: math.Inf(1),
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

func (s *Synthesizer) decode(latents LatentBatch) (Audio, error) {
	if !s.complete() {
		return Audio{}, errors.New("speechsynth: incomplete synthesizer")
	}
	pcm, err := s.model.LatentsToPCM(latents)
	if err != nil {
		return Audio{}, err
	}
	return Audio{PCM: pcm, SampleRate: s.model.Codec.SampleRate, Channels: 1}, nil
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
