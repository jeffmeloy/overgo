package scratchmodel

import (
	"errors"
	"math"
	"slices"
	"strconv"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
)

// TrainingResult reports the shared runtime's pre-update trajectory.
type TrainingResult struct {
	Losses         []float64
	ValidationLoss float64
}

// TrainShared executes the compiled scratch program through hostmath and Muon.
func (c Construction) TrainShared(totalSteps int) (TrainingResult, error) {
	if totalSteps <= 0 || len(c.split.Train) == 0 || len(c.split.Validation) == 0 {
		return TrainingResult{}, errors.New("scratch model: invalid shared training run")
	}
	weights := slices.Clone(c.weights)
	for index := range weights {
		if math.IsNaN(float64(weights[index])) || math.IsInf(float64(weights[index]), 0) {
			return TrainingResult{}, errors.New("scratch model: non-finite initialized weight")
		}
	}
	gradients := make([]float32, len(weights))
	model, err := c.bindMADTransformer(weights)
	if err != nil {
		return TrainingResult{}, err
	}
	gradient, err := c.bindMADTransformer(gradients)
	if err != nil {
		return TrainingResult{}, err
	}
	muon, err := optimizer.New(weights, gradients, c.optimizer, optimizer.Config{
		BaseLearningRate: c.config.BaseLR,
		Momentum:         c.config.MuonMomentum,
		Steps:            totalSteps,
		Schedule:         optimizer.ScheduleLinearDecay,
	})
	if err != nil {
		return TrainingResult{}, err
	}
	result := TrainingResult{Losses: make([]float64, totalSteps)}
	for step := range totalSteps {
		tokens, err := c.Tokens(c.split.Train[step%len(c.split.Train)])
		if err != nil {
			return TrainingResult{}, err
		}
		result.Losses[step], err = hostmath.MADTransformerLossAndGrad(model, gradient, tokens)
		if err != nil {
			return TrainingResult{}, err
		}
		muon.Step()
	}
	validation, err := c.Tokens(c.split.Validation[0])
	if err != nil {
		return TrainingResult{}, err
	}
	result.ValidationLoss, err = hostmath.MADTransformerLoss(model, validation)
	return result, err
}

func (c Construction) bindMADTransformer(slab []float32) (hostmath.MADTransformer, error) {
	view := func(name string) ([]float32, error) {
		binding, ok := c.bindings[name]
		if !ok || binding.start < 0 || binding.end > len(slab) || binding.start >= binding.end {
			return nil, errors.New("scratch model: parameter slab binding differs")
		}
		return slab[binding.start:binding.end], nil
	}
	token, err := view("wte")
	if err != nil {
		return hostmath.MADTransformer{}, err
	}
	head, err := view("lm_head")
	if err != nil {
		return hostmath.MADTransformer{}, err
	}
	position, err := view("wpe")
	if err != nil {
		return hostmath.MADTransformer{}, err
	}
	positionBias, err := view("pos_bias")
	if err != nil {
		return hostmath.MADTransformer{}, err
	}
	temperature, err := view("lt")
	if err != nil {
		return hostmath.MADTransformer{}, err
	}
	layers := make([]hostmath.MADTransformerLayer, c.config.LayerCount)
	for layer := range layers {
		prefix := "l" + strconv.Itoa(layer) + "."
		if layers[layer].QKV, err = view(prefix + "wqkv"); err != nil {
			return hostmath.MADTransformer{}, err
		}
		if layers[layer].Output, err = view(prefix + "wo"); err != nil {
			return hostmath.MADTransformer{}, err
		}
		if layers[layer].Expand, err = view(prefix + "w1"); err != nil {
			return hostmath.MADTransformer{}, err
		}
		if layers[layer].Contract, err = view(prefix + "w2"); err != nil {
			return hostmath.MADTransformer{}, err
		}
	}
	return hostmath.MADTransformer{
		Config: hostmath.MADTransformerConfig{
			Vocab: c.config.VocabSize, Block: c.config.BlockSize, Hidden: c.config.Embedding,
			Heads: c.config.HeadCount, HeadDim: c.config.HeadDim, Layers: c.config.LayerCount,
			Intermediate: c.config.MLPWidth, Window: c.config.AttentionWindow, Epsilon: c.config.Epsilon,
		},
		Token: token, Head: head, Position: position, PositionBias: positionBias,
		Temperature: temperature, Layer: layers,
	}, nil
}
