package rae

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/hostmath"
	"overgo/internal/optimizer"
	"overgo/internal/safetensors"
)

// BridgeExample is one picture a bridge is trained on: the source encoder's
// tokens for it, which the bridge maps, and the picture itself, square of the
// pair's side and channel-first in [0, 1], which the decoder should draw.
type BridgeExample struct {
	SourceTokens []float32
	Picture      []float32
}

// BridgeSample is one example a bridge is trained on through a frozen model:
// the source encoder's tokens, which the bridge maps, and the model's loss of
// what the bridge maps them to, which with backward also gives the loss's
// gradient with respect to the mapped tokens.
type BridgeSample struct {
	Tokens []float32
	Loss   func(mapped []float32, backward bool) (float64, []float32, error)
}

// BridgeTrainer trains a bridge against what a frozen model makes of what it
// maps: the model's loss is carried back through it to the mapped tokens, and
// from them to the bridge. Only the bridge moves; the encoder's tokens are
// given and the model is read. A pair's trainer holds the pair's decoder,
// whose picture against the frame is its examples' loss.
type BridgeTrainer struct {
	decoder   *vitDecoder
	source    int
	target    int
	weights   []float32
	gradients []float32
	plan      optimizer.Plan
	stepper   *optimizer.Optimizer
	gradient  float64
}

// The bridge's two tensors as the optimizer names them.
const (
	bridgeWeightGroup = "bridge.weight"
	bridgeBiasGroup   = "bridge.bias"
)

// NewBridgeTrainer starts training the bridge of a bridged pair from the
// weights it holds, under an optimizer configuration.
func NewBridgeTrainer(pair *BridgedAutoencoder, config optimizer.Config) (*BridgeTrainer, error) {
	if pair == nil {
		return nil, errors.New("rae: no bridged pair to train")
	}
	trainer, err := NewEncoderBridgeTrainer(pair.bridge, config)
	if err != nil {
		return nil, err
	}
	trainer.decoder = pair.decoder
	return trainer, nil
}

// NewEncoderBridgeTrainer starts training bridge from the weights it holds,
// under an optimizer configuration, against the frozen model its samples'
// losses read.
func NewEncoderBridgeTrainer(bridge EncoderBridge, config optimizer.Config) (*BridgeTrainer, error) {
	source, target := bridge.widths()
	if source == 0 {
		return nil, errors.New("rae: a bridge to train needs its weights and offsets")
	}
	matrix := target * source
	weights := make([]float32, matrix+target)
	copy(weights, bridge.Weight)
	copy(weights[matrix:], bridge.Bias)
	plan, err := optimizer.CompilePlan(len(weights), []optimizer.GroupSpec{
		{Name: bridgeWeightGroup, Start: 0, End: matrix, Rows: target, Cols: source},
		{Name: bridgeBiasGroup, Start: matrix, End: len(weights), Rows: 1, Cols: target},
	})
	if err != nil {
		return nil, err
	}
	trainer := &BridgeTrainer{source: source, target: target, weights: weights, gradients: make([]float32, len(weights)), plan: plan}
	if trainer.stepper, err = optimizer.New(trainer.weights, trainer.gradients, plan, config); err != nil {
		return nil, err
	}
	return trainer, nil
}

// TrainableParameterCount is the count of the bridge's weights and offsets.
func (t *BridgeTrainer) TrainableParameterCount() int { return len(t.weights) }

// OptimizerPlan is the layout the bridge trains under.
func (t *BridgeTrainer) OptimizerPlan() optimizer.Plan { return t.plan }

// Snapshot is the optimizer's state after the steps taken.
func (t *BridgeTrainer) Snapshot() optimizer.State { return t.stepper.Snapshot() }

// GradientL2 is the size of the last step's gradient.
func (t *BridgeTrainer) GradientL2() float64 { return t.gradient }

// Bridge is the bridge as trained so far.
func (t *BridgeTrainer) Bridge() EncoderBridge {
	matrix := t.target * t.source
	return EncoderBridge{Weight: append([]float32(nil), t.weights[:matrix]...), Bias: append([]float32(nil), t.weights[matrix:]...)}
}

// SaveWeights writes the trained bridge's two tensors to a safetensors file.
func (t *BridgeTrainer) SaveWeights(path string) error {
	return SaveBridgeWeights(path, t.Bridge())
}

// SaveBridgeWeights writes a bridge's two tensors to a safetensors file: its
// weight, a row of source weights for each target channel, and its offsets.
func SaveBridgeWeights(path string, bridge EncoderBridge) error {
	source, target := bridge.widths()
	if source == 0 {
		return errors.New("rae: a bridge to save needs its weights and offsets")
	}
	return safetensors.Save(path,
		map[string][]float32{bridgeWeightTensor: bridge.Weight, bridgeBiasTensor: bridge.Bias},
		map[string][]int{bridgeWeightTensor: {target, source}, bridgeBiasTensor: {target}}, nil)
}

// SavePair writes the trained bridge as a bridged pair's directory, naming
// the encoder and the collection beside it, as SaveEncoderBridge does.
func (t *BridgeTrainer) SavePair(directory, sourceEncoder, collection string) error {
	return SaveEncoderBridge(directory, t.Bridge(), filepath.Base(filepath.Clean(sourceEncoder)), filepath.Base(filepath.Clean(collection)))
}

// Loss is the mean squared error, over every value of every example's
// picture, of what the decoder draws from the bridged tokens.
func (t *BridgeTrainer) Loss(ctx context.Context, examples []BridgeExample) (float64, error) {
	return t.LossThrough(ctx, t.drawnSamples(examples))
}

// Step takes one optimizer step on the examples as one batch and returns
// the loss the step started from.
func (t *BridgeTrainer) Step(ctx context.Context, examples []BridgeExample) (float64, error) {
	return t.StepThrough(ctx, t.drawnSamples(examples))
}

// LossThrough is the mean over the samples of the loss each gives what the
// bridge maps its tokens to.
func (t *BridgeTrainer) LossThrough(ctx context.Context, samples []BridgeSample) (float64, error) {
	return t.through(ctx, samples, false)
}

// StepThrough takes one optimizer step on the samples as one batch and
// returns the mean loss the step started from.
func (t *BridgeTrainer) StepThrough(ctx context.Context, samples []BridgeSample) (float64, error) {
	loss, err := t.through(ctx, samples, true)
	if err != nil {
		return 0, err
	}
	t.gradient = t.stepper.Step().GradientL2
	return loss, nil
}

// drawnSamples are a pair's examples as samples whose loss is the mean
// squared error, over every value of the picture, of what the decoder draws.
func (t *BridgeTrainer) drawnSamples(examples []BridgeExample) []BridgeSample {
	samples := make([]BridgeSample, len(examples))
	for index, example := range examples {
		samples[index] = BridgeSample{Tokens: example.SourceTokens, Loss: func(mapped []float32, backward bool) (float64, []float32, error) {
			if t.decoder == nil {
				return 0, nil, errors.New("the trainer holds no decoder to draw its examples")
			}
			pixels, trace, err := t.decoder.decodeTraced(mapped)
			if err != nil {
				return 0, nil, err
			}
			if len(pixels) != len(example.Picture) {
				return 0, nil, fmt.Errorf("the picture holds %d values, the decoder draws %d", len(example.Picture), len(pixels))
			}
			count := float64(len(pixels))
			dPixels := make([]float32, len(pixels))
			var squared float64
			for at, value := range pixels {
				difference := float64(value) - float64(example.Picture[at])
				squared += difference * difference
				dPixels[at] = float32((difference + difference) / count)
			}
			if !backward {
				return squared / count, nil, nil
			}
			dTokens, err := t.decoder.backward(trace, dPixels, nil)
			return squared / count, dTokens, err
		}}
	}
	return samples
}

// through runs the samples forward and, when asked, leaves the batch's mean
// gradient with respect to the bridge in the gradient buffer.
func (t *BridgeTrainer) through(ctx context.Context, samples []BridgeSample, backward bool) (float64, error) {
	if len(samples) == 0 {
		return 0, errors.New("rae: no examples to train the bridge on")
	}
	matrix := t.target * t.source
	bridge := EncoderBridge{Weight: t.weights[:matrix], Bias: t.weights[matrix:]}
	if backward {
		clear(t.gradients)
	}
	// The loss is the mean over the batch, so a sample's gradient is its own
	// over the batch's count.
	share := float32(1) / float32(len(samples))
	var total float64
	for index, sample := range samples {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		mapped, err := bridge.Apply(sample.Tokens)
		if err != nil {
			return 0, fmt.Errorf("rae: example %d: %w", index, err)
		}
		loss, dMapped, err := sample.Loss(mapped, backward)
		if err != nil {
			return 0, fmt.Errorf("rae: example %d: %w", index, err)
		}
		total += loss / float64(len(samples))
		if !backward {
			continue
		}
		if len(dMapped) != len(mapped) {
			return 0, fmt.Errorf("rae: example %d's loss gave %d gradient values for %d mapped", index, len(dMapped), len(mapped))
		}
		for at := range dMapped {
			dMapped[at] *= share
		}
		hostmath.LinearBackward(nil, t.gradients[:matrix], t.gradients[matrix:], sample.Tokens, bridge.Weight, dMapped, len(mapped)/t.target, t.source, t.target, false)
	}
	if math.IsNaN(total) || math.IsInf(total, 0) {
		return 0, errors.New("rae: the bridge's loss is not finite")
	}
	return total, nil
}
