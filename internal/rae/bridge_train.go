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

// BridgeTrainer trains a bridge against the picture the decoder draws from
// what it maps: the squared error between that picture and the frame is
// carried back through the frozen decoder to the bridged tokens, and from
// them to the bridge. Only the bridge moves; the encoder's tokens are given
// and the decoder is read.
type BridgeTrainer struct {
	decoder   *vitDecoder
	source    int
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
	source, target := pair.bridge.widths()
	matrix := target * source
	weights := make([]float32, matrix+target)
	copy(weights, pair.bridge.Weight)
	copy(weights[matrix:], pair.bridge.Bias)
	plan, err := optimizer.CompilePlan(len(weights), []optimizer.GroupSpec{
		{Name: bridgeWeightGroup, Start: 0, End: matrix, Rows: target, Cols: source},
		{Name: bridgeBiasGroup, Start: matrix, End: len(weights), Rows: 1, Cols: target},
	})
	if err != nil {
		return nil, err
	}
	trainer := &BridgeTrainer{decoder: pair.decoder, source: source, weights: weights, gradients: make([]float32, len(weights)), plan: plan}
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
	matrix := t.decoder.latent * t.source
	return EncoderBridge{Weight: append([]float32(nil), t.weights[:matrix]...), Bias: append([]float32(nil), t.weights[matrix:]...)}
}

// SaveWeights writes the trained bridge's two tensors to a safetensors file.
func (t *BridgeTrainer) SaveWeights(path string) error {
	bridge := t.Bridge()
	_, target := bridge.widths()
	return safetensors.Save(path,
		map[string][]float32{bridgeWeightTensor: bridge.Weight, bridgeBiasTensor: bridge.Bias},
		map[string][]int{bridgeWeightTensor: {target, t.source}, bridgeBiasTensor: {target}}, nil)
}

// SavePair writes the trained bridge as a bridged pair's directory, naming
// the encoder and the collection beside it, as SaveEncoderBridge does.
func (t *BridgeTrainer) SavePair(directory, sourceEncoder, collection string) error {
	return SaveEncoderBridge(directory, t.Bridge(), filepath.Base(filepath.Clean(sourceEncoder)), filepath.Base(filepath.Clean(collection)))
}

// Loss is the mean squared error, over every value of every example's
// picture, of what the decoder draws from the bridged tokens.
func (t *BridgeTrainer) Loss(ctx context.Context, examples []BridgeExample) (float64, error) {
	return t.pass(ctx, examples, false)
}

// Step takes one optimizer step on the examples as one batch and returns
// the loss the step started from.
func (t *BridgeTrainer) Step(ctx context.Context, examples []BridgeExample) (float64, error) {
	loss, err := t.pass(ctx, examples, true)
	if err != nil {
		return 0, err
	}
	t.gradient = t.stepper.Step().GradientL2
	return loss, nil
}

// pass runs the examples forward and, when asked, leaves the batch's mean
// gradient with respect to the bridge in the gradient buffer.
func (t *BridgeTrainer) pass(ctx context.Context, examples []BridgeExample, backward bool) (float64, error) {
	if len(examples) == 0 {
		return 0, errors.New("rae: no examples to train the bridge on")
	}
	target := t.decoder.latent
	matrix := target * t.source
	bridge := EncoderBridge{Weight: t.weights[:matrix], Bias: t.weights[matrix:]}
	if backward {
		clear(t.gradients)
	}
	var total float64
	for index, example := range examples {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		mapped, err := bridge.Apply(example.SourceTokens)
		if err != nil {
			return 0, fmt.Errorf("rae: example %d: %w", index, err)
		}
		pixels, trace, err := t.decoder.decodeTraced(mapped)
		if err != nil {
			return 0, fmt.Errorf("rae: example %d: %w", index, err)
		}
		if len(pixels) != len(example.Picture) {
			return 0, fmt.Errorf("rae: example %d's picture holds %d values, the decoder draws %d", index, len(example.Picture), len(pixels))
		}
		// The loss is the mean over the batch of each picture's mean squared
		// error, so a value's share of it is its error over both counts.
		share := float64(len(pixels)) * float64(len(examples))
		dPixels := make([]float32, len(pixels))
		var squared float64
		for at, value := range pixels {
			difference := float64(value) - float64(example.Picture[at])
			squared += difference * difference
			dPixels[at] = float32((difference + difference) / share)
		}
		total += squared / share
		if !backward {
			continue
		}
		dTokens, err := t.decoder.backward(trace, dPixels, nil)
		if err != nil {
			return 0, err
		}
		rows := len(mapped) / target
		hostmath.LinearBackward(nil, t.gradients[:matrix], t.gradients[matrix:], example.SourceTokens, bridge.Weight, dTokens, rows, t.source, target, false)
	}
	if math.IsNaN(total) || math.IsInf(total, 0) {
		return 0, errors.New("rae: the bridge's loss is not finite")
	}
	return total, nil
}
