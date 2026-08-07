package tensor

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/tensor/dtype"
)

type ropeOptions struct {
	operation                     Op
	name                          string
	positions                     []uint32
	frequencyFactors              *Tensor
	rotaryDimensions              uint32
	frequencyBase, frequencyScale float32
	yarn                          bool
	originalContext               uint32
	extFactor, attentionFactor    float32
	betaFast, betaSlow            float32
}

// RoPELayout: rotary channel pairing.
type RoPELayout uint8

const (
	RoPELayoutNormal RoPELayout = iota
	RoPELayoutNeoX
)

// RoPEOptions: single-axis rotary graph controls.
type RoPEOptions struct {
	Layout           RoPELayout
	Positions        []uint32
	FrequencyFactors *Tensor
	RotaryDimensions uint32
	FrequencyBase    float32
	FrequencyScale   float32
	YaRN             bool
	OriginalContext  uint32
	ExtFactor        float32
	AttentionFactor  float32
	BetaFast         float32
	BetaSlow         float32
}

// RoPEWithOptions: typed single-axis rotary construction.
func (b *Builder) RoPEWithOptions(input *Tensor, options RoPEOptions) *Tensor {
	operation, name := OpRoPENormal, "rope_normal"
	switch options.Layout {
	case RoPELayoutNormal:
	case RoPELayoutNeoX:
		operation, name = OpRoPENeoX, "rope_neox"
	default:
		b.setError(errors.New("RoPE layout is invalid"))
		return nil
	}
	return b.buildRoPE(input, ropeOptions{
		operation: operation, name: name, positions: options.Positions,
		frequencyFactors: options.FrequencyFactors,
		rotaryDimensions: options.RotaryDimensions,
		frequencyBase:    options.FrequencyBase, frequencyScale: options.FrequencyScale,
		yarn: options.YaRN, originalContext: options.OriginalContext,
		extFactor: options.ExtFactor, attentionFactor: options.AttentionFactor,
		betaFast: options.BetaFast, betaSlow: options.BetaSlow,
	})
}

// RoPENeoX: split-half rotary layout; input [head width, heads, tokens, batch?].
func (b *Builder) RoPENeoX(input *Tensor, positions []uint32, rotaryDimensions uint32, frequencyBase float32) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENeoX, name: "rope_neox", positions: positions,
		rotaryDimensions: rotaryDimensions, frequencyBase: frequencyBase, frequencyScale: 1,
	})
}

func (b *Builder) RoPENeoXScaled(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENeoX, name: "rope_neox", positions: positions,
		rotaryDimensions: rotaryDimensions, frequencyBase: frequencyBase, frequencyScale: frequencyScale,
	})
}

// RoPENeoXYaRN: applies YaRN interpolation/extrapolation and magnitude scaling
// to split-half rotary layout
func (b *Builder) RoPENeoXYaRN(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	originalContext uint32,
	frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow float32,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENeoX, name: "rope_neox", positions: positions,
		rotaryDimensions: rotaryDimensions, originalContext: originalContext,
		frequencyBase: frequencyBase, frequencyScale: frequencyScale, yarn: true,
		extFactor: extFactor, attentionFactor: attentionFactor, betaFast: betaFast, betaSlow: betaSlow,
	})
}

func (b *Builder) RoPENeoXScaledWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENeoX, name: "rope_neox", positions: positions,
		frequencyFactors: frequencyFactors, rotaryDimensions: rotaryDimensions,
		frequencyBase: frequencyBase, frequencyScale: frequencyScale,
	})
}

// RoPENeoXWithFactors: applies one frequency divisor per rotary pair
func (b *Builder) RoPENeoXWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENeoX, name: "rope_neox", positions: positions,
		frequencyFactors: frequencyFactors, rotaryDimensions: rotaryDimensions,
		frequencyBase: frequencyBase, frequencyScale: 1,
	})
}

// RoPENormal: applies rotary embeddings to consecutive channel pairs, as used
// by Llama architecture family
func (b *Builder) RoPENormal(input *Tensor, positions []uint32, rotaryDimensions uint32, frequencyBase float32) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENormal, name: "rope_normal", positions: positions,
		rotaryDimensions: rotaryDimensions, frequencyBase: frequencyBase, frequencyScale: 1,
	})
}

func (b *Builder) RoPENormalScaled(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENormal, name: "rope_normal", positions: positions,
		rotaryDimensions: rotaryDimensions, frequencyBase: frequencyBase, frequencyScale: frequencyScale,
	})
}

// RoPENormalYaRN: applies YaRN interpolation/extrapolation and magnitude
// scaling to consecutive rotary pairs
func (b *Builder) RoPENormalYaRN(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	originalContext uint32,
	frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow float32,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENormal, name: "rope_normal", positions: positions,
		rotaryDimensions: rotaryDimensions, originalContext: originalContext,
		frequencyBase: frequencyBase, frequencyScale: frequencyScale, yarn: true,
		extFactor: extFactor, attentionFactor: attentionFactor, betaFast: betaFast, betaSlow: betaSlow,
	})
}

// RoPENormalYaRNWithFactors: YaRN plus pair divisors.
func (b *Builder) RoPENormalYaRNWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	originalContext uint32,
	frequencyBase, frequencyScale, extFactor, attentionFactor, betaFast, betaSlow float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENormal, name: "rope_normal", positions: positions,
		frequencyFactors: frequencyFactors, rotaryDimensions: rotaryDimensions,
		originalContext: originalContext, frequencyBase: frequencyBase,
		frequencyScale: frequencyScale, yarn: true, extFactor: extFactor,
		attentionFactor: attentionFactor, betaFast: betaFast, betaSlow: betaSlow,
	})
}

func (b *Builder) RoPENormalScaledWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyScale float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENormal, name: "rope_normal", positions: positions,
		frequencyFactors: frequencyFactors, rotaryDimensions: rotaryDimensions,
		frequencyBase: frequencyBase, frequencyScale: frequencyScale,
	})
}

// RoPENormalWithFactors: applies one frequency divisor per rotary pair
func (b *Builder) RoPENormalWithFactors(
	input *Tensor,
	positions []uint32,
	rotaryDimensions uint32,
	frequencyBase float32,
	frequencyFactors *Tensor,
) *Tensor {
	return b.buildRoPE(input, ropeOptions{
		operation: OpRoPENormal, name: "rope_normal", positions: positions,
		frequencyFactors: frequencyFactors, rotaryDimensions: rotaryDimensions,
		frequencyBase: frequencyBase, frequencyScale: 1,
	})
}

// RoPEMulti: adjacent-pair multi-axis rotation; axes T/H/W/extra.
func (b *Builder) RoPEMulti(
	input *Tensor,
	positions [4][]uint32,
	sections [4]int32,
	rotaryDimensions uint32,
	frequencyBase float32,
) *Tensor {
	return b.RoPEMultiScaled(input, positions, sections, rotaryDimensions, frequencyBase, 1)
}

func (b *Builder) RoPEMultiScaled(
	input *Tensor,
	positions [4][]uint32,
	sections [4]int32,
	rotaryDimensions uint32,
	frequencyBase, frequencyScale float32,
) *Tensor {
	if b.err != nil {
		return nil
	}
	if input == nil || input.Shape.Rank < 3 {
		b.setError(errors.New("rope_multi input must have rank 3 or 4"))
		return nil
	}
	if rotaryDimensions == 0 || rotaryDimensions%2 != 0 ||
		uint64(rotaryDimensions) > input.Shape.Dims[0] {
		b.setError(errors.New("rope_multi rotary dimensions are invalid"))
		return nil
	}
	var sectionPairs int64
	for axis := range sections {
		if sections[axis] < 0 {
			b.setError(errors.New("rope_multi section count is negative"))
			return nil
		}
		sectionPairs += int64(sections[axis])
		if len(positions[axis]) != int(input.Shape.Dims[2]) {
			b.setError(errors.New("rope_multi position count differs from token count"))
			return nil
		}
	}
	if sectionPairs == 0 {
		b.setError(errors.New("rope_multi sections are empty"))
		return nil
	}
	if frequencyBase <= 0 || frequencyScale <= 0 ||
		math.IsNaN(float64(frequencyScale)) || math.IsInf(float64(frequencyScale), 0) {
		b.setError(errors.New("rope_multi frequency parameters must be positive and finite"))
		return nil
	}
	attributes := RoPEMultiAttributes{
		Sections:         sections,
		RotaryDimensions: rotaryDimensions,
		FrequencyBase:    frequencyBase,
		FrequencyScale:   frequencyScale,
	}
	for axis := range positions {
		attributes.Positions[axis] = slices.Clone(positions[axis])
	}
	return b.add("", input.Type, input.Shape, OpRoPEMulti, []*Tensor{input}, attributes)
}

func (b *Builder) buildRoPE(input *Tensor, options ropeOptions) *Tensor {
	if b.err != nil {
		return nil
	}
	if options.yarn &&
		(options.originalContext == 0 || options.attentionFactor <= 0 ||
			options.betaFast <= 0 || options.betaSlow <= 0 || options.extFactor < 0) {
		b.setError(errors.New("YaRN RoPE parameters are invalid"))
		return nil
	}
	if input == nil {
		b.setError(fmt.Errorf("%s input is nil", options.name))
		return nil
	}
	if input.Shape.Rank < 3 {
		b.setError(fmt.Errorf("%s input must have rank 3 or 4", options.name))
		return nil
	}
	if options.rotaryDimensions == 0 || options.rotaryDimensions%2 != 0 ||
		uint64(options.rotaryDimensions) > input.Shape.Dims[0] {
		b.setError(fmt.Errorf(
			"%s rotary dimensions %d are invalid for head width %d",
			options.name, options.rotaryDimensions, input.Shape.Dims[0],
		))
		return nil
	}
	if len(options.positions) != int(input.Shape.Dims[2]) {
		b.setError(fmt.Errorf(
			"%s has %d positions, need %d", options.name, len(options.positions), input.Shape.Dims[2],
		))
		return nil
	}
	if options.frequencyBase <= 0 {
		b.setError(fmt.Errorf("%s frequency base must be positive", options.name))
		return nil
	}
	if options.frequencyScale <= 0 {
		b.setError(fmt.Errorf("%s frequency scale must be positive", options.name))
		return nil
	}
	if options.frequencyFactors != nil {
		if options.frequencyFactors.Type != dtype.F32 ||
			options.frequencyFactors.Shape.Rank != 1 ||
			options.frequencyFactors.Shape.Dims[0] != uint64(options.rotaryDimensions/2) {
			b.setError(fmt.Errorf(
				"%s frequency factors must be F32 with shape [%d]",
				options.name,
				options.rotaryDimensions/2,
			))
			return nil
		}
	}
	attributes := RoPEAttributes{
		Positions:        slices.Clone(options.positions),
		RotaryDimensions: options.rotaryDimensions,
		FrequencyBase:    options.frequencyBase,
		FrequencyScale:   options.frequencyScale,
		AttentionFactor:  1,
	}
	if options.yarn {
		attributes.OriginalContext = options.originalContext
		attributes.ExtFactor = options.extFactor
		attributes.AttentionFactor = options.attentionFactor
		attributes.BetaFast = options.betaFast
		attributes.BetaSlow = options.betaSlow
	}
	inputs := []*Tensor{input}
	if options.frequencyFactors != nil {
		inputs = append(inputs, options.frequencyFactors)
	}
	return b.add("", input.Type, input.Shape, options.operation, inputs, attributes)
}
