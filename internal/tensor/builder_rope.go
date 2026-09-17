package tensor

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"overgo/internal/tensor/dtype"
)

// UnitFrequencyScale: unscaled rotary positions.
const UnitFrequencyScale = float32(1)

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
	reverse                       bool
}

// RoPELayout: rotary channel pairing.
type RoPELayout uint8

const (
	RoPELayoutNormal RoPELayout = iota
	RoPELayoutNeoX
)

// RoPEOptions defines rotary coordinates and channel pairing.
type RoPEOptions struct {
	Layout              RoPELayout
	Positions           []uint32
	MultiPositions      *[MaxDimensions][]uint32
	Sections            [MaxDimensions]int32
	InterleavedSections bool
	FrequencyFactors    *Tensor
	RotaryDimensions    uint32
	FrequencyBase       float32
	FrequencyScale      float32
	YaRN                bool
	OriginalContext     uint32
	ExtFactor           float32
	AttentionFactor     float32
	BetaFast            float32
	BetaSlow            float32
	Reverse             bool
}

// RoPEWithOptions builds single-axis or multi-axis rotary operations.
func (b *Builder) RoPEWithOptions(input *Tensor, options RoPEOptions) *Tensor {
	if options.MultiPositions != nil {
		if options.Layout != RoPELayoutNormal || len(options.Positions) != 0 || options.FrequencyFactors != nil || options.YaRN || options.Reverse ||
			options.OriginalContext != 0 || options.ExtFactor != 0 || options.AttentionFactor != 0 || options.BetaFast != 0 || options.BetaSlow != 0 {
			b.setError(errors.New("multi-axis RoPE cannot combine single-axis controls"))
			return nil
		}
		return b.buildRoPEMulti(input, options)
	}
	if options.InterleavedSections || options.Sections != [MaxDimensions]int32{} {
		b.setError(errors.New("RoPE sections require multi-axis positions"))
		return nil
	}
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
		betaFast: options.BetaFast, betaSlow: options.BetaSlow, reverse: options.Reverse,
	})
}

func (b *Builder) buildRoPEMulti(input *Tensor, options RoPEOptions) *Tensor {
	positions, sections := *options.MultiPositions, options.Sections
	rotaryDimensions := options.RotaryDimensions
	frequencyBase, frequencyScale := options.FrequencyBase, options.FrequencyScale
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
		InterleavedSections: options.InterleavedSections,
		Sections:            sections,
		RotaryDimensions:    rotaryDimensions,
		FrequencyBase:       frequencyBase,
		FrequencyScale:      frequencyScale,
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
	if options.reverse {
		attributes.FrequencyScale = -attributes.FrequencyScale
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
