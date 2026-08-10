package routedlm

import (
	"fmt"

	"overgo/internal/hostmath"
	"overgo/internal/safetensors"
)

// Position axes a rope section can rotate on (RowPosition fields).
const (
	AxisTime = iota
	AxisHeight
	AxisWidth
	axisCount
)

// RowPosition: multi-axis position of one row. Text rows advance Time
// causally; an image block shares one Time and rasterizes (H, W) over its
// token grid. Branch selects the weight set (0 = text, 1 = vision).
type RowPosition struct {
	Branch, Time, H, W int
}

func (p RowPosition) axis(axis int) int {
	switch axis {
	case AxisHeight:
		return p.H
	case AxisWidth:
		return p.W
	default:
		return p.Time
	}
}

// RopeSection: one contiguous half-split rotary span of a head row.
type RopeSection struct {
	Width int     // even span width, sections concatenate to head_dim
	Theta float64 // rotation base (dynamic-NTK adjusted at section width)
	Axis  int     // RowPosition axis driving the rotation angle
}

// RopePlan: per-section rotate-half spans covering head_dim. The single
// full-width section at the config theta is the degenerate (rxbrain) case
// and reproduces the plain scalar-position rotation bit-identically.
// Ported from adaptive causalPrefixRuntimeShape + runtimeAxisRoPEInto:
// angle = position[axis] * theta^(-2i/section_width).
type RopePlan struct {
	Sections []RopeSection
	invFreq  [][]float64
}

func newRopePlan(headDim int, sections []RopeSection) (RopePlan, error) {
	total := 0
	plan := RopePlan{Sections: sections, invFreq: make([][]float64, len(sections))}
	for i, section := range sections {
		if section.Width <= 0 || section.Width%2 != 0 || section.Theta <= 0 || section.Axis < 0 || section.Axis >= axisCount {
			return RopePlan{}, fmt.Errorf("routed lm rope plan: invalid section %+v", section)
		}
		total += section.Width
		plan.invFreq[i] = hostmath.RopeInvFreq(section.Theta, section.Width)
	}
	if total != headDim {
		return RopePlan{}, fmt.Errorf("routed lm rope plan: sections span %d, head_dim %d", total, headDim)
	}
	return plan, nil
}

// applyRotary: per-section rotate-half at the row's axis positions with the
// reference bf16 stepping.
func (p RopePlan) applyRotary(row []float32, pos RowPosition) {
	offset := 0
	for i, section := range p.Sections {
		applyRotaryHalfBF16(row[offset:offset+section.Width], p.invFreq[i], pos.axis(section.Axis))
		offset += section.Width
	}
}

// ropePlanFromConfig: the degenerate single full-width section at the config
// theta — the exact arithmetic of the pre-plan scalar-position path.
func ropePlanFromConfig(cfg Config) (RopePlan, error) {
	base, err := RopeInvFreqBase(cfg)
	if err != nil {
		return RopePlan{}, err
	}
	return newRopePlan(cfg.HeadDim, []RopeSection{{Width: cfg.HeadDim, Theta: base, Axis: AxisTime}})
}

// CompileRopePlan: rope sections derived from the binding — widths from the
// QK-norm tensor shapes (never hardcoded), per-section theta from the config
// keys the binding names. One norm section maps to a full-width Time
// rotation; two map to Time + an equal Height/Width split of the hw section
// (adaptive RoPEAxes = [text, hw/2, hw/2] at [theta, theta_hw, theta_hw]).
func CompileRopePlan(src *safetensors.Source, cfg Config, b BranchBinding) (RopePlan, error) {
	if err := b.validate(); err != nil {
		return RopePlan{}, err
	}
	widths := make([]int, len(b.QNormSections))
	for i := range b.QNormSections {
		qWidth, err := normSectionWidth(src, b.LayerTensorName(0, 0, b.QNormSections[i]))
		if err != nil {
			return RopePlan{}, err
		}
		kWidth, err := normSectionWidth(src, b.LayerTensorName(0, 0, b.KNormSections[i]))
		if err != nil {
			return RopePlan{}, err
		}
		if qWidth != kWidth {
			return RopePlan{}, fmt.Errorf("routed lm rope plan: QK norm section %d widths %d != %d", i, qWidth, kWidth)
		}
		widths[i] = qWidth
	}
	thetas := make([]float64, len(b.RopeThetaKeys))
	for i, key := range b.RopeThetaKeys {
		theta, ok := cfg.RopeBase(key)
		if !ok {
			return RopePlan{}, fmt.Errorf("routed lm rope plan: config lacks %s", key)
		}
		thetas[i] = theta
	}
	var sections []RopeSection
	switch len(widths) {
	case 1:
		base, err := ropeSectionBase(thetas[0], widths[0], cfg.RopeScaling)
		if err != nil {
			return RopePlan{}, err
		}
		sections = []RopeSection{{Width: widths[0], Theta: base, Axis: AxisTime}}
	case 2:
		if widths[1]%2 != 0 {
			return RopePlan{}, fmt.Errorf("routed lm rope plan: hw section width %d not splittable", widths[1])
		}
		base, err := ropeSectionBase(thetas[0], widths[0], cfg.RopeScaling)
		if err != nil {
			return RopePlan{}, err
		}
		sections = []RopeSection{
			{Width: widths[0], Theta: base, Axis: AxisTime},
			{Width: widths[1] / 2, Theta: thetas[1], Axis: AxisHeight},
			{Width: widths[1] / 2, Theta: thetas[1], Axis: AxisWidth},
		}
	default:
		return RopePlan{}, fmt.Errorf("routed lm rope plan: unsupported norm section count %d", len(widths))
	}
	return newRopePlan(cfg.HeadDim, sections)
}

func normSectionWidth(src *safetensors.Source, name string) (int, error) {
	t, ok := src.Tensors[name]
	if !ok {
		return 0, fmt.Errorf("routed lm rope plan: missing norm tensor %s", name)
	}
	if len(t.Shape) != 1 || t.Shape[0] <= 0 {
		return 0, fmt.Errorf("routed lm rope plan: norm tensor %s shape %v, want 1-D", name, t.Shape)
	}
	return int(t.Shape[0]), nil
}
