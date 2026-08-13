package routedlm

// Multi-axis per-section rotary on the generic executor, composed from stock
// ops (GroupSlice + RoPENeoX + Concat) — no family kernel. Each RopePlan
// section is a contiguous head-dim span rotated NeoX rotate-half at its OWN
// theta over its OWN axis positions; the device RoPEMulti op cannot express
// this because it carries a single FrequencyBase for every section. Splicing
// per-section RoPENeoX reproduces routedlm.RopePlan.applyRotary exactly (bf16
// stepping aside): NeoX pairs (i, i+width/2) with angle pos*theta^(-2i/width),
// which is applyRotaryHalfBF16's arithmetic with rotary=width, base=theta.

import (
	"fmt"
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor"
)

// axisPositions: the per-token position vector this section rotates on.
func axisPositions(positions []RowPosition, axis int) []uint32 {
	out := make([]uint32, len(positions))
	for i, p := range positions {
		v := p.axis(axis)
		if v < 0 {
			v = 0
		}
		out[i] = uint32(v)
	}
	return out
}

// DeviceApply: emits the plan's multi-axis rotation of q ([head_dim, heads,
// tokens] F32) as a device graph node, one RoPENeoX per section spliced back
// together on the channel axis. Sections must tile head_dim (newRopePlan
// enforces this), so the concatenation is exactly head_dim wide.
func (p RopePlan) DeviceApply(b *tensor.Builder, q *tensor.Tensor, positions []RowPosition) (*tensor.Tensor, error) {
	if q == nil || q.Shape.Rank != 3 {
		return nil, fmt.Errorf("routed lm rope device: q must be rank-3 [head_dim, heads, tokens]")
	}
	headDim := q.Shape.Dims[0]
	heads := q.Shape.Dims[1]
	tokens := q.Shape.Dims[2]
	if int(tokens) != len(positions) {
		return nil, fmt.Errorf("routed lm rope device: tokens=%d positions=%d", tokens, len(positions))
	}
	var out *tensor.Tensor
	offset := uint64(0)
	for _, section := range p.Sections {
		w := uint64(section.Width)
		if offset+w > headDim {
			return nil, fmt.Errorf("routed lm rope device: section span %d exceeds head_dim %d", offset+w, headDim)
		}
		// contiguous channel window [offset, offset+w) over every (head, token).
		sliced := b.GroupSlice(q, offset, w, 1, w) // [w, 1, heads, tokens]
		sec := b.Reshape(sliced, w, heads, tokens) // [w, heads, tokens]
		roped := b.RoPENeoX(sec, axisPositions(positions, section.Axis), uint32(w), float32(section.Theta))
		if out == nil {
			out = roped
		} else {
			out = b.Concat(out, roped, 0)
		}
		offset += w
	}
	if b.Err() != nil {
		return nil, fmt.Errorf("routed lm rope device graph: %w", b.Err())
	}
	return out, nil
}

// DeviceNormalizeApply reproduces routed runtime QK discipline: projection
// round, per-axis-section weighted RMS norm, then section RoPE with BF16 steps.
func (p RopePlan) DeviceNormalizeApply(
	b *tensor.Builder,
	q, weight *tensor.Tensor,
	positions []RowPosition,
	epsilon float32,
) (*tensor.Tensor, error) {
	if q == nil || weight == nil || q.Shape.Rank != 3 || weight.Shape.Rank != 1 ||
		q.Shape.Dims[0] != weight.Shape.Dims[0] || int(q.Shape.Dims[2]) != len(positions) {
		return nil, fmt.Errorf("routed lm rope device: invalid normalized rotation inputs")
	}
	q = b.BF16Round(q)
	heads, tokens := q.Shape.Dims[1], q.Shape.Dims[2]
	var out *tensor.Tensor
	offset := uint64(0)
	for _, section := range p.Sections {
		width := uint64(section.Width)
		window := b.GroupSlice(q, offset, width, 1, width)
		window = b.Reshape(window, width, heads, tokens)
		scale := b.FlatSlice(weight, offset, width)
		window = b.BF16Round(b.WeightedRMSNorm(window, scale, epsilon))
		window = b.RoPENeoX(
			window, axisPositions(positions, section.Axis), uint32(width), float32(section.Theta),
		)
		window = b.BF16Round(window)
		if out == nil {
			out = window
		} else {
			out = b.Concat(out, window, 0)
		}
		offset += width
	}
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("routed lm normalized rope device graph: %w", err)
	}
	return out, nil
}

// HostApplyBF16Rows: the reference SenseNova rotation — applyRotary (bf16
// stepping) over each head row of rows laid out [tokens, heads, head_dim] flat.
func (p RopePlan) HostApplyBF16Rows(rows []float32, positions []RowPosition, heads, headDim int) error {
	if heads <= 0 || headDim <= 0 || len(rows) != len(positions)*heads*headDim {
		return fmt.Errorf("routed lm rope host bf16: rows=%d want %d", len(rows), len(positions)*heads*headDim)
	}
	for t, pos := range positions {
		for h := 0; h < heads; h++ {
			base := (t*heads + h) * headDim
			p.applyRotary(rows[base:base+headDim], pos)
		}
	}
	return nil
}

// HostApplyF32Rows: the same rotation in pure f32 (no bf16 rounding) — the
// arithmetic-correctness oracle for the device kernel, which also runs f32.
func (p RopePlan) HostApplyF32Rows(rows []float32, positions []RowPosition, heads, headDim int) error {
	if heads <= 0 || headDim <= 0 || len(rows) != len(positions)*heads*headDim {
		return fmt.Errorf("routed lm rope host f32: rows=%d want %d", len(rows), len(positions)*heads*headDim)
	}
	invFreq := make([][]float64, len(p.Sections))
	for i, section := range p.Sections {
		invFreq[i] = hostmath.RopeInvFreq(section.Theta, section.Width)
	}
	for t, pos := range positions {
		for h := 0; h < heads; h++ {
			base := (t*heads + h) * headDim
			offset := 0
			for si, section := range p.Sections {
				span := rows[base+offset : base+offset+section.Width]
				half := section.Width / 2
				axisPos := pos.axis(section.Axis)
				for i := 0; i < half; i++ {
					ang := float64(axisPos) * invFreq[si][i]
					c := float32(math.Cos(ang))
					s := float32(math.Sin(ang))
					a, bb := span[i], span[i+half]
					span[i] = a*c - bb*s
					span[i+half] = bb*c + a*s
				}
				offset += section.Width
			}
		}
	}
	return nil
}
