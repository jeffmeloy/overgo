package latentimage

import (
	"fmt"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

type rotaryJointProgram struct {
	input, rope, modulation, unit, output *tensor.Tensor
	weightInputs                          tensor.WeightBindings
}

// Joint modulation positions follow the pinned _Block.forward unbind order.
// Reconsider this order only with a different source contract.
const (
	jointAttentionShift = iota
	jointAttentionScale
	jointAttentionGate
	jointFeedShift
	jointFeedScale
	jointFeedGate
	jointModulationFields
)

func compileRotaryJointProgram(t TransformerSpec, epsilon float32, tokens, index int, storage dtype.Type) (*rotaryJointProgram, error) {
	if index < 0 || index >= t.Layers {
		return nil, fmt.Errorf("rotary joint block %d outside %d layers", index, t.Layers)
	}
	if t.ModFieldsOr6() != jointModulationFields {
		return nil, fmt.Errorf("rotary joint modulation fields %d must be %d", t.ModFieldsOr6(), jointModulationFields)
	}
	p := &rotaryJointProgram{}
	g, err := newRotaryGraph(t, epsilon, tokens, storage, &p.weightInputs)
	if err != nil {
		return nil, err
	}
	b := g.b
	p.input = b.Input("joint_hidden", dtype.F32, tensor.MustShape(g.width, g.sequence))
	p.modulation = b.Input("joint_modulation", dtype.F32, tensor.MustShape(uint64(t.ModFieldsOr6())*g.width))
	p.unit = b.Input("unit", dtype.F32, tensor.MustShape(g.width))
	p.rope = g.rotaryInput("joint_rotary")
	prefix := fmt.Sprintf("layers.%d.", index)
	mod := g.round(p.modulation)
	parts := make([]*tensor.Tensor, t.ModFieldsOr6())
	for i := range parts {
		parts[i] = b.FlatSlice(mod, uint64(i)*g.width, g.width)
	}
	attentionShift, attentionScale, attentionGate := parts[jointAttentionShift], parts[jointAttentionScale], parts[jointAttentionGate]
	feedShift, feedScale, feedGate := parts[jointFeedShift], parts[jointFeedScale], parts[jointFeedGate]
	modulate := func(x, shift, scale *tensor.Tensor) *tensor.Tensor {
		return g.round(b.Add(g.round(b.Multiply(x, g.round(b.Add(scale, p.unit)))), shift))
	}
	hidden := g.round(p.input)
	y := modulate(g.norm(prefix+"norm1", hidden, g.width), attentionShift, attentionScale)
	if err := b.Err(); err != nil {
		return nil, err
	}
	q := g.norm(prefix+"q_norm", b.Reshape(g.linear(prefix+"proj.q", y, g.width, g.width, false), g.headWidth, g.heads, g.sequence), g.headWidth)
	k := g.norm(prefix+"k_norm", b.Reshape(g.linear(prefix+"proj.k", y, g.width, g.width, false), g.headWidth, g.heads, g.sequence), g.headWidth)
	v := b.Reshape(g.linear(prefix+"proj.v", y, g.width, g.width, false), g.headWidth, g.heads, g.sequence)
	hidden = g.round(b.Add(hidden, g.round(b.Multiply(g.linear(prefix+"proj.o", g.attention(q, k, v), g.width, g.width, false), attentionGate))))
	y = modulate(g.norm(prefix+"norm2", hidden, g.width), feedShift, feedScale)
	fused := g.linear(prefix+"proj.ffn_up_gate", y, g.width, tensor.PairedExtent*g.intermediate, false)
	up, gate := g.part(fused, tensor.FirstOffset, g.intermediate), g.part(fused, g.intermediate, g.intermediate)
	p.output = g.round(b.Add(hidden, g.round(b.Multiply(g.linear(prefix+"proj.ffn_down", g.situ(up, gate), g.intermediate, g.width, false), feedGate))))
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("rotary joint: %w", err)
	}
	return p, nil
}
