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
	p.output = g.joint(p.input, p.modulation, p.unit, index)
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("rotary joint: %w", err)
	}
	return p, nil
}

func (g *rotaryGraph) modulate(x, shift, scale, unit *tensor.Tensor) *tensor.Tensor {
	b := g.b
	return g.round(b.Add(g.round(b.Multiply(x, g.round(b.Add(scale, unit)))), shift))
}

func (g *rotaryGraph) joint(input, modulation, unit *tensor.Tensor, index int) *tensor.Tensor {
	b := g.b
	prefix := fmt.Sprintf("layers.%d.", index)
	mod := g.round(modulation)
	parts := make([]*tensor.Tensor, jointModulationFields)
	for i := range parts {
		parts[i] = b.FlatSlice(mod, uint64(i)*g.width, g.width)
	}
	attentionShift, attentionScale, attentionGate := parts[jointAttentionShift], parts[jointAttentionScale], parts[jointAttentionGate]
	feedShift, feedScale, feedGate := parts[jointFeedShift], parts[jointFeedScale], parts[jointFeedGate]
	hidden := g.round(input)
	y := g.modulate(g.norm(prefix+"norm1", hidden, g.width), attentionShift, attentionScale, unit)
	q := g.norm(prefix+"q_norm", b.Reshape(g.linear(prefix+"proj.q", y, g.width, g.width, false), g.headWidth, g.heads, g.sequence), g.headWidth)
	k := g.norm(prefix+"k_norm", b.Reshape(g.linear(prefix+"proj.k", y, g.width, g.width, false), g.headWidth, g.heads, g.sequence), g.headWidth)
	v := b.Reshape(g.linear(prefix+"proj.v", y, g.width, g.width, false), g.headWidth, g.heads, g.sequence)
	hidden = g.round(b.Add(hidden, g.round(b.Multiply(g.linear(prefix+"proj.o", g.attention(q, k, v), g.width, g.width, false), attentionGate))))
	y = g.modulate(g.norm(prefix+"norm2", hidden, g.width), feedShift, feedScale, unit)
	fused := g.linear(prefix+"proj.ffn_up_gate", y, g.width, tensor.PairedExtent*g.intermediate, false)
	up, gate := g.part(fused, tensor.FirstOffset, g.intermediate), g.part(fused, g.intermediate, g.intermediate)
	return g.round(b.Add(hidden, g.round(b.Multiply(g.linear(prefix+"proj.ffn_down", g.situ(up, gate), g.intermediate, g.width, false), feedGate))))
}
