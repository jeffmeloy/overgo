package latentimage

import (
	"encoding/json"
	"fmt"

	"overgo/internal/comfyquant"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// jointProjections are a joint block's projections (rotaryGraph.projection), each under its layer's prefix.
var jointProjections = []string{"proj.q", "proj.k", "proj.v", "proj.o", "proj.ffn_up_gate", "proj.ffn_down"}

// convRotProjections are the weights of the layers' joint projections a source holds in int8 ConvRot over groups of 256
// columns (comfyquant.Find: under its own name, or a q, k or v projection as its third of a fused qkv), each read from
// the header and its comfy_quant record alone, so each projection is held as its file holds it (int8-per-weight): one
// flag read from the first q projection had held them all in int8, and a projection the file holds as plain BF16 was
// refused as sampling bound it (SQA-423).
func convRotProjections(source *safetensors.Source, layers int) (map[string]bool, error) {
	held := map[string]bool{}
	for layer := range layers {
		for _, projection := range jointProjections {
			name := fmt.Sprintf("layers.%d.%s.weight", layer, projection)
			stored, quantized, err := comfyquant.Find(source, name)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
			if quantized && stored.Group() == comfyquant.Group {
				held[name] = true
			}
		}
	}
	return held, nil
}

// int8AttentionMethod is the attention a ComfyUI file declares for a module (its <module>.comfy_attention.config, read by
// ComfyAttention in comfy/ldm/modules/attention.py) that ComfyUI runs as comfy_kitchen's int8_attention.
const int8AttentionMethod = "comfy_kitchen_int8"

// int8AttentionLayers is the joint layers a transformer's file declares comfy_kitchen's int8 attention for: a layer
// whose layers.N.comfy_attention.config, one JSON record or a list of them, names it first, as ComfyAttention takes the
// first method it can run. A layer that declares none keeps the BF16 attention, as ComfyUI keeps its default for it; one
// that declares another method fails, since overgo runs no other.
func int8AttentionLayers(source *safetensors.Source, layers int) (map[int]bool, error) {
	declared := map[int]bool{}
	for layer := range layers {
		name := fmt.Sprintf("layers.%d.comfy_attention.config", layer)
		stored, held := source.Tensors[name]
		if !held {
			continue
		}
		record := make([]byte, stored.Size())
		if _, err := stored.ReadAt(record, 0); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		var configs []struct {
			Attention string `json:"attention"`
		}
		if err := json.Unmarshal(record, &configs); err != nil {
			configs = configs[:0]
			var single struct {
				Attention string `json:"attention"`
			}
			if err := json.Unmarshal(record, &single); err != nil {
				return nil, fmt.Errorf("%s holds %q, no attention record: %w", name, record, err)
			}
			configs = append(configs, single)
		}
		if len(configs) == 0 || configs[0].Attention != int8AttentionMethod {
			return nil, fmt.Errorf("%s declares %q; overgo runs %s alone", name, record, int8AttentionMethod)
		}
		declared[layer] = true
	}
	return declared, nil
}

type rotaryJointProgram struct {
	input, rope, modulation, unit, output *tensor.Tensor
	// attention is the block's attention output, rounded to BF16 as the output projection reads it, which a test reads
	// beside the block's output.
	attention    *tensor.Tensor
	weightInputs tensor.WeightBindings
	// adapterInputs and strengths are each adapter's tensors and strength input, in the request's order (rotaryOptions).
	adapterInputs []tensor.WeightBindings
	strengths     []*tensor.Tensor
	// merged are the projections the block reads with their adapters' pairs merged into their int8 weights, bound as
	// the merge makes them (bindMerged), as the denoiser's are; none where its adapters patch no int8 projection.
	merged []mergedProjection
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

// compileRotaryJointProgram compiles one joint block of MOSS's denoiser over tokens rows, options applying an adapter or 8-bit
// attention as the denoiser's do (rotaryOptions).
func compileRotaryJointProgram(t TransformerSpec, epsilon float32, tokens, index int, storage dtype.Type, options rotaryOptions) (*rotaryJointProgram, error) {
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
	p.adapterInputs = make([]tensor.WeightBindings, len(options.adapters))
	p.strengths = g.use(options, p.adapterInputs)
	b := g.b
	p.input = b.Input("joint_hidden", dtype.F32, tensor.MustShape(g.width, g.sequence))
	p.modulation = b.Input("joint_modulation", dtype.F32, tensor.MustShape(uint64(t.ModFieldsOr6())*g.width))
	p.unit = b.Input("unit", dtype.F32, tensor.MustShape(g.width))
	p.rope = g.rotaryInput("joint_rotary")
	p.output = g.joint(p.input, p.modulation, p.unit, index)
	p.attention = g.attended
	if err := b.Err(); err != nil {
		return nil, fmt.Errorf("rotary joint: %w", err)
	}
	p.merged = *g.merged
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
	// A block's input is the last block's rounded sum, which a rounding would not change (moss-fusion/copies).
	hidden := input
	if input.Op != tensor.OpBF16Round {
		hidden = g.round(input)
	}
	y := g.modulate(g.norm(prefix+"norm1", hidden, g.width), attentionShift, attentionScale, unit)
	q := g.norm(prefix+"q_norm", b.Reshape(g.projection(prefix+"proj.q", y, g.width, g.width), g.headWidth, g.heads, g.sequence), g.headWidth)
	k := g.norm(prefix+"k_norm", b.Reshape(g.projection(prefix+"proj.k", y, g.width, g.width), g.headWidth, g.heads, g.sequence), g.headWidth)
	v := b.Reshape(g.projection(prefix+"proj.v", y, g.width, g.width), g.headWidth, g.heads, g.sequence)
	if g.valueScale != nil {
		v = b.Multiply(v, g.valueScale)
	}
	g.attended = g.attention(q, k, v, g.int8Attention[index])
	hidden = g.round(b.Add(hidden, g.round(b.Multiply(g.projection(prefix+"proj.o", g.attended, g.width, g.width), attentionGate))))
	y = g.modulate(g.norm(prefix+"norm2", hidden, g.width), feedShift, feedScale, unit)
	fused := g.projection(prefix+"proj.ffn_up_gate", y, g.width, tensor.PairedExtent*g.intermediate)
	down := prefix + "proj.ffn_down"
	return g.round(b.Add(hidden, g.round(b.Multiply(g.projection(down, g.feedActivation(down, fused), g.intermediate, g.width), feedGate))))
}
