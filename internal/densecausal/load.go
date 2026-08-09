// Package densecausal owns dense causal-LM training on the host (ladder
// rung 3: full-parameter training step for llama-architecture artifacts).
// Composed from hostmath primitives in the rung-1 posture: f32 storage, f64
// accumulation, recompute-based backward keyed by checkpoint tensor names.
//
// Representation choice (stated once): this HOST path consumes the HF
// safetensors layout directly — q/k head rows pair as split halves, so rope
// is ApplyRotaryHalf, matching the torch golden. The GGUF serving path
// permutes q/k for interleaved rope (ApplyRotaryInterleaved); that layout
// never enters here.
//
// Every dimension is DERIVED from tensor shapes; config.json contributes
// only head_dim (fallback hidden/num_attention_heads), rope_theta, and
// rms_norm_eps.
package densecausal

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/safetensors"
)

// Dims: model geometry, derived from tensor shapes and config.
// AttnBias: q/k/v projection biases present (qwen2); derived from shapes.
type Dims struct {
	Vocab        int
	Hidden       int
	Layers       int
	Heads        int
	KVHeads      int
	HeadDim      int
	Intermediate int
	RopeTheta    float64
	RMSEps       float64
	AttnBias     bool
}

// Model: loaded weights (f32), shapes, and derived dims. HeadName keys the
// lm-head weight: lm_head.weight when untied, the embedding when tied — the
// tied case accumulates head+scatter grads in the one shared slot.
type Model struct {
	Dims     Dims
	Weights  map[string][]float32
	Shapes   map[string][]int
	HeadName string
}

type artifactConfig struct {
	ModelType         string  `json:"model_type"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	HeadDim           int     `json:"head_dim"`
	RopeTheta         float64 `json:"rope_theta"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	TieWordEmbeddings bool    `json:"tie_word_embeddings"`
}

// Load opens the safetensors artifact and materializes every tensor as f32.
func Load(directory string) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(directory, "config.json"))
	if err != nil {
		return nil, err
	}
	var config artifactConfig
	if err := json.Unmarshal(raw, &config); err != nil {
		return nil, fmt.Errorf("densecausal: parse config.json: %w", err)
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()

	shapes := make(map[string][]int, len(source.Tensors))
	weights := make(map[string][]float32, len(source.Tensors))
	for name, tensor := range source.Tensors {
		dims := make([]int, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			if dim == 0 || dim > 1<<31 {
				return nil, fmt.Errorf("densecausal: tensor %q dimension %d out of range", name, dim)
			}
			dims[i] = int(dim)
		}
		shapes[name] = dims
		reader, err := safetensors.F32Reader(tensor)
		if err != nil {
			return nil, fmt.Errorf("densecausal: tensor %q: %w", name, err)
		}
		elements := tensor.Elements()
		buf := make([]byte, elements*4)
		if _, err := io.ReadFull(reader, buf); err != nil {
			return nil, fmt.Errorf("densecausal: tensor %q payload: %w", name, err)
		}
		values := make([]float32, elements)
		for i := range values {
			bits := uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 | uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24
			values[i] = math.Float32frombits(bits)
		}
		weights[name] = values
	}
	m, err := NewModel(weights, shapes, config.NumAttentionHeads, config.HeadDim, config.RopeTheta, config.RMSNormEps)
	if err != nil {
		return nil, err
	}
	// model_type vs derived bias cross-check: qwen2 REQUIRES qkv biases,
	// llama forbids them; anything else is unverified.
	switch config.ModelType {
	case "llama":
		if m.Dims.AttnBias {
			return nil, fmt.Errorf("densecausal: model_type llama but attention biases present")
		}
	case "qwen2":
		if !m.Dims.AttnBias {
			return nil, fmt.Errorf("densecausal: model_type qwen2 but attention biases absent")
		}
	default:
		return nil, fmt.Errorf("densecausal: unsupported model_type %q (llama, qwen2)", config.ModelType)
	}
	// tie_word_embeddings cross-check: tied forbids lm_head.weight, untied requires it.
	if untied := m.HeadName == "lm_head.weight"; untied == config.TieWordEmbeddings {
		return nil, fmt.Errorf("densecausal: tie_word_embeddings=%v but lm_head.weight present=%v", config.TieWordEmbeddings, untied)
	}
	return m, nil
}

// NewModel derives dims from shapes and validates the geometry; weights map
// is adopted, not copied. headDim zero falls back to hidden/heads.
func NewModel(weights map[string][]float32, shapes map[string][]int, heads, headDim int, ropeTheta, rmsEps float64) (*Model, error) {
	var d Dims
	embed, err := shapeOf(shapes, "model.embed_tokens.weight", 2)
	if err != nil {
		return nil, err
	}
	d.Vocab, d.Hidden = embed[0], embed[1]
	headName := "model.embed_tokens.weight"
	if _, untied := shapes["lm_head.weight"]; untied {
		head, err := shapeOf(shapes, "lm_head.weight", 2)
		if err != nil {
			return nil, err
		}
		if head[0] != d.Vocab || head[1] != d.Hidden {
			return nil, fmt.Errorf("densecausal: lm_head.weight %v, want [%d %d]", head, d.Vocab, d.Hidden)
		}
		headName = "lm_head.weight"
	}
	if heads <= 0 {
		return nil, fmt.Errorf("densecausal: config num_attention_heads %d", heads)
	}
	if headDim == 0 {
		headDim = d.Hidden / heads
	}
	if headDim <= 0 {
		return nil, fmt.Errorf("densecausal: head dim %d", headDim)
	}
	d.HeadDim = headDim
	if ropeTheta <= 0 || rmsEps <= 0 {
		return nil, fmt.Errorf("densecausal: rope_theta %g rms_norm_eps %g must be positive", ropeTheta, rmsEps)
	}
	d.RopeTheta, d.RMSEps = ropeTheta, rmsEps
	for layer := 0; ; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		if _, ok := shapes[prefix+"self_attn.q_proj.weight"]; !ok {
			if layer == 0 {
				return nil, fmt.Errorf("densecausal: no model.layers.0")
			}
			d.Layers = layer
			break
		}
		q, err := shapeOf(shapes, prefix+"self_attn.q_proj.weight", 2)
		if err != nil {
			return nil, err
		}
		k, err := shapeOf(shapes, prefix+"self_attn.k_proj.weight", 2)
		if err != nil {
			return nil, err
		}
		gate, err := shapeOf(shapes, prefix+"mlp.gate_proj.weight", 2)
		if err != nil {
			return nil, err
		}
		if q[1] != d.Hidden || q[0]%d.HeadDim != 0 || k[0]%d.HeadDim != 0 {
			return nil, fmt.Errorf("densecausal: layer %d q %v / k %v incompatible with hidden %d head dim %d", layer, q, k, d.Hidden, d.HeadDim)
		}
		if layer == 0 {
			d.Heads = q[0] / d.HeadDim
			d.KVHeads = k[0] / d.HeadDim
			d.Intermediate = gate[0]
			if d.Heads != heads {
				return nil, fmt.Errorf("densecausal: derived heads %d != config %d", d.Heads, heads)
			}
			if d.KVHeads == 0 || d.Heads%d.KVHeads != 0 {
				return nil, fmt.Errorf("densecausal: heads %d not divisible by kv heads %d", d.Heads, d.KVHeads)
			}
		} else if q[0]/d.HeadDim != d.Heads || k[0]/d.HeadDim != d.KVHeads || gate[0] != d.Intermediate {
			return nil, fmt.Errorf("densecausal: layer %d geometry differs from layer 0", layer)
		}
		// q/k/v biases: all-or-none per layer, identical across layers.
		hasBias, err := layerAttnBias(shapes, prefix, q[0], k[0])
		if err != nil {
			return nil, err
		}
		if layer == 0 {
			d.AttnBias = hasBias
		} else if hasBias != d.AttnBias {
			return nil, fmt.Errorf("densecausal: layer %d bias presence differs from layer 0", layer)
		}
	}
	if _, err := shapeOf(shapes, "model.norm.weight", 1); err != nil {
		return nil, err
	}
	// Any bias outside the q/k/v attention triple is an unverified layout.
	for name := range shapes {
		if strings.HasSuffix(name, ".bias") && !attnBiasName(name) {
			return nil, fmt.Errorf("densecausal: unexpected bias tensor %q", name)
		}
	}
	return &Model{Dims: d, Weights: weights, Shapes: shapes, HeadName: headName}, nil
}

// layerAttnBias validates the per-layer q/k/v bias triple: absent entirely,
// or all present with lengths matching the projection out-dims.
func layerAttnBias(shapes map[string][]int, prefix string, qOut, kvOut int) (bool, error) {
	present := 0
	for _, want := range []struct {
		name string
		out  int
	}{
		{prefix + "self_attn.q_proj.bias", qOut},
		{prefix + "self_attn.k_proj.bias", kvOut},
		{prefix + "self_attn.v_proj.bias", kvOut},
	} {
		shape, ok := shapes[want.name]
		if !ok {
			continue
		}
		present++
		if len(shape) != 1 || shape[0] != want.out {
			return false, fmt.Errorf("densecausal: bias %q shape %v, want [%d]", want.name, shape, want.out)
		}
	}
	if present != 0 && present != 3 {
		return false, fmt.Errorf("densecausal: %sself_attn has %d of 3 q/k/v biases", prefix, present)
	}
	return present == 3, nil
}

// attnBiasName: model.layers.N.self_attn.{q,k,v}_proj.bias.
func attnBiasName(name string) bool {
	rest, ok := strings.CutPrefix(name, "model.layers.")
	if !ok {
		return false
	}
	if dot := strings.IndexByte(rest, '.'); dot >= 0 {
		rest = rest[dot+1:]
	}
	return rest == "self_attn.q_proj.bias" || rest == "self_attn.k_proj.bias" || rest == "self_attn.v_proj.bias"
}

func shapeOf(shapes map[string][]int, name string, rank int) ([]int, error) {
	shape, ok := shapes[name]
	if !ok {
		return nil, fmt.Errorf("densecausal: missing tensor %q", name)
	}
	if len(shape) != rank {
		return nil, fmt.Errorf("densecausal: tensor %q rank %d, want %d", name, len(shape), rank)
	}
	return shape, nil
}
