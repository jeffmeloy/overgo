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
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
	"overgo/internal/tensorcatalog"
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

// Model owns weights, shapes, geometry, and compiled bindings.
type Model struct {
	Dims    Dims
	Weights map[string][]float32
	Shapes  map[string][]int
	tensors modelTensorBindings
	layers  []layer
}

type tensorBinding struct {
	name   string
	values []float32
}

type modelTensorBindings struct {
	embedding tensorBinding
	finalNorm tensorBinding
	head      tensorBinding
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
	var config artifactConfig
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &config); err != nil {
		return nil, fmt.Errorf("densecausal: parse config.json: %w", err)
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()

	shapes, err := source.IntShapes()
	if err != nil {
		return nil, fmt.Errorf("densecausal: inventory: %w", err)
	}
	weights, err := source.ReadAllF32()
	if err != nil {
		return nil, fmt.Errorf("densecausal: materialize: %w", err)
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
	if untied := m.tensors.head.name != m.tensors.embedding.name; untied == config.TieWordEmbeddings {
		return nil, fmt.Errorf("densecausal: tie_word_embeddings=%v but lm_head.weight present=%v", config.TieWordEmbeddings, untied)
	}
	return m, nil
}

// NewModel derives dims from shapes and validates the geometry; weights map
// is adopted, not copied. headDim zero falls back to hidden/heads.
func NewModel(weights map[string][]float32, shapes map[string][]int, heads, headDim int, ropeTheta, rmsEps float64) (*Model, error) {
	const (
		embeddingName  = "model.embed_tokens.weight"
		finalNormName  = "model.norm.weight"
		untiedHeadName = "lm_head.weight"
	)
	var d Dims
	embed, err := tensorcatalog.Shape(shapes, embeddingName, 2)
	if err != nil {
		return nil, err
	}
	d.Vocab, d.Hidden = embed[0], embed[1]
	headName := embeddingName
	if _, untied := shapes[untiedHeadName]; untied {
		head, err := tensorcatalog.Shape(shapes, untiedHeadName, 2)
		if err != nil {
			return nil, err
		}
		if head[0] != d.Vocab || head[1] != d.Hidden {
			return nil, fmt.Errorf("densecausal: lm_head.weight %v, want [%d %d]", head, d.Vocab, d.Hidden)
		}
		headName = untiedHeadName
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
		names := denseLayerTensorNames(layer)
		if _, ok := shapes[names.q]; !ok {
			if layer == 0 {
				return nil, fmt.Errorf("densecausal: no model.layers.0")
			}
			d.Layers = layer
			break
		}
		q, err := tensorcatalog.Shape(shapes, names.q, 2)
		if err != nil {
			return nil, err
		}
		k, err := tensorcatalog.Shape(shapes, names.k, 2)
		if err != nil {
			return nil, err
		}
		gate, err := tensorcatalog.Shape(shapes, names.gate, 2)
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
		hasBias, err := layerAttnBias(shapes, names, q[0], k[0])
		if err != nil {
			return nil, err
		}
		if layer == 0 {
			d.AttnBias = hasBias
		} else if hasBias != d.AttnBias {
			return nil, fmt.Errorf("densecausal: layer %d bias presence differs from layer 0", layer)
		}
	}
	if _, err := tensorcatalog.Shape(shapes, finalNormName, 1); err != nil {
		return nil, err
	}
	// Any bias outside the q/k/v attention triple is an unverified layout.
	for name := range shapes {
		if strings.HasSuffix(name, ".bias") && !attnBiasName(name) {
			return nil, fmt.Errorf("densecausal: unexpected bias tensor %q", name)
		}
	}
	model := &Model{
		Dims: d, Weights: weights, Shapes: shapes,
		tensors: modelTensorBindings{
			embedding: tensorBinding{name: embeddingName, values: weights[embeddingName]},
			finalNorm: tensorBinding{name: finalNormName, values: weights[finalNormName]},
			head:      tensorBinding{name: headName, values: weights[headName]},
		},
		layers: make([]layer, d.Layers),
	}
	for index := range model.layers {
		model.layers[index], err = compileLayer(weights, index, d.AttnBias)
		if err != nil {
			return nil, err
		}
	}
	return model, nil
}

// layerAttnBias validates the per-layer q/k/v bias triple: absent entirely,
// or all present with lengths matching the projection out-dims.
func layerAttnBias(shapes map[string][]int, names layerTensorNames, qOut, kvOut int) (bool, error) {
	present := 0
	for _, want := range []struct {
		name string
		out  int
	}{
		{names.qb, qOut},
		{names.kb, kvOut},
		{names.vb, kvOut},
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
		return false, fmt.Errorf("densecausal: layer containing %q has %d of 3 q/k/v biases", names.q, present)
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
