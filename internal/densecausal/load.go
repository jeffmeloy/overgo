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
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
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
	// MoE: router policy for routed layers; TopK zero means the artifact
	// declares no mixture and every layer is dense. Which layers are routed
	// derives from each layer's own tensors, never from a count.
	MoE MoERouterPolicy
	// AttentionWindow: the artifact's declared serving window; the host
	// trainer runs full causal attention, exact for sequences within the
	// window, and refuses longer training sequences.
	AttentionWindow int
	ContextLength   int
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
	NumAttentionHeads int     `json:"num_attention_heads"`
	HeadDim           int     `json:"head_dim"`
	RopeTheta         float64 `json:"rope_theta"`
	RMSNormEps        float64 `json:"rms_norm_eps"`
	TieWordEmbeddings bool    `json:"tie_word_embeddings"`

	// Routed-mixture declarations (the DeepSeek-V2 family). Absent fields
	// keep the declared family defaults: softmax scoring, no top-k
	// normalization, unit routed scaling.
	NumExpertsPerTok      int      `json:"num_experts_per_tok"`
	MoEIntermediateSize   int      `json:"moe_intermediate_size"`
	RoutedScalingFactor   *float64 `json:"routed_scaling_factor"`
	NormTopKProb          bool     `json:"norm_topk_prob"`
	ScoringFunc           string   `json:"scoring_func"`
	SlidingWindowSize     int      `json:"sliding_window_size"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`

	// Multimodal wrappers (Unlimited-OCR) nest the decoder declarations.
	LanguageConfig *artifactConfig `json:"language_config"`
}

// moePolicy compiles the declared router policy; zero TopK = dense model.
func (c artifactConfig) moePolicy() MoERouterPolicy {
	if c.NumExpertsPerTok <= 0 {
		return MoERouterPolicy{}
	}
	policy := MoERouterPolicy{
		TopK:              c.NumExpertsPerTok,
		Scoring:           MoEScoringSoftmax,
		NormalizeTopKProb: c.NormTopKProb,
		RoutedScaling:     1,
		ExpertInter:       c.MoEIntermediateSize,
	}
	if c.ScoringFunc != "" {
		policy.Scoring = MoEScoring(c.ScoringFunc)
	}
	if c.RoutedScalingFactor != nil {
		policy.RoutedScaling = float32(*c.RoutedScalingFactor)
	}
	return policy
}

func (c artifactConfig) resolvedDecoder() (artifactConfig, bool, error) {
	decoder, nested := c, c.LanguageConfig != nil
	if nested {
		decoder = *c.LanguageConfig
	}
	if !checked.PositiveFinite64(decoder.RopeTheta) || !checked.PositiveFinite64(decoder.RMSNormEps) {
		return artifactConfig{}, false, errors.New("densecausal: rope_theta and rms_norm_eps must be declared positive finite facts")
	}
	if decoder.MaxPositionEmbeddings < 0 {
		return artifactConfig{}, false, errors.New("densecausal: negative context length")
	}
	return decoder, nested, nil
}

// Load opens the decoder artifact and materializes its selected F32 inventory.
func Load(directory string) (*Model, error) {
	var config artifactConfig
	if err := jsonfile.Decode(filepath.Join(directory, "config.json"), &config); err != nil {
		return nil, fmt.Errorf("densecausal: parse config.json: %w", err)
	}
	decoder, nested, err := config.resolvedDecoder()
	if err != nil {
		return nil, err
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()

	selection := safetensors.F32Selection{RetainShapes: true}
	if nested {
		selection.Keep = decoderTensor
	}
	catalog, err := source.MaterializeF32(selection)
	if err != nil {
		return nil, fmt.Errorf("densecausal: materialize: %w", err)
	}
	m, err := NewMixtureModel(catalog.Values, catalog.Shapes, decoder.NumAttentionHeads, decoder.HeadDim,
		decoder.RopeTheta, decoder.RMSNormEps, decoder.moePolicy(), decoder.SlidingWindowSize)
	if err != nil {
		return nil, err
	}
	m.Dims.ContextLength = decoder.MaxPositionEmbeddings
	config.TieWordEmbeddings = decoder.TieWordEmbeddings || config.TieWordEmbeddings
	// tie_word_embeddings cross-check: tied forbids lm_head.weight, untied requires it.
	if untied := m.tensors.head.name != m.tensors.embedding.name; untied == config.TieWordEmbeddings {
		return nil, fmt.Errorf("densecausal: tie_word_embeddings=%v but lm_head.weight present=%v", config.TieWordEmbeddings, untied)
	}
	return m, nil
}

func decoderTensor(name string) bool {
	return name == "model.embed_tokens.weight" || name == "model.norm.weight" ||
		name == "lm_head.weight" || strings.HasPrefix(name, "model.layers.")
}

// NewModel derives dims from shapes and validates the geometry; weights map
// is adopted, not copied. headDim zero falls back to hidden/heads.
func NewModel(weights map[string][]float32, shapes map[string][]int, heads, headDim int, ropeTheta, rmsEps float64) (*Model, error) {
	return NewMixtureModel(weights, shapes, heads, headDim, ropeTheta, rmsEps, MoERouterPolicy{}, 0)
}

// NewMixtureModel is NewModel with a declared router policy for artifacts
// whose layers carry routed mixtures; per-layer FFN kind still derives from
// each layer's own tensors.
func NewMixtureModel(weights map[string][]float32, shapes map[string][]int, heads, headDim int, ropeTheta, rmsEps float64, policy MoERouterPolicy, window int) (*Model, error) {
	const (
		embeddingName  = "model.embed_tokens.weight"
		finalNormName  = "model.norm.weight"
		untiedHeadName = "lm_head.weight"
	)
	var d Dims
	embed, err := tensorcatalog.Shape(shapes, embeddingName, tensor.PairedExtent)
	if err != nil {
		return nil, err
	}
	d.Vocab, d.Hidden = embed[0], embed[1]
	headName := embeddingName
	if _, untied := shapes[untiedHeadName]; untied {
		head, err := tensorcatalog.Shape(shapes, untiedHeadName, tensor.PairedExtent)
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
	if !checked.PositiveFinite64(ropeTheta) || !checked.PositiveFinite64(rmsEps) {
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
		q, err := tensorcatalog.Shape(shapes, names.q, tensor.PairedExtent)
		if err != nil {
			return nil, err
		}
		k, err := tensorcatalog.Shape(shapes, names.k, tensor.PairedExtent)
		if err != nil {
			return nil, err
		}
		if q[1] != d.Hidden || q[0]%d.HeadDim != 0 || k[0]%d.HeadDim != 0 {
			return nil, fmt.Errorf("densecausal: layer %d q %v / k %v incompatible with hidden %d head dim %d", layer, q, k, d.Hidden, d.HeadDim)
		}
		if layer == 0 {
			d.Heads = q[0] / d.HeadDim
			d.KVHeads = k[0] / d.HeadDim
			if d.Heads != heads {
				return nil, fmt.Errorf("densecausal: derived heads %d != config %d", d.Heads, heads)
			}
			if d.KVHeads == 0 || d.Heads%d.KVHeads != 0 {
				return nil, fmt.Errorf("densecausal: heads %d not divisible by kv heads %d", d.Heads, d.KVHeads)
			}
		} else if q[0]/d.HeadDim != d.Heads || k[0]/d.HeadDim != d.KVHeads {
			return nil, fmt.Errorf("densecausal: layer %d attention geometry differs from layer 0", layer)
		}
		// Dense FFN geometry: the intermediate derives from the first dense
		// layer and every dense layer must agree; routed layers validate
		// against the declared policy in compileMoELayer instead.
		if _, dense := shapes[names.gate]; dense {
			gate, err := tensorcatalog.Shape(shapes, names.gate, tensor.PairedExtent)
			if err != nil {
				return nil, err
			}
			if d.Intermediate == 0 {
				d.Intermediate = gate[0]
			} else if gate[0] != d.Intermediate {
				return nil, fmt.Errorf("densecausal: layer %d dense intermediate %d differs from %d", layer, gate[0], d.Intermediate)
			}
		} else if policy.TopK <= 0 {
			return nil, fmt.Errorf("densecausal: layer %d has no dense mlp and no declared mixture policy", layer)
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
	if _, err := tensorcatalog.Shape(shapes, finalNormName, tensor.SingletonExtent); err != nil {
		return nil, err
	}
	// Any bias outside the q/k/v attention triple is an unverified layout.
	for name := range shapes {
		if strings.HasSuffix(name, ".bias") && !attnBiasName(name) {
			return nil, fmt.Errorf("densecausal: unexpected bias tensor %q", name)
		}
	}
	d.MoE, d.AttentionWindow = policy, window
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
		model.layers[index], err = compileLayer(weights, index, d.AttnBias, policy, d.Hidden)
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
