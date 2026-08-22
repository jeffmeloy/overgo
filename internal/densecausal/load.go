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
	ModelType         string  `json:"model_type"`
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
	// Multimodal wrappers nest the decoder declarations; the decoder tensor
	// subset (embedding, layers, final norm, head) is the trainable text
	// model — vision towers stay out of the pack and the bias-layout scan.
	decoder := config
	if config.LanguageConfig != nil {
		nested := *config.LanguageConfig
		if nested.ModelType == "" {
			nested.ModelType = config.ModelType
		}
		decoder = nested
		weights, shapes = decoderTensorSubset(weights, shapes)
	}
	if decoder.ModelType == "unlimited-ocr" {
		// Declared DeepSeek-V2 family defaults for config-absent facts.
		if decoder.RopeTheta == 0 {
			decoder.RopeTheta = 10000
		}
		if decoder.RMSNormEps == 0 {
			decoder.RMSNormEps = 1e-6
		}
	}
	m, err := NewMixtureModel(weights, shapes, decoder.NumAttentionHeads, decoder.HeadDim,
		decoder.RopeTheta, decoder.RMSNormEps, decoder.moePolicy(), decoder.SlidingWindowSize)
	if err != nil {
		return nil, err
	}
	if decoder.MaxPositionEmbeddings < 0 {
		return nil, fmt.Errorf("densecausal: negative context length")
	}
	m.Dims.ContextLength = decoder.MaxPositionEmbeddings
	// model_type vs derived cross-checks: qwen2 REQUIRES qkv biases, llama
	// forbids them, unlimited-ocr declares a routed mixture; anything else
	// is unverified.
	switch decoder.ModelType {
	case "llama":
		if m.Dims.AttnBias {
			return nil, fmt.Errorf("densecausal: model_type llama but attention biases present")
		}
	case "qwen2":
		if !m.Dims.AttnBias {
			return nil, fmt.Errorf("densecausal: model_type qwen2 but attention biases absent")
		}
	case "unlimited-ocr":
		if m.Dims.AttnBias {
			return nil, fmt.Errorf("densecausal: model_type unlimited-ocr but attention biases present")
		}
		if m.Dims.MoE.TopK <= 0 {
			return nil, fmt.Errorf("densecausal: model_type unlimited-ocr but no mixture declared")
		}
	default:
		return nil, fmt.Errorf("densecausal: unsupported model_type %q (llama, qwen2, unlimited-ocr)", decoder.ModelType)
	}
	config.TieWordEmbeddings = decoder.TieWordEmbeddings || config.TieWordEmbeddings
	// tie_word_embeddings cross-check: tied forbids lm_head.weight, untied requires it.
	if untied := m.tensors.head.name != m.tensors.embedding.name; untied == config.TieWordEmbeddings {
		return nil, fmt.Errorf("densecausal: tie_word_embeddings=%v but lm_head.weight present=%v", config.TieWordEmbeddings, untied)
	}
	return m, nil
}

// decoderTensorSubset keeps the causal-decoder tensors of a multimodal
// artifact: embedding, model.layers.*, final norm, and the head. Everything
// else (vision towers, projectors) stays out of the model and the pack.
func decoderTensorSubset(weights map[string][]float32, shapes map[string][]int) (map[string][]float32, map[string][]int) {
	keep := func(name string) bool {
		return name == "model.embed_tokens.weight" || name == "model.norm.weight" ||
			name == "lm_head.weight" || strings.HasPrefix(name, "model.layers.")
	}
	outWeights := make(map[string][]float32)
	outShapes := make(map[string][]int)
	for name, values := range weights {
		if keep(name) {
			outWeights[name] = values
			outShapes[name] = shapes[name]
		}
	}
	return outWeights, outShapes
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
			gate, err := tensorcatalog.Shape(shapes, names.gate, 2)
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
	if _, err := tensorcatalog.Shape(shapes, finalNormName, 1); err != nil {
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
