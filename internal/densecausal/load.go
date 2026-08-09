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

	"overgo/internal/safetensors"
)

// Dims: model geometry, derived from tensor shapes and config.
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
}

// Model: loaded weights (f32), shapes, and derived dims. lm head is the
// embedding when tied (the only layout this path has evidence for).
type Model struct {
	Dims    Dims
	Weights map[string][]float32
	Shapes  map[string][]int
}

type artifactConfig struct {
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
	return NewModel(weights, shapes, config.NumAttentionHeads, config.HeadDim, config.RopeTheta, config.RMSNormEps)
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
	if _, untied := shapes["lm_head.weight"]; untied {
		return nil, fmt.Errorf("densecausal: untied lm_head.weight present; only tied embeddings are verified")
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
	}
	if _, err := shapeOf(shapes, "model.norm.weight", 1); err != nil {
		return nil, err
	}
	return &Model{Dims: d, Weights: weights, Shapes: shapes}, nil
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
