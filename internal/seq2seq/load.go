// Package seq2seq owns the layered-attention encoder-decoder capability
// (ladder rung: the first encoder-decoder task). A capability port from
// adaptive_new's layeredAttentionSeq2Seq expressed fresh against overgo's
// owners; parity is judged against the imported JAX forward oracle.
//
// Architecture (attention-only layers, no feed-forward anywhere):
// tied embeddings scaled by sqrt(d); bidirectional RoPE encoder; decoder
// alternates causal RoPE self-attention and bidirectional RoPE-free
// cross-attention over the encoder output; every layer is pre-norm with
// 1+w RMSNorm scales, per-head 1+w q/k norms, and a sigmoid-gated scalar
// residual. Logits are the tied embedding head over the final 1+w RMSNorm.
//
// Every dimension is DERIVED from the artifact's own tensor shapes; config
// supplies only non-shape facts (rope_theta, rms_norm_eps, start/eos ids).
package seq2seq

import (
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/hostmath"
	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
)

// Dims: model geometry, derived from tensor shapes and config.
type Dims struct {
	Vocab         int
	DModel        int
	Heads         int
	KVHeads       int
	HeadDim       int
	EncoderLayers int
	DecoderLayers int
	RopeTheta     float64
	RMSEps        float64
	StartToken    int
	EOSToken      int
}

// attnBlock: one attention-only layer's weight views. Norm weights carry the
// 1+w fold applied at load; gate carries the sigmoid fold. All slices view
// the loaded f32 tensors.
type attnBlock struct {
	inNorm       []float32 // [d], folded 1+w
	q, k, v, o   []uint16  // BF16 [heads*hd,d], [kv*hd,d], [kv*hd,d], [d,heads*hd]
	qNorm, kNorm []float32 // [hd], folded 1+w
	gate         float32   // folded sigmoid(raw)
}

// Model: loaded weights and derived dims.
type Model struct {
	Dims         Dims
	embed        []uint16  // BF16 [vocab,d]; tied head
	encFinalNorm []float32 // [d], folded 1+w
	decFinalNorm []float32 // [d], folded 1+w
	encoder      []attnBlock
	decoderSelf  []attnBlock
	decoderCross []attnBlock
	invFreq      []float64 // rotate-half ladder from RopeTheta, HeadDim
	embedScale   float32   // sqrt(d): reference scaled word embedding
	scoreScale   float32   // 1/sqrt(hd), folded into q before attention
}

type artifactConfig struct {
	ModelType           string  `json:"model_type"`
	RMSNormEps          float64 `json:"rms_norm_eps"`
	RopeTheta           float64 `json:"rope_theta"`
	DecoderStartTokenID int     `json:"decoder_start_token_id"`
	EOSTokenID          int     `json:"eos_token_id"`
}

// Checkpoint tensor-name wiring (transformers layout of this artifact family;
// names are artifact facts, not model constants).
const (
	embedName        = "model.embed_tokens.weight"
	encFinalNormName = "model.encoder.final_norm.weight"
	decFinalNormName = "model.decoder.norm.weight"
	encLayerPrefix   = "model.encoder.layers."
	decLayerPrefix   = "model.decoder.layers."
)

// Load opens the safetensors artifact, retains matrix tensors as BF16,
// promotes transformed vectors to F32, and derives dims from shapes.
func Load(directory string) (*Model, error) {
	config, err := loadConfig(filepath.Join(directory, "config.json"))
	if err != nil {
		return nil, err
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()

	shapes := make(map[string][]int, len(source.Tensors))
	for name, tensor := range source.Tensors {
		dims := make([]int, len(tensor.Shape))
		for i, dim := range tensor.Shape {
			if dim == 0 || dim > 1<<31 {
				return nil, fmt.Errorf("seq2seq: tensor %q dimension %d out of range", name, dim)
			}
			dims[i] = int(dim)
		}
		shapes[name] = dims
	}
	weights := make(map[string][]float32, len(source.Tensors))
	read := func(name string, wantShape ...int) ([]float32, error) {
		if cached, ok := weights[name]; ok {
			return cached, nil
		}
		tensor, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("seq2seq: tensor %q missing", name)
		}
		shape := shapes[name]
		if len(shape) != len(wantShape) {
			return nil, fmt.Errorf("seq2seq: tensor %q rank %d, want %d", name, len(shape), len(wantShape))
		}
		for i, want := range wantShape {
			if want > 0 && shape[i] != want {
				return nil, fmt.Errorf("seq2seq: tensor %q shape %v, want dim %d = %d", name, shape, i, want)
			}
		}
		values, err := safetensors.ReadF32(tensor)
		if err != nil {
			return nil, fmt.Errorf("seq2seq: tensor %q: %w", name, err)
		}
		weights[name] = values
		return values, nil
	}
	readBF16 := func(name string, wantShape ...int) ([]uint16, error) {
		tensor, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("seq2seq: tensor %q missing", name)
		}
		shape := shapes[name]
		if len(shape) != len(wantShape) {
			return nil, fmt.Errorf("seq2seq: tensor %q rank %d, want %d", name, len(shape), len(wantShape))
		}
		for i, want := range wantShape {
			if want > 0 && shape[i] != want {
				return nil, fmt.Errorf("seq2seq: tensor %q shape %v, want dim %d = %d", name, shape, i, want)
			}
		}
		values, err := safetensors.ReadBF16(tensor)
		if err != nil {
			return nil, fmt.Errorf("seq2seq: tensor %q: %w", name, err)
		}
		return values, nil
	}

	dims, err := deriveDims(shapes, config)
	if err != nil {
		return nil, err
	}
	m := &Model{
		Dims:       dims,
		invFreq:    hostmath.RopeInvFreq(dims.RopeTheta, dims.HeadDim),
		embedScale: float32(math.Sqrt(float64(dims.DModel))),
		scoreScale: float32(1 / math.Sqrt(float64(dims.HeadDim))),
	}
	if m.embed, err = readBF16(embedName, dims.Vocab, dims.DModel); err != nil {
		return nil, err
	}
	if m.encFinalNorm, err = read(encFinalNormName, dims.DModel); err != nil {
		return nil, err
	}
	if m.decFinalNorm, err = read(decFinalNormName, dims.DModel); err != nil {
		return nil, err
	}
	fold1p(m.encFinalNorm)
	fold1p(m.decFinalNorm)

	loadStack := func(prefix, attn, norm, gate string, layers int) ([]attnBlock, error) {
		d, qw, kvw, hd := dims.DModel, dims.Heads*dims.HeadDim, dims.KVHeads*dims.HeadDim, dims.HeadDim
		stack := make([]attnBlock, layers)
		for layer := range stack {
			base := fmt.Sprintf("%s%d.", prefix, layer)
			a := base + attn + "."
			block := &stack[layer]
			var err error
			if block.inNorm, err = read(base+norm, d); err != nil {
				return nil, err
			}
			if block.q, err = readBF16(a+"q_proj.weight", qw, d); err != nil {
				return nil, err
			}
			if block.k, err = readBF16(a+"k_proj.weight", kvw, d); err != nil {
				return nil, err
			}
			if block.v, err = readBF16(a+"v_proj.weight", kvw, d); err != nil {
				return nil, err
			}
			if block.o, err = readBF16(a+"out_proj.weight", d, qw); err != nil {
				return nil, err
			}
			if block.qNorm, err = read(a+"q_norm.weight", hd); err != nil {
				return nil, err
			}
			if block.kNorm, err = read(a+"k_norm.weight", hd); err != nil {
				return nil, err
			}
			gateValues, err := read(base+gate, 1)
			if err != nil {
				return nil, err
			}
			fold1p(block.inNorm)
			fold1p(block.qNorm)
			fold1p(block.kNorm)
			block.gate = float32(1 / (1 + math.Exp(-float64(gateValues[0]))))
		}
		return stack, nil
	}
	if m.encoder, err = loadStack(encLayerPrefix, "self_attn", "input_layernorm.weight", "attn_gate", dims.EncoderLayers); err != nil {
		return nil, err
	}
	if m.decoderSelf, err = loadStack(decLayerPrefix, "self_attn", "input_layernorm.weight", "self_attn_gate", dims.DecoderLayers); err != nil {
		return nil, err
	}
	if m.decoderCross, err = loadStack(decLayerPrefix, "encoder_attn", "encoder_attn_layer_norm.weight", "cross_attn_gate", dims.DecoderLayers); err != nil {
		return nil, err
	}
	return m, nil
}

// fold1p: the checkpoint stores RMSNorm scales as w with effective scale 1+w
// (zero-centered init); fold once at load so runtime norms are plain.
func fold1p(w []float32) {
	for i := range w {
		w[i]++
	}
}

func loadConfig(path string) (artifactConfig, error) {
	var config artifactConfig
	if err := jsonfile.Decode(path, &config); err != nil {
		return artifactConfig{}, fmt.Errorf("seq2seq: parse config.json: %w", err)
	}
	if config.RopeTheta <= 0 {
		return artifactConfig{}, fmt.Errorf("seq2seq: config.json lacks positive rope_theta")
	}
	if config.RMSNormEps <= 0 {
		return artifactConfig{}, fmt.Errorf("seq2seq: config.json lacks positive rms_norm_eps")
	}
	return config, nil
}

// deriveDims: every geometric fact from tensor shapes; config supplies only
// what shapes cannot say (rope_theta, eps, start/eos token ids).
func deriveDims(shapes map[string][]int, config artifactConfig) (Dims, error) {
	var d Dims
	embed, ok := shapes[embedName]
	if !ok || len(embed) != 2 {
		return d, fmt.Errorf("seq2seq: embedding %q missing or non-matrix", embedName)
	}
	d.Vocab, d.DModel = embed[0], embed[1]

	headNorm, ok := shapes[encLayerPrefix+"0.self_attn.q_norm.weight"]
	if !ok || len(headNorm) != 1 {
		return d, fmt.Errorf("seq2seq: layer-0 q_norm missing; head dim underivable")
	}
	d.HeadDim = headNorm[0]

	qProj, ok := shapes[encLayerPrefix+"0.self_attn.q_proj.weight"]
	if !ok || len(qProj) != 2 || qProj[0]%d.HeadDim != 0 {
		return d, fmt.Errorf("seq2seq: layer-0 q_proj incompatible with head dim %d", d.HeadDim)
	}
	d.Heads = qProj[0] / d.HeadDim

	kProj, ok := shapes[encLayerPrefix+"0.self_attn.k_proj.weight"]
	if !ok || len(kProj) != 2 || kProj[0]%d.HeadDim != 0 {
		return d, fmt.Errorf("seq2seq: layer-0 k_proj incompatible with head dim %d", d.HeadDim)
	}
	d.KVHeads = kProj[0] / d.HeadDim
	if d.KVHeads <= 0 || d.Heads%d.KVHeads != 0 {
		return d, fmt.Errorf("seq2seq: heads %d not grouped by kv heads %d", d.Heads, d.KVHeads)
	}

	var err error
	if d.EncoderLayers, err = layerCount(shapes, encLayerPrefix, ".self_attn.q_proj.weight"); err != nil {
		return d, err
	}
	if d.DecoderLayers, err = layerCount(shapes, decLayerPrefix, ".self_attn.q_proj.weight"); err != nil {
		return d, err
	}
	d.RopeTheta = config.RopeTheta
	d.RMSEps = config.RMSNormEps
	d.StartToken = config.DecoderStartTokenID
	d.EOSToken = config.EOSTokenID
	return d, nil
}

// layerCount: contiguous prefix+N indices starting at zero.
func layerCount(shapes map[string][]int, prefix, suffix string) (int, error) {
	count := 0
	for {
		if _, ok := shapes[fmt.Sprintf("%s%d%s", prefix, count, suffix)]; !ok {
			break
		}
		count++
	}
	if count == 0 {
		return 0, fmt.Errorf("seq2seq: no layers under %q", prefix)
	}
	return count, nil
}
