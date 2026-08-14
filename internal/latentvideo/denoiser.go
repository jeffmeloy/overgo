// Latent-video denoiser: a conditioned diffusion transformer routed through
// the shared tensor graph. Two graphs own the compute: a context graph
// projecting the text context to per-block cross-attention K/V once per
// branch, and a step graph running patch embedding, every transformer
// block, and the modulated head. The graph is the single arithmetic
// definition; reference and CUDA executors both consume it. Host boundaries
// are layout-only (patchify/unpatchify) plus the sampling/timestep math in
// sampler.go and timestep.go.
package latentvideo

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"overgo/internal/model"
	"overgo/internal/safetensors"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
	"overgo/internal/tensor/reference"
)

// DenoiserPolicy: published pipeline facts the checkpoint cannot carry.
type DenoiserPolicy struct {
	NumTrainTimesteps   int
	SinusoidalPeriod    int
	RotaryFrequencyBase float64
	VAEStride           [3]int
}

// DenoiserConfig: checkpoint config.json facts plus policy.
type DenoiserConfig struct {
	Dim, FFNDim, FreqDim int
	InDim, OutDim        int
	NumHeads, NumLayers  int
	TextLen              int
	Eps                  float64
	PatchSize            [3]int
	Policy               DenoiserPolicy
}

// LoadDenoiserConfig: config.json plus the patch geometry carried by the
// patch embedding tensor shape.
func LoadDenoiserConfig(dir string, policy DenoiserPolicy) (DenoiserConfig, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return DenoiserConfig{}, err
	}
	var parsed struct {
		Dim       int     `json:"dim"`
		FFNDim    int     `json:"ffn_dim"`
		FreqDim   int     `json:"freq_dim"`
		InDim     int     `json:"in_dim"`
		OutDim    int     `json:"out_dim"`
		NumHeads  int     `json:"num_heads"`
		NumLayers int     `json:"num_layers"`
		TextLen   int     `json:"text_len"`
		Eps       float64 `json:"eps"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return DenoiserConfig{}, err
	}
	source, err := safetensors.OpenSource(dir)
	if err != nil {
		return DenoiserConfig{}, err
	}
	defer source.Close()
	patch, ok := source.Tensors["patch_embedding.weight"]
	if !ok || len(patch.Shape) != 5 {
		return DenoiserConfig{}, fmt.Errorf("denoiser config: patch_embedding.weight rank-5 tensor is absent")
	}
	cfg := DenoiserConfig{
		Dim: parsed.Dim, FFNDim: parsed.FFNDim, FreqDim: parsed.FreqDim,
		InDim: parsed.InDim, OutDim: parsed.OutDim,
		NumHeads: parsed.NumHeads, NumLayers: parsed.NumLayers,
		TextLen: parsed.TextLen, Eps: parsed.Eps,
		PatchSize: [3]int{int(patch.Shape[2]), int(patch.Shape[3]), int(patch.Shape[4])},
		Policy:    policy,
	}
	if uint64(cfg.Dim) != patch.Shape[0] || uint64(cfg.InDim) != patch.Shape[1] {
		return DenoiserConfig{}, fmt.Errorf("denoiser config: patch embedding shape %v incompatible with dim=%d in_dim=%d", patch.Shape, cfg.Dim, cfg.InDim)
	}
	return cfg, cfg.validate()
}

func (c DenoiserConfig) validate() error {
	for _, field := range [...]struct {
		name  string
		value int
	}{
		{"dim", c.Dim}, {"ffn_dim", c.FFNDim}, {"freq_dim", c.FreqDim},
		{"in_dim", c.InDim}, {"out_dim", c.OutDim}, {"num_heads", c.NumHeads},
		{"num_layers", c.NumLayers}, {"text_len", c.TextLen},
		{"num_train_timesteps", c.Policy.NumTrainTimesteps},
		{"sinusoidal_period", c.Policy.SinusoidalPeriod},
	} {
		if field.value <= 0 {
			return fmt.Errorf("denoiser config: %s must be positive, got %d", field.name, field.value)
		}
	}
	if c.Eps <= 0 || c.Policy.RotaryFrequencyBase <= 0 {
		return fmt.Errorf("denoiser config: eps/rotary base must be positive")
	}
	if c.Dim%c.NumHeads != 0 || (c.Dim/c.NumHeads)%2 != 0 {
		return fmt.Errorf("denoiser config: dim %d must split into even-width heads (%d)", c.Dim, c.NumHeads)
	}
	for axis := range c.PatchSize {
		if c.PatchSize[axis] <= 0 || c.Policy.VAEStride[axis] <= 0 {
			return fmt.Errorf("denoiser config: patch/vae stride axis %d must be positive", axis)
		}
	}
	return nil
}

func (c DenoiserConfig) patchIn() int {
	return c.InDim * c.PatchSize[0] * c.PatchSize[1] * c.PatchSize[2]
}

func (c DenoiserConfig) patchOut() int {
	return c.OutDim * c.PatchSize[0] * c.PatchSize[1] * c.PatchSize[2]
}

// LatentGeometry: compiled latent extent for one requested clip.
type LatentGeometry struct {
	Channels                                int
	LatentFrames, LatentHeight, LatentWidth int
	Grid                                    [3]int
	Seq                                     int
}

func (c DenoiserConfig) CompileLatentGeometry(frames, width, height int) (LatentGeometry, error) {
	if frames <= 0 || width <= 0 || height <= 0 {
		return LatentGeometry{}, fmt.Errorf("latent geometry: frames/width/height must be positive, got %d/%d/%d", frames, width, height)
	}
	geometry := LatentGeometry{
		Channels:     c.InDim,
		LatentFrames: (frames-1)/c.Policy.VAEStride[0] + 1,
		LatentHeight: height / c.Policy.VAEStride[1],
		LatentWidth:  width / c.Policy.VAEStride[2],
	}
	if geometry.LatentHeight <= 0 || geometry.LatentWidth <= 0 ||
		geometry.LatentHeight%c.PatchSize[1] != 0 || geometry.LatentWidth%c.PatchSize[2] != 0 ||
		geometry.LatentFrames%c.PatchSize[0] != 0 {
		return LatentGeometry{}, fmt.Errorf("latent geometry %dx%dx%d incompatible with patch %v", geometry.LatentFrames, geometry.LatentHeight, geometry.LatentWidth, c.PatchSize)
	}
	geometry.Grid = [3]int{
		geometry.LatentFrames / c.PatchSize[0],
		geometry.LatentHeight / c.PatchSize[1],
		geometry.LatentWidth / c.PatchSize[2],
	}
	geometry.Seq = geometry.Grid[0] * geometry.Grid[1] * geometry.Grid[2]
	return geometry, nil
}

func (g LatentGeometry) Elements() int {
	return g.Channels * g.LatentFrames * g.LatentHeight * g.LatentWidth
}

// DenoiserWeights: the full F32 tensor set, loaded once and shared by both
// graphs without duplication.
type DenoiserWeights struct {
	values map[string][]float32
	Bytes  int64
}

func (w *DenoiserWeights) tensor(name string) []float32 {
	if w == nil {
		return nil
	}
	return w.values[name]
}

func denoiserBlockPrefix(layer int) string {
	return fmt.Sprintf("blocks.%d.", layer)
}

// denoiserTensorLengths: complete name -> element-count contract.
func denoiserTensorLengths(c DenoiserConfig) map[string]int {
	d, f := c.Dim, c.FFNDim
	lengths := map[string]int{
		"patch_embedding.weight":   d * c.patchIn(),
		"patch_embedding.bias":     d,
		"time_embedding.0.weight":  d * c.FreqDim,
		"time_embedding.0.bias":    d,
		"time_embedding.2.weight":  d * d,
		"time_embedding.2.bias":    d,
		"time_projection.1.weight": 6 * d * d,
		"time_projection.1.bias":   6 * d,
		"head.modulation":          2 * d,
		"head.head.weight":         c.patchOut() * d,
		"head.head.bias":           c.patchOut(),
	}
	for layer := 0; layer < c.NumLayers; layer++ {
		prefix := denoiserBlockPrefix(layer)
		for _, attention := range []string{"self_attn.", "cross_attn."} {
			for _, projection := range []string{"q", "k", "v", "o"} {
				lengths[prefix+attention+projection+".weight"] = d * d
				lengths[prefix+attention+projection+".bias"] = d
			}
			lengths[prefix+attention+"norm_q.weight"] = d
			lengths[prefix+attention+"norm_k.weight"] = d
		}
		lengths[prefix+"ffn.0.weight"] = f * d
		lengths[prefix+"ffn.0.bias"] = f
		lengths[prefix+"ffn.2.weight"] = d * f
		lengths[prefix+"ffn.2.bias"] = d
		lengths[prefix+"modulation"] = 6 * d
		lengths[prefix+"norm3.weight"] = d
		lengths[prefix+"norm3.bias"] = d
	}
	return lengths
}

// LoadDenoiserWeights: reads every denoiser tensor as F32 exactly once.
func LoadDenoiserWeights(dir string, c DenoiserConfig) (*DenoiserWeights, error) {
	source, err := safetensors.OpenSource(dir)
	if err != nil {
		return nil, err
	}
	defer source.Close()
	lengths := denoiserTensorLengths(c)
	weights := &DenoiserWeights{values: make(map[string][]float32, len(lengths))}
	for name, want := range lengths {
		payload, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("denoiser weights: tensor %s is absent", name)
		}
		if payload.DType != "F32" || payload.Elements() != uint64(want) {
			return nil, fmt.Errorf("denoiser weights: tensor %s dtype=%s elements=%d, want F32 %d", name, payload.DType, payload.Elements(), want)
		}
		reader, err := safetensors.F32Reader(payload)
		if err != nil {
			return nil, fmt.Errorf("denoiser weights %s: %w", name, err)
		}
		raw := make([]byte, want*4)
		if _, err := io.ReadFull(reader, raw); err != nil {
			return nil, fmt.Errorf("denoiser weights %s payload: %w", name, err)
		}
		decoded := make([]float32, want)
		for i := range decoded {
			bits := uint32(raw[4*i]) | uint32(raw[4*i+1])<<8 | uint32(raw[4*i+2])<<16 | uint32(raw[4*i+3])<<24
			decoded[i] = math.Float32frombits(bits)
		}
		weights.values[name] = decoded
		weights.Bytes += payload.Size()
	}
	return weights, nil
}

// TimestepWeights: the timestep-conditioning slices bound to this artifact.
func (w *DenoiserWeights) TimestepWeights(c DenoiserConfig) TimestepConditioningWeights {
	return TimestepConditioningWeights{
		Embed0W: w.tensor("time_embedding.0.weight"), Embed0B: w.tensor("time_embedding.0.bias"),
		Embed2W: w.tensor("time_embedding.2.weight"), Embed2B: w.tensor("time_embedding.2.bias"),
		ProjectW: w.tensor("time_projection.1.weight"), ProjectB: w.tensor("time_projection.1.bias"),
		FreqDim: c.FreqDim, Dim: c.Dim, Period: c.Policy.SinusoidalPeriod,
	}
}

// GraphRunner: one backend executing a tensor graph. reference.Execute
// satisfies it directly; the CUDA executor satisfies it through a context
// closure.
type GraphRunner func(outputs []*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) (map[*tensor.Tensor]reference.Value, error)

// DenoiserProgram: both compiled graphs plus their weight bindings.
type DenoiserProgram struct {
	Config   DenoiserConfig
	Geometry LatentGeometry

	// matmulWeightType: storage type of every rank-2 projection weight input
	// (F32 exact; BF16 = storage rounding for the device tensor-core path).
	matmulWeightType dtype.Type

	weights         *DenoiserWeights
	timestepWeights TimestepConditioningWeights

	contextInput        *tensor.Tensor
	contextWeightInputs map[string]*tensor.Tensor
	contextKeys         []*tensor.Tensor
	contextValues       []*tensor.Tensor

	stepPatch        *tensor.Tensor
	stepBlockE       *tensor.Tensor
	stepHeadE        *tensor.Tensor
	stepCrossKeys    []*tensor.Tensor
	stepCrossValues  []*tensor.Tensor
	stepWeightInputs map[string]*tensor.Tensor

	// Blocks expose every intra-block seam for parity probes; Head is the
	// [patchOut, seq] pre-unpatchify output.
	Blocks []model.ConditionedDiffusionBlockResult
	Head   *tensor.Tensor
}

// weightInputBinder: binds named weight inputs on one builder; rank-2
// projection weights take the program's matmul storage type, everything
// else stays F32.
type weightInputBinder struct {
	builder    *tensor.Builder
	inputs     map[string]*tensor.Tensor
	matmulType dtype.Type
}

func (b weightInputBinder) input(name string, dimensions ...uint64) *tensor.Tensor {
	storage := dtype.F32
	if len(dimensions) == 2 {
		storage = b.matmulType
	}
	node := b.builder.Input(name, storage, tensor.MustShape(dimensions...))
	b.inputs[name] = node
	return node
}

func crossAttentionContextWeights(bind weightInputBinder, prefix string, d uint64) model.ConditionedDiffusionAttentionWeights {
	return model.ConditionedDiffusionAttentionWeights{
		Key: bind.input(prefix+"k.weight", d, d), KeyBias: bind.input(prefix+"k.bias", d),
		Value: bind.input(prefix+"v.weight", d, d), ValueBias: bind.input(prefix+"v.bias", d),
		KeyNorm: bind.input(prefix+"norm_k.weight", d),
	}
}

func attentionQueryOutputWeights(bind weightInputBinder, prefix string, d uint64) model.ConditionedDiffusionAttentionWeights {
	return model.ConditionedDiffusionAttentionWeights{
		Query: bind.input(prefix+"q.weight", d, d), QueryBias: bind.input(prefix+"q.bias", d),
		Output: bind.input(prefix+"o.weight", d, d), OutputBias: bind.input(prefix+"o.bias", d),
		QueryNorm: bind.input(prefix+"norm_q.weight", d),
	}
}

func selfAttentionWeights(bind weightInputBinder, prefix string, d uint64) model.ConditionedDiffusionAttentionWeights {
	weights := attentionQueryOutputWeights(bind, prefix, d)
	weights.Key = bind.input(prefix+"k.weight", d, d)
	weights.KeyBias = bind.input(prefix+"k.bias", d)
	weights.Value = bind.input(prefix+"v.weight", d, d)
	weights.ValueBias = bind.input(prefix+"v.bias", d)
	weights.KeyNorm = bind.input(prefix+"norm_k.weight", d)
	return weights
}

// DenoiserPrecision: storage-rounding levers. MatmulWeights selects the
// rank-2 projection weight storage (F32 exact; BF16 device tensor-core
// path). RoundAttentionStorage declares BF16 rounding of attention q/k/v in
// the graph (backends may fuse to a tensor-core flash kernel).
type DenoiserPrecision struct {
	MatmulWeights         dtype.Type
	RoundAttentionStorage bool
}

// CompileDenoiserProgram: builds the context and step graphs for one latent
// geometry with exact F32 storage everywhere.
func CompileDenoiserProgram(c DenoiserConfig, weights *DenoiserWeights, geometry LatentGeometry) (*DenoiserProgram, error) {
	return CompileDenoiserProgramPrecision(c, weights, geometry, DenoiserPrecision{MatmulWeights: dtype.F32})
}

// CompileDenoiserProgramPrecision: builds both graphs under one precision
// declaration.
func CompileDenoiserProgramPrecision(c DenoiserConfig, weights *DenoiserWeights, geometry LatentGeometry, precision DenoiserPrecision) (*DenoiserProgram, error) {
	matmulWeightType := precision.MatmulWeights
	if weights == nil {
		return nil, fmt.Errorf("denoiser program: weights are nil")
	}
	if geometry.Seq <= 0 {
		return nil, fmt.Errorf("denoiser program: geometry has no tokens")
	}
	if matmulWeightType != dtype.F32 && matmulWeightType != dtype.BF16 {
		return nil, fmt.Errorf("denoiser program: matmul weight type %s is unsupported", matmulWeightType)
	}
	matmulCompute := tensor.MulMatComputeExact
	if matmulWeightType == dtype.BF16 {
		matmulCompute = tensor.MulMatComputeBF16TensorCore
	}
	d := uint64(c.Dim)
	heads := uint64(c.NumHeads)
	headWidth := d / heads
	textLen := uint64(c.TextLen)
	seq := uint64(geometry.Seq)
	program := &DenoiserProgram{
		Config: c, Geometry: geometry, weights: weights,
		timestepWeights: weights.TimestepWeights(c), matmulWeightType: matmulWeightType,
	}
	options := model.ConditionedDiffusionBlockOptions{
		Dim: d, Heads: heads, FFNDim: uint64(c.FFNDim), Epsilon: float32(c.Eps),
		RotaryBase:            float32(c.Policy.RotaryFrequencyBase),
		AxisChannels:          model.ThreeAxisRotaryChannels(headWidth),
		RoundAttentionStorage: precision.RoundAttentionStorage,
	}
	for axis := range options.AxisPositions {
		options.AxisPositions[axis] = make([]uint32, geometry.Seq)
	}
	for token := 0; token < geometry.Seq; token++ {
		spatial := geometry.Grid[1] * geometry.Grid[2]
		options.AxisPositions[0][token] = uint32(token / spatial)
		options.AxisPositions[1][token] = uint32(token % spatial / geometry.Grid[2])
		options.AxisPositions[2][token] = uint32(token % geometry.Grid[2])
	}
	diffusion, err := model.CompileConditionedDiffusionProgram(options)
	if err != nil {
		return nil, err
	}

	// Context graph: per-block cross-attention K/V from the text context.
	contextBuilder := tensor.NewBuilder()
	contextBuilder.SetMulMatCompute(matmulCompute)
	program.contextWeightInputs = make(map[string]*tensor.Tensor)
	contextBind := weightInputBinder{builder: contextBuilder, inputs: program.contextWeightInputs, matmulType: matmulWeightType}
	program.contextInput = contextBuilder.Input("context", dtype.F32, tensor.MustShape(d, textLen))
	for layer := 0; layer < c.NumLayers; layer++ {
		prefix := denoiserBlockPrefix(layer) + "cross_attn."
		key, value, err := diffusion.BuildCrossContext(
			contextBuilder, program.contextInput, crossAttentionContextWeights(contextBind, prefix, d),
		)
		if err != nil {
			return nil, fmt.Errorf("denoiser program context layer %d: %w", layer, err)
		}
		program.contextKeys = append(program.contextKeys, key)
		program.contextValues = append(program.contextValues, value)
	}
	if err := contextBuilder.Err(); err != nil {
		return nil, fmt.Errorf("denoiser program context graph: %w", err)
	}

	// Step graph: patch embedding -> blocks -> head.
	builder := tensor.NewBuilder()
	builder.SetMulMatCompute(matmulCompute)
	program.stepWeightInputs = make(map[string]*tensor.Tensor)
	bind := weightInputBinder{builder: builder, inputs: program.stepWeightInputs, matmulType: matmulWeightType}
	program.stepPatch = builder.Input("patch_tokens", dtype.F32, tensor.MustShape(uint64(c.patchIn()), seq))
	program.stepBlockE = builder.Input("conditioning_block", dtype.F32, tensor.MustShape(6*d))
	program.stepHeadE = builder.Input("conditioning_head", dtype.F32, tensor.MustShape(d))
	embedW := bind.input("patch_embedding.weight", uint64(c.patchIn()), d)
	embedB := bind.input("patch_embedding.bias", d)
	hidden := builder.Add(builder.MulMat(embedW, program.stepPatch), embedB)

	for layer := 0; layer < c.NumLayers; layer++ {
		prefix := denoiserBlockPrefix(layer)
		crossKey := builder.Input(prefix+"cross_key", dtype.F32, tensor.MustShape(headWidth, heads, textLen))
		crossValue := builder.Input(prefix+"cross_value", dtype.F32, tensor.MustShape(headWidth, heads, textLen))
		program.stepCrossKeys = append(program.stepCrossKeys, crossKey)
		program.stepCrossValues = append(program.stepCrossValues, crossValue)
		blockWeights := model.ConditionedDiffusionBlockWeights{
			Modulation:      bind.input(prefix+"modulation", 6*d),
			SelfAttention:   selfAttentionWeights(bind, prefix+"self_attn.", d),
			CrossAttention:  attentionQueryOutputWeights(bind, prefix+"cross_attn.", d),
			CrossNormWeight: bind.input(prefix+"norm3.weight", d),
			CrossNormBias:   bind.input(prefix+"norm3.bias", d),
			FFNExpand:       bind.input(prefix+"ffn.0.weight", d, uint64(c.FFNDim)),
			FFNExpandBias:   bind.input(prefix+"ffn.0.bias", uint64(c.FFNDim)),
			FFNContract:     bind.input(prefix+"ffn.2.weight", uint64(c.FFNDim), d),
			FFNContractBias: bind.input(prefix+"ffn.2.bias", d),
		}
		result, err := diffusion.BuildBlock(
			builder, hidden, program.stepBlockE, crossKey, crossValue, blockWeights,
		)
		if err != nil {
			return nil, fmt.Errorf("denoiser program block %d: %w", layer, err)
		}
		program.Blocks = append(program.Blocks, result)
		hidden = result.Output
	}
	head, err := diffusion.BuildHead(
		builder, hidden, program.stepHeadE,
		bind.input("head.modulation", 2*d),
		bind.input("head.head.weight", d, uint64(c.patchOut())),
		bind.input("head.head.bias", uint64(c.patchOut())),
	)
	if err != nil {
		return nil, fmt.Errorf("denoiser program head: %w", err)
	}
	program.Head = head
	if err := builder.Err(); err != nil {
		return nil, fmt.Errorf("denoiser program step graph: %w", err)
	}
	return program, nil
}

func (p *DenoiserProgram) weightFeeds(inputs map[string]*tensor.Tensor, feeds map[*tensor.Tensor]reference.Value) error {
	if p.weights == nil {
		return fmt.Errorf("denoiser host weights released to resident session")
	}
	for name, node := range inputs {
		if node.Type != dtype.F32 {
			return fmt.Errorf("denoiser weight feed %s: host feeds require F32 storage, program compiled %s", name, node.Type)
		}
		data := p.weights.tensor(name)
		elements, err := node.Shape.Elements()
		if err != nil || uint64(len(data)) != elements {
			return fmt.Errorf("denoiser weight feed %s: have %d elements, need %d", name, len(data), elements)
		}
		feeds[node] = reference.Value{Shape: node.Shape, Data: data}
	}
	return nil
}

func feedValue(node *tensor.Tensor, data []float32, what string) (reference.Value, error) {
	elements, err := node.Shape.Elements()
	if err != nil || uint64(len(data)) != elements {
		return reference.Value{}, fmt.Errorf("denoiser feed %s: have %d elements, need %d", what, len(data), elements)
	}
	return reference.Value{Shape: node.Shape, Data: data}, nil
}

// ContextProjection: one branch's per-block cross-attention K/V.
type ContextProjection struct {
	Keys, Values [][]float32
}

// ProjectContext: runs the context graph once for one text context.
func (p *DenoiserProgram) ProjectContext(run GraphRunner, context []float32) (ContextProjection, error) {
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.contextWeightInputs)+1)
	if err := p.weightFeeds(p.contextWeightInputs, feeds); err != nil {
		return ContextProjection{}, err
	}
	value, err := feedValue(p.contextInput, context, "context")
	if err != nil {
		return ContextProjection{}, err
	}
	feeds[p.contextInput] = value
	outputs := make([]*tensor.Tensor, 0, 2*len(p.contextKeys))
	outputs = append(outputs, p.contextKeys...)
	outputs = append(outputs, p.contextValues...)
	results, err := run(outputs, feeds)
	if err != nil {
		return ContextProjection{}, fmt.Errorf("denoiser context projection: %w", err)
	}
	projection := ContextProjection{}
	for layer := range p.contextKeys {
		projection.Keys = append(projection.Keys, results[p.contextKeys[layer]].Data)
		projection.Values = append(projection.Values, results[p.contextValues[layer]].Data)
	}
	return projection, nil
}

// Forward: one step-graph execution over embedded patch tokens; outputs
// selects which graph tensors to materialize (Head plus any seams).
func (p *DenoiserProgram) Forward(
	run GraphRunner,
	patchTokens, blockE, headE []float32,
	context ContextProjection,
	outputs []*tensor.Tensor,
) (map[*tensor.Tensor]reference.Value, error) {
	if len(context.Keys) != len(p.stepCrossKeys) || len(context.Values) != len(p.stepCrossValues) {
		return nil, fmt.Errorf("denoiser forward: context projection has %d/%d layers, need %d", len(context.Keys), len(context.Values), len(p.stepCrossKeys))
	}
	feeds := make(map[*tensor.Tensor]reference.Value, len(p.stepWeightInputs)+3+2*len(p.stepCrossKeys))
	if err := p.weightFeeds(p.stepWeightInputs, feeds); err != nil {
		return nil, err
	}
	for _, feed := range []struct {
		node *tensor.Tensor
		data []float32
		what string
	}{
		{p.stepPatch, patchTokens, "patch tokens"},
		{p.stepBlockE, blockE, "block conditioning"},
		{p.stepHeadE, headE, "head conditioning"},
	} {
		value, err := feedValue(feed.node, feed.data, feed.what)
		if err != nil {
			return nil, err
		}
		feeds[feed.node] = value
	}
	for layer := range p.stepCrossKeys {
		key, err := feedValue(p.stepCrossKeys[layer], context.Keys[layer], "cross key")
		if err != nil {
			return nil, err
		}
		value, err := feedValue(p.stepCrossValues[layer], context.Values[layer], "cross value")
		if err != nil {
			return nil, err
		}
		feeds[p.stepCrossKeys[layer]] = key
		feeds[p.stepCrossValues[layer]] = value
	}
	return run(outputs, feeds)
}

// PatchifyLatent: channel-major latent -> token-major patch columns, the
// layout consumed by the step graph's embedding matmul.
func (p *DenoiserProgram) PatchifyLatent(sample []float32) ([]float32, error) {
	c, g := p.Config, p.Geometry
	if len(sample) != g.Elements() {
		return nil, fmt.Errorf("patchify: sample has %d elements, need %d", len(sample), g.Elements())
	}
	ps0, ps1, ps2 := c.PatchSize[0], c.PatchSize[1], c.PatchSize[2]
	frames, height, width := g.LatentFrames, g.LatentHeight, g.LatentWidth
	patchIn := c.patchIn()
	out := make([]float32, g.Seq*patchIn)
	for token := 0; token < g.Seq; token++ {
		ft := token / (g.Grid[1] * g.Grid[2])
		remainder := token % (g.Grid[1] * g.Grid[2])
		hy, wx := remainder/g.Grid[2], remainder%g.Grid[2]
		row := out[token*patchIn : (token+1)*patchIn]
		for channel := 0; channel < c.InDim; channel++ {
			for kt := 0; kt < ps0; kt++ {
				for kh := 0; kh < ps1; kh++ {
					for kw := 0; kw < ps2; kw++ {
						source := (((channel*frames+ft*ps0+kt)*height + hy*ps1 + kh) * width) + wx*ps2 + kw
						row[((channel*ps0+kt)*ps1+kh)*ps2+kw] = sample[source]
					}
				}
			}
		}
	}
	return out, nil
}

// UnpatchifyLatent: token-major head patches -> channel-major latent.
func (p *DenoiserProgram) UnpatchifyLatent(patches []float32) ([]float32, error) {
	c, g := p.Config, p.Geometry
	patchOut := c.patchOut()
	if len(patches) != g.Seq*patchOut {
		return nil, fmt.Errorf("unpatchify: patches have %d elements, need %d", len(patches), g.Seq*patchOut)
	}
	ps0, ps1, ps2 := c.PatchSize[0], c.PatchSize[1], c.PatchSize[2]
	height, width := g.LatentHeight, g.LatentWidth
	outputFrames := g.Grid[0] * ps0
	out := make([]float32, c.OutDim*outputFrames*height*width)
	for token := 0; token < g.Seq; token++ {
		vector := patches[token*patchOut : (token+1)*patchOut]
		frame := token / (g.Grid[1] * g.Grid[2])
		remainder := token % (g.Grid[1] * g.Grid[2])
		row, column := remainder/g.Grid[2], remainder%g.Grid[2]
		for kt := 0; kt < ps0; kt++ {
			for kh := 0; kh < ps1; kh++ {
				for kw := 0; kw < ps2; kw++ {
					for channel := 0; channel < c.OutDim; channel++ {
						source := ((kt*ps1+kh)*ps2+kw)*c.OutDim + channel
						destination := (((channel*outputFrames+frame*ps0+kt)*height + row*ps1 + kh) * width) + column*ps2 + kw
						out[destination] = vector[source]
					}
				}
			}
		}
	}
	return out, nil
}

// DenoiseRequest: one guided sampling run. InitialSample seeds the
// trajectory directly (e.g. a captured noise tensor); when nil the Philox
// noise plan generates it.
type DenoiseRequest struct {
	Steps         int
	Shift         float64
	GuideScale    float64
	CondContext   []float32
	UncondContext []float32
	InitialSample []float32
	Noise         NoisePlan
	TraceSteps    bool
	// StepHook: optional per-step progress observer (heartbeat for long runs).
	StepHook func(step int, timestep int64)
}

// DenoiseStepTrace: per-step sampler boundary tensors.
type DenoiseStepTrace struct {
	Timestep                           int64
	SampleIn, CondOutput, UncondOutput []float32
	ModelOutput, SampleOut             []float32
}

// DenoiseResult: final latent plus optional per-step traces.
type DenoiseResult struct {
	Latent    []float32
	Timesteps []int64
	Sigmas    []float32
	Steps     []DenoiseStepTrace
}

// DenoiseBackend: one execution engine for the guided step loop. Branch
// contexts are opaque engine-resident cross-attention projections;
// ForwardHead returns the head patches [seq*patchOut] for one branch.
type DenoiseBackend interface {
	ProjectBranchContext(context []float32) (any, error)
	ForwardHead(patchTokens, blockE, headE []float32, branchContext any) ([]float32, error)
}

// graphRunnerBackend: the host-feed adapter; reference.Execute or any
// GraphRunner closure satisfies the loop through it.
type graphRunnerBackend struct {
	program *DenoiserProgram
	run     GraphRunner
}

func (b graphRunnerBackend) ProjectBranchContext(context []float32) (any, error) {
	return b.program.ProjectContext(b.run, context)
}

func (b graphRunnerBackend) ForwardHead(patchTokens, blockE, headE []float32, branchContext any) ([]float32, error) {
	projection, ok := branchContext.(ContextProjection)
	if !ok {
		return nil, fmt.Errorf("denoise forward: branch context is %T, need ContextProjection", branchContext)
	}
	values, err := b.program.Forward(b.run, patchTokens, blockE, headE, projection, []*tensor.Tensor{b.program.Head})
	if err != nil {
		return nil, err
	}
	return values[b.program.Head].Data, nil
}

// Denoise: the full guided UniPC trajectory through the step graph.
func (p *DenoiserProgram) Denoise(run GraphRunner, request DenoiseRequest) (DenoiseResult, error) {
	return p.DenoiseWithBackend(graphRunnerBackend{program: p, run: run}, request)
}

// DenoiseWithBackend: the guided UniPC trajectory over one execution engine.
func (p *DenoiserProgram) DenoiseWithBackend(backend DenoiseBackend, request DenoiseRequest) (DenoiseResult, error) {
	var result DenoiseResult
	elements := p.Geometry.Elements()
	timesteps, sigmas, err := UniPCSchedule(p.Config.Policy.NumTrainTimesteps, request.Steps, request.Shift)
	if err != nil {
		return result, err
	}
	result.Timesteps, result.Sigmas = timesteps, sigmas
	sampler, err := NewUniPCSampler(sigmas, elements)
	if err != nil {
		return result, err
	}
	sample := make([]float32, elements)
	if request.InitialSample != nil {
		if len(request.InitialSample) != elements {
			return result, fmt.Errorf("denoise: initial sample has %d elements, need %d", len(request.InitialSample), elements)
		}
		copy(sample, request.InitialSample)
	} else if err := FillNormalNoise(sample, request.Noise); err != nil {
		return result, err
	}
	condContext, err := backend.ProjectBranchContext(request.CondContext)
	if err != nil {
		return result, err
	}
	uncondContext, err := backend.ProjectBranchContext(request.UncondContext)
	if err != nil {
		return result, err
	}
	next := make([]float32, elements)
	guided := make([]float32, elements)
	for i, timestep := range timesteps {
		headE, blockE, err := CompileTimestepConditioning([]float64{float64(timestep)}, p.timestepWeights)
		if err != nil {
			return result, err
		}
		patchTokens, err := p.PatchifyLatent(sample)
		if err != nil {
			return result, err
		}
		branch := func(context any) ([]float32, error) {
			patches, err := backend.ForwardHead(patchTokens, blockE, headE, context)
			if err != nil {
				return nil, err
			}
			return p.UnpatchifyLatent(patches)
		}
		condOut, err := branch(condContext)
		if err != nil {
			return result, fmt.Errorf("denoise step %d conditional branch: %w", i, err)
		}
		uncondOut, err := branch(uncondContext)
		if err != nil {
			return result, fmt.Errorf("denoise step %d unconditional branch: %w", i, err)
		}
		if err := GuideInto(guided, condOut, uncondOut, request.GuideScale); err != nil {
			return result, err
		}
		var trace DenoiseStepTrace
		if request.TraceSteps {
			trace = DenoiseStepTrace{
				Timestep:   timestep,
				SampleIn:   append([]float32(nil), sample...),
				CondOutput: condOut, UncondOutput: uncondOut,
				ModelOutput: append([]float32(nil), guided...),
			}
		}
		if err := sampler.Step(next, sample, guided); err != nil {
			return result, err
		}
		sample, next = next, sample
		if request.TraceSteps {
			trace.SampleOut = append([]float32(nil), sample...)
			result.Steps = append(result.Steps, trace)
		}
		if request.StepHook != nil {
			request.StepHook(i, timestep)
		}
	}
	result.Latent = sample
	return result, nil
}
