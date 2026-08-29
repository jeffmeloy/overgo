package routedlm

// Flow terminal: the generation-lane terminal of the routed LM (the vocab
// terminal's sibling, see terminal.go). Ported from adaptive_new go/extmodel
// causal_gqa_media_cuda_windows.go (host math of the CUDA kernels:
// media_patch_gather, media_gelu_exact, media_patch_rope_2d,
// media_dense_gather, runtime_silu_bf16, runtime_bias_bf16,
// runtime_add_repeated_bf16, latent_flow_velocity), models.go
// (causalGQAMediaConfig, ImagePlan, compileShiftedFlowSchedule) and media.go
// (sinusoidalEmbeddingRows); cross-checked against the checkpoint's reference
// modeling_fm_modules.py / modeling_neo_vit.py / modeling_neo_chat.py.

import (
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
	"overgo/internal/tensor/dtype"
)

// FlowProfile binds checkpoint names and embedding policy.
type FlowProfile struct {
	ID                       artifact.ID `json:"-"`
	Version                  uint16      `json:"version"`
	TimestepEmbedderPrefix   string      `json:"timestep_embedder_prefix"`
	NoiseScaleEmbedderPrefix string      `json:"noise_scale_embedder_prefix"`
	FlowHeadPrefix           string      `json:"flow_head_prefix"`
	SourceVisionPrefix       string      `json:"source_vision_prefix"`
	GenerationVisionPrefix   string      `json:"generation_vision_prefix"`
	SinusoidalPeriod         float64     `json:"sinusoidal_period"`
}

func (profile FlowProfile) validate() error {
	if profile.Version != artifact.InitialDocumentVersion || profile.TimestepEmbedderPrefix == "" ||
		profile.NoiseScaleEmbedderPrefix == "" || profile.FlowHeadPrefix == "" || profile.SourceVisionPrefix == "" ||
		profile.GenerationVisionPrefix == "" || !checked.PositiveFinite64(profile.SinusoidalPeriod) {
		return fmt.Errorf("routed lm flow profile: invalid contract")
	}
	return nil
}

// FlowVisionConfig: vision_config section of config.json.
type FlowVisionConfig struct {
	HiddenSize      int     `json:"hidden_size"`
	LLMHiddenSize   int     `json:"llm_hidden_size"`
	NumChannels     int     `json:"num_channels"`
	PatchSize       int     `json:"patch_size"`
	DownsampleRatio float64 `json:"downsample_ratio"`
	RopeTheta       float64 `json:"rope_theta_vision"`
}

// FlowConfig: top-level flow/media fields of config.json (the section above
// llm_config; port of adaptive causalGQAMediaConfig).
type FlowConfig struct {
	DownsampleRatio           float64          `json:"downsample_ratio"`
	PatchSize                 int              `json:"patch_size"`
	TimestepShift             float64          `json:"timestep_shift"`
	TimeShiftType             string           `json:"time_shift_type"`
	NoiseScaleMode            string           `json:"noise_scale_mode"`
	NoiseScaleBaseImageSeqLen int              `json:"noise_scale_base_image_seq_len"`
	NoiseScaleMaxValue        float64          `json:"noise_scale_max_value"`
	NoiseScale                float64          `json:"noise_scale"`
	TEps                      float64          `json:"t_eps"`
	FMHeadLayers              int              `json:"fm_head_layers"`
	Vision                    FlowVisionConfig `json:"vision_config"`
}

// LoadFlowConfig: flow fields from config.json (flat top level).
func LoadFlowConfig(modelDir string) (FlowConfig, error) {
	var cfg FlowConfig
	if err := jsonfile.Decode(filepath.Join(modelDir, "config.json"), &cfg); err != nil {
		return FlowConfig{}, fmt.Errorf("routed lm flow config: %w", err)
	}
	return cfg, cfg.validate()
}

// validate: flow parts of adaptive validateCausalGQAMediaConfig.
func (c FlowConfig) validate() error {
	switch {
	case c.DownsampleRatio <= 0 || c.PatchSize <= 0 || c.TEps <= 0:
		return fmt.Errorf("routed lm flow config: invalid ratio/patch/t_eps")
	case c.Vision.PatchSize != c.PatchSize || c.Vision.DownsampleRatio != c.DownsampleRatio:
		return fmt.Errorf("routed lm flow config: vision geometry disagrees with model geometry")
	case c.Vision.NumChannels <= 0 || c.Vision.HiddenSize <= 0 || c.Vision.RopeTheta <= 0:
		return fmt.Errorf("routed lm flow config: invalid vision config")
	case c.NoiseScaleBaseImageSeqLen <= 0 || c.NoiseScaleMaxValue <= 0 || c.NoiseScale <= 0:
		return fmt.Errorf("routed lm flow config: invalid noise-scale config")
	}
	return nil
}

// FlowPlan: dims derived from config + checkpoint tensor shapes.
type FlowPlan struct {
	Hidden, VisionHidden        int
	VisionChannels, VisionPatch int
	ImageMerge                  int // dense_embedding kernel (tensor-owned evidence)
	FrequencyDim                int // embedder mlp.0 input width (tensor-owned evidence)
	FlowDim                     int // fm_head output width = channels*(patch*merge)^2
	VisionRopeTheta             float64
	SinusoidalPeriod            float64
	NoiseScaleMode              string
	NoiseScaleBase              int
	NoiseScale, NoiseScaleMax   float64
	TEps                        float64
}

func tensorShape(src *safetensors.Source, name string) ([]int64, error) {
	t, ok := src.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("routed lm flow plan: missing tensor %s", name)
	}
	// safetensors stores shapes as []uint64; the plan/validation math is int64.
	shape := make([]int64, len(t.Shape))
	for i, d := range t.Shape {
		shape[i] = int64(d)
	}
	return shape, nil
}

// CompileFlowPlan: port of adaptive compileCausalGQAMediaCompilation's flow
// half. merge comes from the generation dense_embedding kernel shape and must
// agree with 1/downsample_ratio; FrequencyDim from the timestep mlp.0 width;
// FlowDim from fm_head.2 and cross-checked against channels*(patch*merge)^2.
func CompileFlowPlan(src *safetensors.Source, cfg Config, flow FlowConfig, profile FlowProfile) (FlowPlan, error) {
	if err := profile.validate(); err != nil {
		return FlowPlan{}, err
	}
	if err := flow.validate(); err != nil {
		return FlowPlan{}, err
	}
	if flow.Vision.LLMHiddenSize != cfg.HiddenSize {
		return FlowPlan{}, fmt.Errorf("routed lm flow plan: vision llm_hidden_size=%d, llm hidden=%d", flow.Vision.LLMHiddenSize, cfg.HiddenSize)
	}
	merge := int(1 / flow.DownsampleRatio)
	if merge <= 0 || float64(merge)*flow.DownsampleRatio != 1 {
		return FlowPlan{}, fmt.Errorf("routed lm flow plan: invalid config-derived merge from ratio %g", flow.DownsampleRatio)
	}
	dense, err := tensorShape(src, profile.GenerationVisionPrefix+"dense_embedding.weight")
	if err != nil {
		return FlowPlan{}, err
	}
	if len(dense) != 4 || dense[2] <= 0 || dense[2] != dense[3] || int(dense[2]) != merge {
		return FlowPlan{}, fmt.Errorf("routed lm flow plan: dense_embedding kernel %v disagrees with merge %d", dense, merge)
	}
	first, err := tensorShape(src, profile.TimestepEmbedderPrefix+"0.weight")
	if err != nil {
		return FlowPlan{}, err
	}
	if len(first) != 2 || int(first[0]) != cfg.HiddenSize || first[1] <= 0 {
		return FlowPlan{}, fmt.Errorf("routed lm flow plan: timestep mlp.0 shape %v, want [%d,*]", first, cfg.HiddenSize)
	}
	head, err := tensorShape(src, profile.FlowHeadPrefix+"2.weight")
	if err != nil {
		return FlowPlan{}, err
	}
	flowDim := flow.Vision.NumChannels * flow.PatchSize * flow.PatchSize * merge * merge
	if len(head) != 2 || int(head[0]) != flowDim || int(head[1]) != cfg.HiddenSize {
		return FlowPlan{}, fmt.Errorf("routed lm flow plan: fm_head.2 shape %v, want [%d,%d]", head, flowDim, cfg.HiddenSize)
	}
	return FlowPlan{
		Hidden: cfg.HiddenSize, VisionHidden: flow.Vision.HiddenSize,
		VisionChannels: flow.Vision.NumChannels, VisionPatch: flow.PatchSize,
		ImageMerge: merge, FrequencyDim: int(first[1]), FlowDim: flowDim,
		VisionRopeTheta: flow.Vision.RopeTheta, SinusoidalPeriod: profile.SinusoidalPeriod,
		NoiseScaleMode: flow.NoiseScaleMode, NoiseScaleBase: flow.NoiseScaleBaseImageSeqLen,
		NoiseScale: flow.NoiseScale, NoiseScaleMax: flow.NoiseScaleMaxValue, TEps: flow.TEps,
	}, nil
}

// FlowImagePlan: geometry + resolution-derived noise scale for one target
// image (port of adaptive causalGQAMediaProgram.ImagePlan).
type FlowImagePlan struct {
	Width, Height           int
	PixelPatch, TokenPatch  int
	GridWidth, GridHeight   int // pixel-patch grid
	TokenWidth, TokenHeight int
	Tokens                  int
	NoiseScale              float64
}

// ImagePlan: noise scale derives from resolution: the base config declares
// the reference token count (noise_scale_base_image_seq_len = 64, the 256x256
// training base at token patch 32 -> 8x8 tokens); the initial-noise sigma
// scales with sqrt(tokens/base) so per-token noise energy matches the base
// resolution, capped at noise_scale_max_value.
func (p FlowPlan) ImagePlan(width, height int) (FlowImagePlan, error) {
	var out FlowImagePlan
	if p.ImageMerge <= 0 || p.VisionPatch <= 0 || p.NoiseScaleBase <= 0 {
		return out, fmt.Errorf("routed lm flow image plan: plan lacks derived geometry")
	}
	tokenPatch := p.VisionPatch * p.ImageMerge
	if width <= 0 || height <= 0 || width%tokenPatch != 0 || height%tokenPatch != 0 {
		return out, fmt.Errorf("routed lm flow image plan: %dx%d must align to token patch %d", width, height, tokenPatch)
	}
	out = FlowImagePlan{Width: width, Height: height, PixelPatch: p.VisionPatch, TokenPatch: tokenPatch}
	out.GridWidth, out.GridHeight = width/out.PixelPatch, height/out.PixelPatch
	out.TokenWidth, out.TokenHeight = width/tokenPatch, height/tokenPatch
	out.Tokens = out.TokenWidth * out.TokenHeight
	scale := p.NoiseScale
	switch p.NoiseScaleMode {
	case "resolution", "dynamic", "dynamic_sqrt":
		scale *= math.Sqrt(float64(out.Tokens) / float64(p.NoiseScaleBase))
		if p.NoiseScaleMode == "dynamic_sqrt" {
			scale = math.Sqrt(scale)
		}
	}
	out.NoiseScale = min(scale, p.NoiseScaleMax)
	return out, nil
}

// NormalizedNoiseScale: the noise-scale embedder input (reference feeds
// noise_scale / noise_scale_max_value).
func (p FlowPlan) NormalizedNoiseScale(scale float64) float64 { return scale / p.NoiseScaleMax }

// ShiftedFlowTimeSchedule: ascending timestep knots t_0=0..t_steps=1 (port of
// adaptive compileShiftedFlowSchedule, shiftedFlowTimeAscending; the
// reference _apply_time_schedule "standard" path: sigma'=shift*sigma/
// (1+(shift-1)*sigma), t=1-sigma'). The config's time_shift_type
// "exponential" names the dynamic-mu form shift=exp(mu); the reference
// forces time_schedule "standard" with the caller's explicit shift, so the
// dynamic branch is inert in this release.
func ShiftedFlowTimeSchedule(steps int, shift float64) ([]float64, error) {
	if steps <= 0 || shift <= 0 || math.IsNaN(shift) || math.IsInf(shift, 0) {
		return nil, fmt.Errorf("routed lm flow schedule: steps=%d shift=%g", steps, shift)
	}
	out := make([]float64, steps+1)
	for i := range out {
		out[i] = 1 - hostmath.ShiftFlowSigma(1-float64(i)/float64(steps), shift)
	}
	return out, nil
}

// SinusoidalEmbedding: [len(values), dim] rows, layout [cos half | sin half],
// angle_i = value*scale*period^(-i/half) (port of adaptive
// sinusoidalEmbeddingRows == reference timestep_embedding with max_period).
func SinusoidalEmbedding(values []float64, dim int, period, scale float64) ([]float32, error) {
	if dim <= 0 || dim%2 != 0 {
		return nil, fmt.Errorf("routed lm sinusoidal: dimension must be positive and even, got %d", dim)
	}
	if period <= 0 || math.IsNaN(period) || math.IsInf(period, 0) || math.IsNaN(scale) || math.IsInf(scale, 0) {
		return nil, fmt.Errorf("routed lm sinusoidal: invalid period/scale %g/%g", period, scale)
	}
	out := make([]float32, len(values)*dim)
	half := dim / 2
	logPeriod := math.Log(period)
	for row, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("routed lm sinusoidal: value %d is non-finite", row)
		}
		for i := range half {
			angle := value * scale * math.Exp(-logPeriod*float64(i)/float64(half))
			out[row*dim+i], out[row*dim+half+i] = float32(math.Cos(angle)), float32(math.Sin(angle))
		}
	}
	return out, nil
}

// FlowMLPWeights: two-linear module named "<prefix>0"/"<prefix>2" (the
// nn.Sequential indices; index 1 is the activation).
type FlowMLPWeights struct {
	W0 BF16Matrix
	B0 []float32
	W2 BF16Matrix
	B2 []float32
}

// FlowTerminalWeights: scalar embedders + flow head.
type FlowTerminalWeights struct {
	Timestep, NoiseScale FlowMLPWeights // freq -> hidden -> SiLU -> hidden
	Head                 FlowMLPWeights // hidden -> hidden -> GELU -> flow
}

func loadFlowMLP(src *safetensors.Source, prefix string, in, mid, out int) (FlowMLPWeights, error) {
	var w FlowMLPWeights
	var err error
	if w.W0, err = materializeBF16(src, prefix+"0.weight", in, mid); err != nil {
		return w, err
	}
	if w.B0, err = materializeVectorF32(src, prefix+"0.bias", mid); err != nil {
		return w, err
	}
	if w.W2, err = materializeBF16(src, prefix+"2.weight", mid, out); err != nil {
		return w, err
	}
	if w.B2, err = materializeVectorF32(src, prefix+"2.bias", out); err != nil {
		return w, err
	}
	return w, nil
}

// LoadFlowTerminalWeights: fm_modules embedders + head, shape-validated
// against the plan's derived dims.
func LoadFlowTerminalWeights(src *safetensors.Source, plan FlowPlan, profile FlowProfile) (FlowTerminalWeights, error) {
	if err := profile.validate(); err != nil {
		return FlowTerminalWeights{}, err
	}
	var out FlowTerminalWeights
	var err error
	if out.Timestep, err = loadFlowMLP(src, profile.TimestepEmbedderPrefix, plan.FrequencyDim, plan.Hidden, plan.Hidden); err != nil {
		return FlowTerminalWeights{}, err
	}
	if out.NoiseScale, err = loadFlowMLP(src, profile.NoiseScaleEmbedderPrefix, plan.FrequencyDim, plan.Hidden, plan.Hidden); err != nil {
		return FlowTerminalWeights{}, err
	}
	if out.Head, err = loadFlowMLP(src, profile.FlowHeadPrefix, plan.Hidden, plan.Hidden, plan.FlowDim); err != nil {
		return FlowTerminalWeights{}, err
	}
	return out, nil
}

// VisionEmbedderWeights: conv-only embedder (patch conv + merge conv), each
// conv flattened to a [out, in*k*k] linear over gathered patch vectors.
type VisionEmbedderWeights struct {
	Patch     BF16Matrix // [visionHidden, channels*patch*patch]
	PatchBias []float32
	Dense     BF16Matrix // [hidden, visionHidden*merge*merge]
	DenseBias []float32
}

// LoadVisionEmbedderWeights: one conv embedder by tensor prefix (source or
// generation lane; both share this shape).
func LoadVisionEmbedderWeights(src *safetensors.Source, prefix string, plan FlowPlan) (VisionEmbedderWeights, error) {
	patchShape, err := tensorShape(src, prefix+"patch_embedding.weight")
	if err != nil {
		return VisionEmbedderWeights{}, err
	}
	wantPatch := []int64{int64(plan.VisionHidden), int64(plan.VisionChannels), int64(plan.VisionPatch), int64(plan.VisionPatch)}
	if len(patchShape) != 4 || patchShape[0] != wantPatch[0] || patchShape[1] != wantPatch[1] || patchShape[2] != wantPatch[2] || patchShape[3] != wantPatch[3] {
		return VisionEmbedderWeights{}, fmt.Errorf("routed lm vision embedder: %spatch_embedding.weight shape %v, want %v", prefix, patchShape, wantPatch)
	}
	denseShape, err := tensorShape(src, prefix+"dense_embedding.weight")
	if err != nil {
		return VisionEmbedderWeights{}, err
	}
	wantDense := []int64{int64(plan.Hidden), int64(plan.VisionHidden), int64(plan.ImageMerge), int64(plan.ImageMerge)}
	if len(denseShape) != 4 || denseShape[0] != wantDense[0] || denseShape[1] != wantDense[1] || denseShape[2] != wantDense[2] || denseShape[3] != wantDense[3] {
		return VisionEmbedderWeights{}, fmt.Errorf("routed lm vision embedder: %sdense_embedding.weight shape %v, want %v", prefix, denseShape, wantDense)
	}
	var w VisionEmbedderWeights
	patchIn := plan.VisionChannels * plan.VisionPatch * plan.VisionPatch
	denseIn := plan.VisionHidden * plan.ImageMerge * plan.ImageMerge
	if w.Patch, err = materializeBF16(src, prefix+"patch_embedding.weight", patchIn, plan.VisionHidden); err != nil {
		return VisionEmbedderWeights{}, err
	}
	if w.PatchBias, err = materializeVectorF32(src, prefix+"patch_embedding.bias", plan.VisionHidden); err != nil {
		return VisionEmbedderWeights{}, err
	}
	if w.Dense, err = materializeBF16(src, prefix+"dense_embedding.weight", denseIn, plan.Hidden); err != nil {
		return VisionEmbedderWeights{}, err
	}
	if w.DenseBias, err = materializeVectorF32(src, prefix+"dense_embedding.bias", plan.Hidden); err != nil {
		return VisionEmbedderWeights{}, err
	}
	return w, nil
}

// linearBiasRounded: out = bf16(f32(x*W^T) + bias) — GEMM f64-accumulate
// followed by the runtime_bias_bf16 discipline (single bf16 round at the f32
// bias add).
func linearBiasRounded(out, x []float32, w BF16Matrix, bias []float32, rows int) {
	hostmath.LinearBF16F64(out, x, w.Data, nil, rows, w.In, w.Out)
	for r := range rows {
		row := out[r*w.Out : (r+1)*w.Out]
		for c, v := range row {
			row[c] = dtype.RoundBF16(v + bias[c])
		}
	}
}

// siluRounded: x = bf16(silu(x)) with f64 sigmoid (runtime_silu_bf16).
func siluRounded(x []float32) {
	for i, value := range x {
		v := float64(value)
		x[i] = dtype.RoundBF16(float32(v / (1.0 + math.Exp(-v))))
	}
}

// geluExactRounded: x = bf16(0.5*v*(1+erf(v/sqrt2))) with f64 erf
// (media_gelu_exact; nn.GELU() exact form).
func geluExactRounded(x []float32) {
	for i, value := range x {
		v := float64(value)
		x[i] = dtype.RoundBF16(float32(0.5 * v * (1.0 + math.Erf(v*0.7071067811865475))))
	}
}

// ScalarConditionRow: one embedder forward: sinusoidal(value) -> linear0 ->
// SiLU -> linear2. The frequency vector is bf16-rounded before the GEMM (the
// device GEMM converts its f32 inputs to bf16).
func ScalarConditionRow(w FlowMLPWeights, value float64, plan FlowPlan) ([]float32, error) {
	freq, err := SinusoidalEmbedding([]float64{value}, plan.FrequencyDim, plan.SinusoidalPeriod, 1)
	if err != nil {
		return nil, err
	}
	if w.W0.In != plan.FrequencyDim || w.W2.Out != plan.Hidden {
		return nil, fmt.Errorf("routed lm scalar condition: weights are not [%d->%d]", plan.FrequencyDim, plan.Hidden)
	}
	dtype.RoundBF16Slice(freq)
	mid := make([]float32, w.W0.Out)
	linearBiasRounded(mid, freq, w.W0, w.B0, 1)
	siluRounded(mid)
	out := make([]float32, plan.Hidden)
	linearBiasRounded(out, mid, w.W2, w.B2, 1)
	return out, nil
}

// FlowConditionRow: condition = bf16(timestepEmbed + noiseEmbed)
// (runtime_add_repeated_bf16; all token rows share this row). normalizedNoise
// is NormalizedNoiseScale of the image plan's derived scale.
func FlowConditionRow(w FlowTerminalWeights, plan FlowPlan, timestep, normalizedNoise float64) ([]float32, error) {
	if timestep < 0 || timestep >= 1 {
		return nil, fmt.Errorf("routed lm flow condition: timestep %g outside [0,1)", timestep)
	}
	tEmb, err := ScalarConditionRow(w.Timestep, timestep, plan)
	if err != nil {
		return nil, err
	}
	nEmb, err := ScalarConditionRow(w.NoiseScale, normalizedNoise, plan)
	if err != nil {
		return nil, err
	}
	for i, v := range tEmb {
		tEmb[i] = dtype.RoundBF16(v + nEmb[i])
	}
	return tEmb, nil
}

// AddConditionRows: hidden[row] += condition, every row:
// out = bf16(bf16(hidden) + condition) (runtime_add_rounded_into_bf16).
func AddConditionRows(hidden, condition []float32) error {
	if len(condition) == 0 || len(hidden)%len(condition) != 0 {
		return fmt.Errorf("routed lm flow condition add: hidden len=%d not row-divisible by %d", len(hidden), len(condition))
	}
	d := len(condition)
	for i, v := range hidden {
		hidden[i] = dtype.RoundBF16(dtype.RoundBF16(v) + condition[i%d])
	}
	return nil
}

// FlowHeadVelocity: hidden rows -> pixel-patch velocity rows:
// linear0 -> exact GELU -> linear2 -> v = bf16((x - z) / max(1-t, t_eps))
// (latent_flow_velocity; the device divide is __fdividef, host reference is
// the exact f32 divide). z is the current planar state gathered in the same
// [rows, FlowDim] patch layout as the head output.
func FlowHeadVelocity(w FlowMLPWeights, plan FlowPlan, hidden, z []float32, timestep float64) ([]float32, error) {
	if timestep < 0 || timestep >= 1 {
		return nil, fmt.Errorf("routed lm flow head: timestep %g outside [0,1)", timestep)
	}
	if w.W0.In != plan.Hidden || w.W0.Out != plan.Hidden || w.W2.In != plan.Hidden || w.W2.Out != plan.FlowDim {
		return nil, fmt.Errorf("routed lm flow head: weights are not [%d->%d->%d]", plan.Hidden, plan.Hidden, plan.FlowDim)
	}
	if len(hidden) == 0 || len(hidden)%plan.Hidden != 0 {
		return nil, fmt.Errorf("routed lm flow head: hidden len=%d not divisible by %d", len(hidden), plan.Hidden)
	}
	rows := len(hidden) / plan.Hidden
	if len(z) != rows*plan.FlowDim {
		return nil, fmt.Errorf("routed lm flow head: z len=%d, want %d", len(z), rows*plan.FlowDim)
	}
	mid := make([]float32, rows*plan.Hidden)
	linearBiasRounded(mid, hidden, w.W0, w.B0, rows)
	geluExactRounded(mid)
	out := make([]float32, rows*plan.FlowDim)
	linearBiasRounded(out, mid, w.W2, w.B2, rows)
	denom := float32(max(1-timestep, plan.TEps))
	for i, v := range out {
		out[i] = dtype.RoundBF16((v - z[i]) / denom)
	}
	return out, nil
}

// GenerationFinalHidden applies the generation terminal's BF16 norm steps.
func GenerationFinalHidden(hidden, weight []float32, rows, dim int, eps float64) ([]float32, error) {
	if rows <= 0 || dim <= 0 || len(hidden) != rows*dim || len(weight) != dim || eps <= 0 {
		return nil, fmt.Errorf("routed lm generation terminal: invalid shape")
	}
	out := make([]float32, len(hidden))
	for row := range rows {
		base := row * dim
		var sum float64
		for _, value := range hidden[base : base+dim] {
			sum += float64(value) * float64(value)
		}
		inv := 1 / math.Sqrt(sum/float64(dim)+eps)
		for column := range dim {
			normalized := dtype.RoundBF16(float32(float64(hidden[base+column]) * inv))
			out[base+column] = dtype.RoundBF16(normalized * weight[column])
		}
	}
	return out, nil
}

// GuidedFlowVelocity combines ordered guidance terms, then stores BF16.
func GuidedFlowVelocity(velocities [][]float32, coefficients []float32) ([]float32, error) {
	if len(velocities) == 0 || len(velocities) != len(coefficients) || len(velocities[0]) == 0 {
		return nil, fmt.Errorf("routed lm guidance: invalid terms")
	}
	out := make([]float32, len(velocities[0]))
	for term, velocity := range velocities {
		if len(velocity) != len(out) {
			return nil, fmt.Errorf("routed lm guidance: term %d elements=%d want=%d", term, len(velocity), len(out))
		}
		for index, value := range velocity {
			out[index] += coefficients[term] * value
		}
	}
	dtype.RoundBF16Slice(out)
	return out, nil
}

// FlowEulerStep advances packed state with BF16 storage.
func FlowEulerStep(state, velocity []float32, delta float32) error {
	if len(state) == 0 || len(state) != len(velocity) || delta <= 0 {
		return fmt.Errorf("routed lm Euler: invalid state or delta")
	}
	for index, value := range velocity {
		state[index] = dtype.RoundBF16(state[index] + delta*value)
	}
	return nil
}

// patchVector: one pixel patch gathered from a planar [C,H,W] image into the
// conv-as-linear input layout [c][r][q], bf16-rounded (media_patch_gather +
// the device GEMM's f32->bf16 input conversion).
func patchVector(dst, image []float32, patch int, shape FlowImagePlan, channels int) {
	p := shape.PixelPatch
	py, px := patch/shape.GridWidth, patch%shape.GridWidth
	i := 0
	for c := range channels {
		for r := range p {
			base := (c*shape.Height+py*p+r)*shape.Width + px*p
			for q := range p {
				dst[i] = dtype.RoundBF16(image[base+q])
				i++
			}
		}
	}
}

// ropePatch2D: in-place 2-D rotary embedding over one patch row (port of
// media_patch_rope_2d): first half rotates with the x position, second half
// with y; adjacent pairs (d, d+1); f32 products/sums, single bf16 round.
func ropePatch2D(row []float32, y, x int, theta float64) {
	dim := len(row)
	half := dim / 2
	for part := range 2 {
		pos := float64(x)
		if part == 1 {
			pos = float64(y)
		}
		for local := 0; local+1 < half; local += 2 {
			d := part*half + local
			pair := local / 2
			angle := pos * math.Pow(theta, -2.0*float64(pair)/float64(half))
			c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
			a, b := row[d], row[d+1]
			row[d] = dtype.RoundBF16(a*c - b*s)
			row[d+1] = dtype.RoundBF16(a*s + b*c)
		}
	}
}

// denseVector: one token's merge-block gathered from patch embeddings into
// the merge-conv input layout [c][dy][dx] (media_dense_gather). Patch
// embeddings are already bf16.
func denseVector(dst, patches []float32, token int, shape FlowImagePlan, visionHidden int) {
	merge := shape.TokenPatch / shape.PixelPatch
	ty, tx := token/shape.TokenWidth, token%shape.TokenWidth
	i := 0
	for c := range visionHidden {
		for dy := range merge {
			for dx := range merge {
				dst[i] = patches[((ty*merge+dy)*shape.GridWidth+tx*merge+dx)*visionHidden+c]
				i++
			}
		}
	}
}

// VisionEmbedTokens: conv-only vision embedder over a planar [C,H,W] f32
// image: patch conv (+bias) -> exact GELU -> 2-D rope -> merge conv (+bias)
// -> [Tokens, Hidden] rows (projectMediaTokenLayoutDeviceCUDA host math).
func VisionEmbedTokens(w VisionEmbedderWeights, plan FlowPlan, image []float32, width, height int) ([]float32, FlowImagePlan, error) {
	shape, err := plan.ImagePlan(width, height)
	if err != nil {
		return nil, shape, err
	}
	if len(image) != plan.VisionChannels*width*height {
		return nil, shape, fmt.Errorf("routed lm vision embed: image len=%d, want %d", len(image), plan.VisionChannels*width*height)
	}
	patchIn := plan.VisionChannels * plan.VisionPatch * plan.VisionPatch
	patchRows := shape.GridHeight * shape.GridWidth
	patches := make([]float32, patchRows*plan.VisionHidden)
	hostmath.ParallelRangeF64(patchRows, patchIn*plan.VisionHidden, func(lo, hi int) {
		gathered := make([]float32, patchIn)
		for patch := lo; patch < hi; patch++ {
			patchVector(gathered, image, patch, shape, plan.VisionChannels)
			row := patches[patch*plan.VisionHidden : (patch+1)*plan.VisionHidden]
			linearBiasRounded(row, gathered, w.Patch, w.PatchBias, 1)
			geluExactRounded(row)
			ropePatch2D(row, patch/shape.GridWidth, patch%shape.GridWidth, plan.VisionRopeTheta)
		}
	})
	denseIn := plan.VisionHidden * plan.ImageMerge * plan.ImageMerge
	tokens := make([]float32, shape.Tokens*plan.Hidden)
	hostmath.ParallelRangeF64(shape.Tokens, denseIn*plan.Hidden, func(lo, hi int) {
		gathered := make([]float32, denseIn)
		for token := lo; token < hi; token++ {
			denseVector(gathered, patches, token, shape, plan.VisionHidden)
			linearBiasRounded(tokens[token*plan.Hidden:(token+1)*plan.Hidden], gathered, w.Dense, w.DenseBias, 1)
		}
	})
	return tokens, shape, nil
}
