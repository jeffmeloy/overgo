// Package speechsynth owns the speech-synthesis capability (ladder rung 7):
// the flow-LM conditioner transformer, the conditional flow head, the
// frame-latent generation loop, and the mimi codec DECODER (latent -> PCM),
// judged against the imported pockettts golden ladder (fixtures/pockettts).
//
// The mimi ENCODER (PCM -> voice latent) is not ported: the decode path
// never calls it and the artifact's voice-cloning encoder is amputated;
// voice conditioning enters from the golden ladder (codec.go).
//
// Architecture (all ported behavior, written fresh against overgo owners):
// pre-norm causal transformer — LayerNorm(norm1) -> fused qkv -> interleaved
// RoPE -> softmax(1/sqrt(hd)) attention -> out proj -> residual; then
// LayerNorm(norm2) -> linear -> exact-erf GELU -> linear -> residual. Text
// tokens embed by table lookup; frame latents enter through input_linear;
// out_norm LayerNorm feeds both the EOS head and the flow head.
//
// Every geometric dimension is DERIVED from the artifact's tensor shapes;
// config supplies only what shapes cannot say (num_heads hides inside the
// fused in_proj; max_period is the RoPE base, present in no tensor). Shape
// facts the config restates are cross-checked, never trusted alone.
package speechsynth

import (
	_ "embed"
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/checked"
	"overgo/internal/hostmath"
	"overgo/internal/jsonfile"
	"overgo/internal/media"
	"overgo/internal/safetensors"
	"overgo/internal/strictjson"
	"overgo/internal/tensor"
	"overgo/internal/tensorcatalog"
)

// Vendor-source epsilons: not config fields, not tensors — code facts of the
// reference implementation (adaptive reference facts speech-flow-synthesis).
//
//go:embed speech_program.json
var speechProgramJSON []byte

func loadSpeechProgram() (media.NormalizationProgram, error) {
	var program media.NormalizationProgram
	if err := strictjson.DecodeBytes(speechProgramJSON, &program); err != nil {
		return program, fmt.Errorf("speechsynth: decode normalization program: %w", err)
	}
	if err := program.Validate(); err != nil {
		return program, err
	}
	return program, nil
}

// Dims: model geometry, derived from tensor shapes plus the two
// config-only facts (Heads, MaxPeriod).
type Dims struct {
	DModel    int // out_norm.weight width
	Heads     int // config: fused in_proj hides the head partition
	HeadDim   int // DModel / Heads
	FF        int // linear1.weight rows
	Layers    int // contiguous transformer.layers.N
	TextVocab int // conditioner.embed.weight rows (n_bins + 1)
	LatentDim int // input_linear.weight columns (frame latent width)
	FlowDim   int // flow_net.input_proj.weight rows
	FlowDepth int // contiguous flow_net.res_blocks.N
	TimeFreqs int // flow_net.time_embed.0.freqs length
	MaxPeriod float64
}

// attnLayer: one backbone transformer layer's weight views.
type attnLayer struct {
	norm1W, norm1B []float32 // [d]
	norm2W, norm2B []float32 // [d]
	inProj         []float32 // [3d, d] fused q,k,v
	outProj        []float32 // [d, d]
	lin1           []float32 // [ff, d]
	lin2           []float32 // [d, ff]
}

// timeEmbed: one flow timestep embedder (cos/sin features -> MLP -> the
// alpha-scaled unbiased-variance RMS quirk).
type timeEmbed struct {
	freqs    []float32 // [half]
	l0w, l0b []float32 // [flowDim, 2*half], [flowDim]
	l2w, l2b []float32 // [flowDim, flowDim], [flowDim]
	alpha    []float32 // [flowDim]
}

// flowBlock: one adaLN-modulated residual MLP block.
type flowBlock struct {
	inLnW, inLnB []float32 // [flowDim]
	mlp0w, mlp0b []float32 // [flowDim, flowDim], [flowDim]
	mlp2w, mlp2b []float32 // [flowDim, flowDim], [flowDim]
	adaW, adaB   []float32 // [3*flowDim, flowDim], [3*flowDim]
}

// flowNet: the conditional flow head F(cond, s, t, x).
type flowNet struct {
	inputProjW, inputProjB []float32 // latent -> flowDim
	condEmbedW, condEmbedB []float32 // d -> flowDim
	timeEmbeds             [tensor.PairedExtent]timeEmbed
	blocks                 []flowBlock
	finalAdaW, finalAdaB   []float32 // flowDim -> 2*flowDim
	finalLinW, finalLinB   []float32 // flowDim -> latent
}

// Model: loaded backbone weights and derived dims.
type Model struct {
	Dims          Dims
	Normalization media.NormalizationProgram

	layers      []attnLayer
	condEmbed   []float32 // [textVocab, d] token lookup
	inputLinear []float32 // [d, latent], no bias
	outNormW    []float32 // [d]
	outNormB    []float32 // [d]
	outEosW     []float32 // [1, d]
	outEosB     float32
	BosEmb      []float32 // [latent]
	EmbMean     []float32 // [latent] (codec-bridge stats; verified at load)
	EmbStd      []float32 // [latent]

	flow flowNet

	// Codec: the mimi decoder half; LatentsToPCM refuses while nil.
	Codec *CodecDecoder

	invFreq    []float64 // interleaved-RoPE ladder from MaxPeriod, HeadDim
	scoreScale float32   // 1/sqrt(hd), folded into q before attention
}

// artifactConfig mirrors the vendor-resolved config JSON; only the fields
// this slice consumes.
type artifactConfig struct {
	FlowLM struct {
		Flow struct {
			Depth int `json:"depth"`
			Dim   int `json:"dim"`
		} `json:"flow"`
		LookupTable struct {
			Dim   int `json:"dim"`
			NBins int `json:"n_bins"`
		} `json:"lookup_table"`
		Transformer struct {
			DModel      int     `json:"d_model"`
			HiddenScale int     `json:"hidden_scale"`
			MaxPeriod   float64 `json:"max_period"`
			NumHeads    int     `json:"num_heads"`
			NumLayers   int     `json:"num_layers"`
		} `json:"transformer"`
	} `json:"flow_lm"`
	Mimi mimiConfig `json:"mimi"`
}

// mimiConfig: the codec section of the vendor-resolved config.
type mimiConfig struct {
	FrameRate  float64 `json:"frame_rate"`
	SampleRate int     `json:"sample_rate"`
	OuterDim   int     `json:"outer_dim"`
	Quantizer  struct {
		Dimension       int `json:"dimension"`
		OutputDimension int `json:"output_dimension"`
	} `json:"quantizer"`
	Seanet      seanetConfig `json:"seanet"`
	Transformer struct {
		Context        int     `json:"context"`
		DModel         int     `json:"d_model"`
		DimFeedforward int     `json:"dim_feedforward"`
		MaxPeriod      float64 `json:"max_period"`
		NumHeads       int     `json:"num_heads"`
		NumLayers      int     `json:"num_layers"`
	} `json:"transformer"`
}

// seanetConfig: SEANet layout facts (widths, kernels, strides) — pure
// geometry the tensors cannot fully self-describe (stage count and order).
type seanetConfig struct {
	Channels           int   `json:"channels"`
	Compress           int   `json:"compress"`
	Dimension          int   `json:"dimension"`
	KernelSize         int   `json:"kernel_size"`
	LastKernelSize     int   `json:"last_kernel_size"`
	NFilters           int   `json:"n_filters"`
	Ratios             []int `json:"ratios"`
	ResidualKernelSize int   `json:"residual_kernel_size"`
}

// configFileName: the vendor-resolved serving config beside the weights.
const configFileName = "pockettts_config.json"

func loadConfig(path string) (artifactConfig, error) {
	var config artifactConfig
	if err := jsonfile.Decode(path, &config); err != nil {
		return artifactConfig{}, fmt.Errorf("speechsynth: parse %s: %w", filepath.Base(path), err)
	}
	tr := config.FlowLM.Transformer
	if tr.NumHeads <= tensor.FirstOffset {
		return artifactConfig{}, fmt.Errorf("speechsynth: config lacks positive flow_lm.transformer.num_heads")
	}
	if tr.MaxPeriod <= tensor.FirstOffset {
		return artifactConfig{}, fmt.Errorf("speechsynth: config lacks positive flow_lm.transformer.max_period")
	}
	return config, nil
}

// Checkpoint tensor-name wiring (vendor layout; names are artifact facts).
const (
	layerPrefix    = "flow_lm.transformer.layers."
	resBlockPrefix = "flow_lm.flow_net.res_blocks."
)

// Load opens the safetensors artifact, derives dims from tensor shapes,
// cross-checks the config's restated shape facts, and materializes every
// backbone tensor as f32 (BF16 checkpoint decoded once at load).
func Load(directory string) (*Model, error) {
	config, err := loadConfig(filepath.Join(directory, configFileName))
	if err != nil {
		return nil, err
	}
	normalization, err := loadSpeechProgram()
	if err != nil {
		return nil, err
	}
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		return nil, err
	}
	defer source.Close()

	shapes, err := source.IntShapes()
	if err != nil {
		return nil, fmt.Errorf("speechsynth: inventory: %w", err)
	}

	read := func(name string, wantShape ...int) ([]float32, error) {
		tensor, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("speechsynth: tensor %q missing", name)
		}
		dimensions, err := tensorcatalog.Dimensions(shapes, name, len(wantShape))
		if err != nil {
			return nil, fmt.Errorf("speechsynth: %w", err)
		}
		for i, want := range wantShape {
			if checked.PositiveInts(want) && !checked.Equal(dimensions[i], want) {
				return nil, fmt.Errorf("speechsynth: tensor %q shape %v, want dim %d = %d", name, dimensions, i, want)
			}
		}
		values, err := safetensors.ReadF32(tensor)
		if err != nil {
			return nil, fmt.Errorf("speechsynth: tensor %q payload: %w", name, err)
		}
		return values, nil
	}

	dims, err := deriveDims(shapes, config)
	if err != nil {
		return nil, err
	}
	m := &Model{
		Dims:          dims,
		Normalization: normalization,
		invFreq:       hostmath.RopeInvFreq(dims.MaxPeriod, dims.HeadDim),
		scoreScale:    float32(float64(tensor.SingletonExtent) / math.Sqrt(float64(dims.HeadDim))),
	}

	d, ff, latent, flowDim := dims.DModel, dims.FF, dims.LatentDim, dims.FlowDim
	m.layers = make([]attnLayer, dims.Layers)
	for i := range m.layers {
		p := fmt.Sprintf("%s%d.", layerPrefix, i)
		l := &m.layers[i]
		for _, f := range []struct {
			dst  *[]float32
			name string
			want []int
		}{
			{&l.norm1W, p + "norm1.weight", []int{d}},
			{&l.norm1B, p + "norm1.bias", []int{d}},
			{&l.norm2W, p + "norm2.weight", []int{d}},
			{&l.norm2B, p + "norm2.bias", []int{d}},
			{&l.inProj, p + "self_attn.in_proj.weight", []int{tensor.TripleExtent * d, d}},
			{&l.outProj, p + "self_attn.out_proj.weight", []int{d, d}},
			{&l.lin1, p + "linear1.weight", []int{ff, d}},
			{&l.lin2, p + "linear2.weight", []int{d, ff}},
		} {
			if *f.dst, err = read(f.name, f.want...); err != nil {
				return nil, err
			}
		}
	}
	for _, f := range []struct {
		dst  *[]float32
		name string
		want []int
	}{
		{&m.condEmbed, "flow_lm.conditioner.embed.weight", []int{dims.TextVocab, d}},
		{&m.inputLinear, "flow_lm.input_linear.weight", []int{d, latent}},
		{&m.outNormW, "flow_lm.out_norm.weight", []int{d}},
		{&m.outNormB, "flow_lm.out_norm.bias", []int{d}},
		{&m.outEosW, "flow_lm.out_eos.weight", []int{tensor.SingletonExtent, d}},
		{&m.BosEmb, "flow_lm.bos_emb", []int{latent}},
		{&m.EmbMean, "flow_lm.emb_mean", []int{latent}},
		{&m.EmbStd, "flow_lm.emb_std", []int{latent}},
	} {
		if *f.dst, err = read(f.name, f.want...); err != nil {
			return nil, err
		}
	}
	eosBias, err := read("flow_lm.out_eos.bias", tensor.SingletonExtent)
	if err != nil {
		return nil, err
	}
	m.outEosB = eosBias[tensor.FirstOffset]

	fn := &m.flow
	for _, f := range []struct {
		dst  *[]float32
		name string
		want []int
	}{
		{&fn.inputProjW, "flow_lm.flow_net.input_proj.weight", []int{flowDim, latent}},
		{&fn.inputProjB, "flow_lm.flow_net.input_proj.bias", []int{flowDim}},
		{&fn.condEmbedW, "flow_lm.flow_net.cond_embed.weight", []int{flowDim, d}},
		{&fn.condEmbedB, "flow_lm.flow_net.cond_embed.bias", []int{flowDim}},
		{&fn.finalAdaW, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight", []int{tensor.PairedExtent * flowDim, flowDim}},
		{&fn.finalAdaB, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias", []int{tensor.PairedExtent * flowDim}},
		{&fn.finalLinW, "flow_lm.flow_net.final_layer.linear.weight", []int{latent, flowDim}},
		{&fn.finalLinB, "flow_lm.flow_net.final_layer.linear.bias", []int{latent}},
	} {
		if *f.dst, err = read(f.name, f.want...); err != nil {
			return nil, err
		}
	}
	for i := range fn.timeEmbeds {
		p := fmt.Sprintf("flow_lm.flow_net.time_embed.%d.", i)
		te := &fn.timeEmbeds[i]
		for _, f := range []struct {
			dst  *[]float32
			name string
			want []int
		}{
			{&te.freqs, p + "freqs", []int{dims.TimeFreqs}},
			{&te.l0w, p + "mlp.0.weight", []int{flowDim, tensor.PairedExtent * dims.TimeFreqs}},
			{&te.l0b, p + "mlp.0.bias", []int{flowDim}},
			{&te.l2w, p + "mlp.2.weight", []int{flowDim, flowDim}},
			{&te.l2b, p + "mlp.2.bias", []int{flowDim}},
			{&te.alpha, p + "mlp.3.alpha", []int{flowDim}},
		} {
			if *f.dst, err = read(f.name, f.want...); err != nil {
				return nil, err
			}
		}
	}
	fn.blocks = make([]flowBlock, dims.FlowDepth)
	for i := range fn.blocks {
		p := fmt.Sprintf("%s%d.", resBlockPrefix, i)
		b := &fn.blocks[i]
		for _, f := range []struct {
			dst  *[]float32
			name string
			want []int
		}{
			{&b.inLnW, p + "in_ln.weight", []int{flowDim}},
			{&b.inLnB, p + "in_ln.bias", []int{flowDim}},
			{&b.mlp0w, p + "mlp.0.weight", []int{flowDim, flowDim}},
			{&b.mlp0b, p + "mlp.0.bias", []int{flowDim}},
			{&b.mlp2w, p + "mlp.2.weight", []int{flowDim, flowDim}},
			{&b.mlp2b, p + "mlp.2.bias", []int{flowDim}},
			{&b.adaW, p + "adaLN_modulation.1.weight", []int{tensor.TripleExtent * flowDim, flowDim}},
			{&b.adaB, p + "adaLN_modulation.1.bias", []int{tensor.TripleExtent * flowDim}},
		} {
			if *f.dst, err = read(f.name, f.want...); err != nil {
				return nil, err
			}
		}
	}
	if m.Codec, err = loadCodecDecoder(read, shapes, config, dims.LatentDim, normalization); err != nil {
		return nil, err
	}
	return m, nil
}

// deriveDims: every geometric fact from tensor shapes; config supplies only
// num_heads and max_period, and its restated shape facts must agree.
func deriveDims(shapes map[string][]int, config artifactConfig) (Dims, error) {
	var d Dims
	outNorm, err := tensorcatalog.Dimensions(shapes, "flow_lm.out_norm.weight", tensor.SingletonExtent)
	if err != nil {
		return d, fmt.Errorf("speechsynth: out_norm.weight missing; d_model underivable")
	}
	d.DModel = outNorm[tensor.FirstOffset]

	if d.Layers, err = tensorcatalog.IndexedCount(shapes, layerPrefix, ".norm1.weight"); err != nil {
		return d, err
	}
	lin1, err := tensorcatalog.Dimensions(shapes, layerPrefix+"0.linear1.weight", tensor.PairedExtent)
	if err != nil || !checked.Equal(lin1[tensor.SingletonExtent], d.DModel) {
		return d, fmt.Errorf("speechsynth: layer-0 linear1 incompatible with d_model %d", d.DModel)
	}
	d.FF = lin1[tensor.FirstOffset]
	inProj, err := tensorcatalog.Dimensions(shapes, layerPrefix+"0.self_attn.in_proj.weight", tensor.PairedExtent)
	if err != nil || !checked.Equal(inProj[tensor.FirstOffset], tensor.TripleExtent*d.DModel) || !checked.Equal(inProj[tensor.SingletonExtent], d.DModel) {
		return d, fmt.Errorf("speechsynth: layer-0 in_proj %v is not fused [3d, d]", inProj)
	}

	embed, err := tensorcatalog.Dimensions(shapes, "flow_lm.conditioner.embed.weight", tensor.PairedExtent)
	if err != nil || !checked.Equal(embed[tensor.SingletonExtent], d.DModel) {
		return d, fmt.Errorf("speechsynth: conditioner embed missing or width != d_model")
	}
	d.TextVocab = embed[tensor.FirstOffset]

	inputLinear, err := tensorcatalog.Dimensions(shapes, "flow_lm.input_linear.weight", tensor.PairedExtent)
	if err != nil || !checked.Equal(inputLinear[tensor.FirstOffset], d.DModel) {
		return d, fmt.Errorf("speechsynth: input_linear missing or rows != d_model")
	}
	d.LatentDim = inputLinear[tensor.SingletonExtent]

	inputProj, err := tensorcatalog.Dimensions(shapes, "flow_lm.flow_net.input_proj.weight", tensor.PairedExtent)
	if err != nil || !checked.Equal(inputProj[tensor.SingletonExtent], d.LatentDim) {
		return d, fmt.Errorf("speechsynth: flow input_proj missing or columns != latent dim %d", d.LatentDim)
	}
	d.FlowDim = inputProj[tensor.FirstOffset]

	if d.FlowDepth, err = tensorcatalog.IndexedCount(shapes, resBlockPrefix, ".in_ln.weight"); err != nil {
		return d, err
	}
	freqs, err := tensorcatalog.Dimensions(shapes, "flow_lm.flow_net.time_embed.0.freqs", tensor.SingletonExtent)
	if err != nil {
		return d, fmt.Errorf("speechsynth: time_embed.0.freqs missing")
	}
	d.TimeFreqs = freqs[tensor.FirstOffset]

	tr := config.FlowLM.Transformer
	d.Heads = tr.NumHeads
	d.MaxPeriod = tr.MaxPeriod
	if _, ok := checked.DivExactInt(d.DModel, d.Heads); !ok {
		return d, fmt.Errorf("speechsynth: heads %d do not partition d_model %d", d.Heads, d.DModel)
	}
	d.HeadDim = d.DModel / d.Heads
	if !checked.EvenInt(d.HeadDim) {
		return d, fmt.Errorf("speechsynth: head dim %d incompatible with paired RoPE", d.HeadDim)
	}

	// Config restatements of shape facts: agree or refuse (0 = unstated).
	nBinsPlusOne := tensor.FirstOffset
	if checked.Nonzero(config.FlowLM.LookupTable.NBins) {
		nBinsPlusOne = config.FlowLM.LookupTable.NBins + tensor.SingletonExtent
	}
	for _, check := range []struct {
		name             string
		derived, claimed int
	}{
		{"d_model", d.DModel, tr.DModel},
		{"num_layers", d.Layers, tr.NumLayers},
		{"hidden_scale*d_model", d.FF, tr.HiddenScale * tr.DModel},
		{"lookup_table.dim", d.DModel, config.FlowLM.LookupTable.Dim},
		{"lookup_table.n_bins+1", d.TextVocab, nBinsPlusOne},
		{"flow.dim", d.FlowDim, config.FlowLM.Flow.Dim},
		{"flow.depth", d.FlowDepth, config.FlowLM.Flow.Depth},
	} {
		if checked.Nonzero(check.claimed) && !checked.Equal(check.claimed, check.derived) {
			return d, fmt.Errorf("speechsynth: config %s=%d contradicts derived %d", check.name, check.claimed, check.derived)
		}
	}
	return d, nil
}
