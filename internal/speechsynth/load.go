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
	"fmt"
	"math"
	"path/filepath"

	"overgo/internal/hostmath"
	"overgo/internal/jsonfile"
	"overgo/internal/safetensors"
	"overgo/internal/tensorcatalog"
)

// Vendor-source epsilons: not config fields, not tensors — code facts of the
// reference implementation (adaptive reference facts speech-flow-synthesis).
const (
	// torch.nn.LayerNorm default eps; the flow-LM transformer keeps it.
	transformerLayerNormEps = 1e-5
	// SimpleMLPAdaLN ResBlock/FinalLayer LayerNorm eps.
	flowLayerNormEps = 1e-6
	// TimestepEmbedder RMS eps (paired with its UNBIASED-variance quirk).
	timeEmbedRMSEps = 1e-5
)

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
	timeEmbeds             [2]timeEmbed
	blocks                 []flowBlock
	finalAdaW, finalAdaB   []float32 // flowDim -> 2*flowDim
	finalLinW, finalLinB   []float32 // flowDim -> latent
}

// Model: loaded backbone weights and derived dims.
type Model struct {
	Dims Dims

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
	if tr.NumHeads <= 0 {
		return artifactConfig{}, fmt.Errorf("speechsynth: config lacks positive flow_lm.transformer.num_heads")
	}
	if tr.MaxPeriod <= 0 {
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
				return nil, fmt.Errorf("speechsynth: tensor %q dimension %d out of range", name, dim)
			}
			dims[i] = int(dim)
		}
		shapes[name] = dims
	}

	read := func(name string, wantShape ...int) ([]float32, error) {
		tensor, ok := source.Tensors[name]
		if !ok {
			return nil, fmt.Errorf("speechsynth: tensor %q missing", name)
		}
		shape := shapes[name]
		if len(shape) != len(wantShape) {
			return nil, fmt.Errorf("speechsynth: tensor %q rank %d, want %d", name, len(shape), len(wantShape))
		}
		for i, want := range wantShape {
			if want > 0 && shape[i] != want {
				return nil, fmt.Errorf("speechsynth: tensor %q shape %v, want dim %d = %d", name, shape, i, want)
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
		Dims:       dims,
		invFreq:    hostmath.RopeInvFreq(dims.MaxPeriod, dims.HeadDim),
		scoreScale: float32(1 / math.Sqrt(float64(dims.HeadDim))),
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
			{&l.inProj, p + "self_attn.in_proj.weight", []int{3 * d, d}},
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
		{&m.outEosW, "flow_lm.out_eos.weight", []int{1, d}},
		{&m.BosEmb, "flow_lm.bos_emb", []int{latent}},
		{&m.EmbMean, "flow_lm.emb_mean", []int{latent}},
		{&m.EmbStd, "flow_lm.emb_std", []int{latent}},
	} {
		if *f.dst, err = read(f.name, f.want...); err != nil {
			return nil, err
		}
	}
	eosBias, err := read("flow_lm.out_eos.bias", 1)
	if err != nil {
		return nil, err
	}
	m.outEosB = eosBias[0]

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
		{&fn.finalAdaW, "flow_lm.flow_net.final_layer.adaLN_modulation.1.weight", []int{2 * flowDim, flowDim}},
		{&fn.finalAdaB, "flow_lm.flow_net.final_layer.adaLN_modulation.1.bias", []int{2 * flowDim}},
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
			{&te.l0w, p + "mlp.0.weight", []int{flowDim, 2 * dims.TimeFreqs}},
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
			{&b.adaW, p + "adaLN_modulation.1.weight", []int{3 * flowDim, flowDim}},
			{&b.adaB, p + "adaLN_modulation.1.bias", []int{3 * flowDim}},
		} {
			if *f.dst, err = read(f.name, f.want...); err != nil {
				return nil, err
			}
		}
	}
	if m.Codec, err = loadCodecDecoder(read, shapes, config, dims.LatentDim); err != nil {
		return nil, err
	}
	return m, nil
}

// deriveDims: every geometric fact from tensor shapes; config supplies only
// num_heads and max_period, and its restated shape facts must agree.
func deriveDims(shapes map[string][]int, config artifactConfig) (Dims, error) {
	var d Dims
	outNorm, ok := shapes["flow_lm.out_norm.weight"]
	if !ok || len(outNorm) != 1 {
		return d, fmt.Errorf("speechsynth: out_norm.weight missing; d_model underivable")
	}
	d.DModel = outNorm[0]

	var err error
	if d.Layers, err = tensorcatalog.IndexedCount(shapes, layerPrefix, ".norm1.weight"); err != nil {
		return d, err
	}
	lin1, ok := shapes[layerPrefix+"0.linear1.weight"]
	if !ok || len(lin1) != 2 || lin1[1] != d.DModel {
		return d, fmt.Errorf("speechsynth: layer-0 linear1 incompatible with d_model %d", d.DModel)
	}
	d.FF = lin1[0]
	inProj, ok := shapes[layerPrefix+"0.self_attn.in_proj.weight"]
	if !ok || len(inProj) != 2 || inProj[0] != 3*d.DModel || inProj[1] != d.DModel {
		return d, fmt.Errorf("speechsynth: layer-0 in_proj %v is not fused [3d, d]", inProj)
	}

	embed, ok := shapes["flow_lm.conditioner.embed.weight"]
	if !ok || len(embed) != 2 || embed[1] != d.DModel {
		return d, fmt.Errorf("speechsynth: conditioner embed missing or width != d_model")
	}
	d.TextVocab = embed[0]

	inputLinear, ok := shapes["flow_lm.input_linear.weight"]
	if !ok || len(inputLinear) != 2 || inputLinear[0] != d.DModel {
		return d, fmt.Errorf("speechsynth: input_linear missing or rows != d_model")
	}
	d.LatentDim = inputLinear[1]

	inputProj, ok := shapes["flow_lm.flow_net.input_proj.weight"]
	if !ok || len(inputProj) != 2 || inputProj[1] != d.LatentDim {
		return d, fmt.Errorf("speechsynth: flow input_proj missing or columns != latent dim %d", d.LatentDim)
	}
	d.FlowDim = inputProj[0]

	if d.FlowDepth, err = tensorcatalog.IndexedCount(shapes, resBlockPrefix, ".in_ln.weight"); err != nil {
		return d, err
	}
	freqs, ok := shapes["flow_lm.flow_net.time_embed.0.freqs"]
	if !ok || len(freqs) != 1 {
		return d, fmt.Errorf("speechsynth: time_embed.0.freqs missing")
	}
	d.TimeFreqs = freqs[0]

	tr := config.FlowLM.Transformer
	d.Heads = tr.NumHeads
	d.MaxPeriod = tr.MaxPeriod
	if d.DModel%d.Heads != 0 {
		return d, fmt.Errorf("speechsynth: heads %d do not partition d_model %d", d.Heads, d.DModel)
	}
	d.HeadDim = d.DModel / d.Heads
	if d.HeadDim%2 != 0 {
		return d, fmt.Errorf("speechsynth: head dim %d incompatible with paired RoPE", d.HeadDim)
	}

	// Config restatements of shape facts: agree or refuse (0 = unstated).
	nBinsPlusOne := 0
	if config.FlowLM.LookupTable.NBins != 0 {
		nBinsPlusOne = config.FlowLM.LookupTable.NBins + 1
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
		if check.claimed != 0 && check.claimed != check.derived {
			return d, fmt.Errorf("speechsynth: config %s=%d contradicts derived %d", check.name, check.claimed, check.derived)
		}
	}
	return d, nil
}
