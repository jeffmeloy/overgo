// Package latentimage recognizes and describes a dual-stream latent-image
// diffusion pipeline (image DiT + text-fusion stream, a QwenImage-family 3-D
// causal VAE, a Qwen3-VL text encoder with selected-layer hidden fusion, and a
// Qwen2 tokenizer) packaged as one artifact directory. Geometry is DERIVED:
// every Spec field is READ from a config key or DERIVED from another config
// field, and cross-checked against real tensor shapes by VerifyCheckpoint. No
// magic numbers, no vendor branching in executor types. Recognition and policy
// come from modelrecipe.ImageProfile. Parallels
// internal/latentvideo (Wan) for the media spine; this rung is arch recognition
// + shape-derived spec only (serving/denoise is a later rung).
package latentimage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/modelrecipe"
)

// TokenizerKind names the tokenizer class from model_index.json.
type TokenizerKind string

// TransformerSpec: dual-stream latent-image diffusion transformer geometry.
// Field comments name the SOURCE: [cfg K] a config key, [der ...] a derivation,
// [xcheck ...] the tensor(s) VerifyCheckpoint asserts it against.
type TransformerSpec struct {
	Layers        int     // [cfg num_layers] xcheck count(transformer_blocks.N)
	Heads         int     // [cfg num_attention_heads]
	KVHeads       int     // [cfg num_key_value_heads] GQA
	HeadDim       int     // [cfg attention_head_dim] xcheck len(attn.norm_q.weight)
	Hidden        int     // [der Heads*HeadDim] xcheck img_in.weight[0], attn.to_q.weight[0]
	KVDim         int     // [der KVHeads*HeadDim] xcheck attn.to_k.weight[0], attn.to_v.weight[0]
	InChannels    int     // [cfg in_channels] xcheck z_dim*patch^2, img_in.weight[1], final_layer.linear.weight[0]
	Intermediate  int     // [cfg intermediate_size] xcheck ff.gate.weight[0], ff.down.weight[1]
	RopeAxes      [3]int  // [cfg axes_dims_rope] xcheck sum==HeadDim
	RopeTheta     float64 // [cfg rope_theta]
	NormEps       float64 // [cfg norm_eps] zero-centered RMSNorm epsilon
	TimestepEmbed int     // [cfg timestep_embed_dim] xcheck time_embed.linear_1.weight[1]
	ModFields     int     // [der time_mod_proj.weight[0]/Hidden] xcheck transformer_blocks.0.scale_shift_table[0]

	// text-fusion stream
	TextLayers          int // [cfg num_text_layers] xcheck text_fusion.projector.weight[1], len(SelectLayers)
	TextHidden          int // [cfg text_hidden_dim] xcheck txt_in.linear_1.weight[1], txt_in.norm.weight
	TextIntermediate    int // [cfg text_intermediate_size] xcheck layerwise_blocks.0.ff.gate.weight[0]
	TextHeads           int // [cfg text_num_attention_heads] xcheck layerwise_blocks.0.attn.to_q.weight[0]/HeadDim
	TextKVHeads         int // [cfg text_num_key_value_heads]
	LayerwiseTextBlocks int // [cfg num_layerwise_text_blocks] xcheck count(text_fusion.layerwise_blocks.N)
	RefinerTextBlocks   int // [cfg num_refiner_text_blocks] xcheck count(text_fusion.refiner_blocks.N)
}

// VAESpec: QwenImage-family 3-D causal VAE geometry.
type VAESpec struct {
	ZDim          int       // [cfg z_dim] xcheck post_quant_conv.weight[1], decoder.conv_in.weight[1]
	BaseDim       int       // [cfg base_dim] xcheck encoder.conv_in.weight[0], decoder.conv_out.weight[1]
	DimMult       []int     // [cfg dim_mult]
	InputChannels int       // [cfg input_channels] xcheck encoder.conv_in.weight[1], decoder.conv_out.weight[0]
	ResBlocks     int       // [cfg num_res_blocks]
	DeepestDim    int       // [der BaseDim*DimMult[last]] xcheck encoder.conv_out.weight[1], decoder.conv_in.weight[0]
	QuantChannels int       // [der 2*ZDim] xcheck quant_conv.weight[0], encoder.conv_out.weight[0]
	SpatialScale  int       // [der 2^(len(DimMult)-1)] spatial downsample factor
	LatentsMean   []float64 // [cfg latents_mean] xcheck len==ZDim
	LatentsStd    []float64 // [cfg latents_std] xcheck len==ZDim
}

// TextEncoderSpec: Qwen3-VL text encoder + selected-layer fusion contract.
type TextEncoderSpec struct {
	ModelType    string  // [cfg model_type] == profile recognition
	HiddenLayers int     // [cfg text_config.num_hidden_layers] xcheck count(language_model.layers.N)
	Hidden       int     // [cfg text_config.hidden_size] xcheck language_model.embed_tokens.weight[1]
	Intermediate int     // [cfg text_config.intermediate_size] xcheck mlp.down_proj.weight[1]
	Heads        int     // [cfg text_config.num_attention_heads] xcheck q_proj.weight[0]/HeadDim
	KVHeads      int     // [cfg text_config.num_key_value_heads] xcheck k_proj.weight[0]/HeadDim
	HeadDim      int     // [cfg text_config.head_dim] xcheck self_attn.q_norm.weight
	VocabSize    int     // [cfg text_config.vocab_size] xcheck embed_tokens.weight[0]
	RopeTheta    float64 // [cfg text_config.rope_parameters.rope_theta]
	RMSNormEps   float64 // [cfg text_config.rms_norm_eps] standard RMSNorm epsilon
	SelectLayers []int   // [cfg model_index.text_encoder_select_layers] fusion pick (hidden_states idx)
}

// Spec: the recognized, geometry-derived artifact description. Serving-only;
// no training lane. Sub-specs are functional; Family/Pipeline carry the tag.
type Spec struct {
	Profile     modelrecipe.ImageProfile
	Pipeline    string // model_index _class_name
	Family      string // profile discovery scope
	Scheduler   string // model_index scheduler class
	ServingOnly bool   // always true for this pipeline
	Distilled   bool   // model_index is_distilled
	PatchSize   int    // model_index patch_size (latent->token patch)
	Transformer TransformerSpec
	VAE         VAESpec
	TextEncoder TextEncoderSpec
	Tokenizer   TokenizerKind
}

// ---- config JSON shapes (read-only, trusted artifact) ----------------------

type classEntry [2]string // ["library","ClassName"]

type modelIndex struct {
	ClassName    string     `json:"_class_name"`
	PatchSize    int        `json:"patch_size"`
	IsDistilled  bool       `json:"is_distilled"`
	SelectLayers []int      `json:"text_encoder_select_layers"`
	Scheduler    classEntry `json:"scheduler"`
	TextEncoder  classEntry `json:"text_encoder"`
	Tokenizer    classEntry `json:"tokenizer"`
	Transformer  classEntry `json:"transformer"`
	VAE          classEntry `json:"vae"`
}

type transformerConfig struct {
	ClassName         string  `json:"_class_name"`
	AttentionHeadDim  int     `json:"attention_head_dim"`
	AxesDimsRope      []int   `json:"axes_dims_rope"`
	InChannels        int     `json:"in_channels"`
	IntermediateSize  int     `json:"intermediate_size"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	NumKeyValueHeads  int     `json:"num_key_value_heads"`
	NormEps           float64 `json:"norm_eps"`
	NumLayers         int     `json:"num_layers"`
	NumLayerwiseText  int     `json:"num_layerwise_text_blocks"`
	NumRefinerText    int     `json:"num_refiner_text_blocks"`
	NumTextLayers     int     `json:"num_text_layers"`
	RopeTheta         float64 `json:"rope_theta"`
	TextHiddenDim     int     `json:"text_hidden_dim"`
	TextIntermediate  int     `json:"text_intermediate_size"`
	TextNumAttnHeads  int     `json:"text_num_attention_heads"`
	TextNumKVHeads    int     `json:"text_num_key_value_heads"`
	TimestepEmbedDim  int     `json:"timestep_embed_dim"`
}

type vaeConfig struct {
	ClassName     string    `json:"_class_name"`
	BaseDim       int       `json:"base_dim"`
	DimMult       []int     `json:"dim_mult"`
	InputChannels int       `json:"input_channels"`
	LatentsMean   []float64 `json:"latents_mean"`
	LatentsStd    []float64 `json:"latents_std"`
	NumResBlocks  int       `json:"num_res_blocks"`
	ZDim          int       `json:"z_dim"`
}

type textEncoderConfig struct {
	ModelType  string `json:"model_type"`
	TextConfig struct {
		HeadDim          int     `json:"head_dim"`
		HiddenSize       int     `json:"hidden_size"`
		IntermediateSize int     `json:"intermediate_size"`
		NumAttentionHead int     `json:"num_attention_heads"`
		NumHiddenLayers  int     `json:"num_hidden_layers"`
		NumKeyValueHeads int     `json:"num_key_value_heads"`
		VocabSize        int     `json:"vocab_size"`
		RMSNormEps       float64 `json:"rms_norm_eps"`
		RopeParameters   struct {
			RopeTheta float64 `json:"rope_theta"`
		} `json:"rope_parameters"`
	} `json:"text_config"`
}

type schedulerConfig struct {
	ClassName         string  `json:"_class_name"`
	NumTrainTimesteps int     `json:"num_train_timesteps"`
	MaxShift          float64 `json:"max_shift"`
}

type tokenizerConfig struct {
	ClassName string `json:"tokenizer_class"`
	PadToken  string `json:"pad_token"`
}

// ---- recognition + config-level derivation ---------------------------------

// RecognizePipeline reports whether dir holds this pipeline (model_index.json
// _class_name resolves a recipe profile. A directory without model_index.json,
// or with an unknown class, reports (nil,false,nil), so discovery can
// probe any directory. Malformed JSON reports an error.
func RecognizePipeline(dir string) (*Spec, bool, error) {
	recognized, err := IsPipeline(dir)
	if err != nil || !recognized {
		return nil, recognized, err
	}
	spec, err := Derive(dir)
	if err != nil {
		return nil, false, err
	}
	return spec, true, nil
}

// IsPipeline: bounded class probe without config derivation.
func IsPipeline(dir string) (bool, error) {
	path := filepath.Join(dir, "model_index.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("latentimage: read model_index: %w", err)
	}
	var index modelIndex
	if err := json.Unmarshal(raw, &index); err != nil {
		return false, fmt.Errorf("latentimage: parse model_index: %w", err)
	}
	_, recognized, err := modelrecipe.ImageProfileForPipeline(index.ClassName)
	return recognized, err
}

func imageProfileFromDir(dir string) (modelrecipe.ImageProfile, error) {
	var index modelIndex
	if err := readJSON(filepath.Join(dir, "model_index.json"), &index); err != nil {
		return modelrecipe.ImageProfile{}, err
	}
	profile, ok, err := modelrecipe.ImageProfileForPipeline(index.ClassName)
	if err != nil {
		return modelrecipe.ImageProfile{}, err
	}
	if !ok {
		return modelrecipe.ImageProfile{}, fmt.Errorf("latentimage: pipeline class %q has no image profile", index.ClassName)
	}
	return profile, nil
}

// Derive reads model_index.json + the three sub-configs and builds the Spec,
// running every CONFIG-vs-CONFIG cross-check (the ones that need no checkpoint):
// in_channels == z_dim*patch^2, sum(axes_dims_rope) == head_dim,
// len(select_layers) == num_text_layers, len(latents_*) == z_dim, and the
// cross-model text_hidden_dim == text_encoder.hidden_size fusion boundary.
// VerifyCheckpoint then asserts the config values against real tensor shapes.
func Derive(dir string) (*Spec, error) {
	var index modelIndex
	if err := readJSON(filepath.Join(dir, "model_index.json"), &index); err != nil {
		return nil, err
	}
	profile, recognized, err := modelrecipe.ImageProfileForPipeline(index.ClassName)
	if err != nil {
		return nil, err
	}
	if !recognized {
		return nil, fmt.Errorf("latentimage: pipeline class %q has no image profile", index.ClassName)
	}
	var tcfg transformerConfig
	if err := readJSON(filepath.Join(dir, "transformer", "config.json"), &tcfg); err != nil {
		return nil, err
	}
	if tcfg.ClassName != profile.Recognition.Transformer {
		return nil, fmt.Errorf("latentimage: transformer class %q != profile %q", tcfg.ClassName, profile.Recognition.Transformer)
	}
	var vcfg vaeConfig
	if err := readJSON(filepath.Join(dir, "vae", "config.json"), &vcfg); err != nil {
		return nil, err
	}
	if vcfg.ClassName != profile.Recognition.VAE {
		return nil, fmt.Errorf("latentimage: vae class %q != profile %q", vcfg.ClassName, profile.Recognition.VAE)
	}
	var ecfg textEncoderConfig
	if err := readJSON(filepath.Join(dir, "text_encoder", "config.json"), &ecfg); err != nil {
		return nil, err
	}
	if ecfg.ModelType != profile.Recognition.TextEncoder {
		return nil, fmt.Errorf("latentimage: text_encoder model_type %q != profile %q", ecfg.ModelType, profile.Recognition.TextEncoder)
	}
	var scfg schedulerConfig
	if err := readJSON(filepath.Join(dir, "scheduler", "scheduler_config.json"), &scfg); err != nil {
		return nil, err
	}
	if index.Scheduler[1] != profile.Recognition.Scheduler || scfg.ClassName != profile.Recognition.Scheduler ||
		scfg.NumTrainTimesteps != profile.Sampling.NumTrainTimesteps || scfg.MaxShift != profile.Sampling.DynamicShiftMu {
		return nil, fmt.Errorf("latentimage: scheduler artifact disagrees with profile %q", profile.ID)
	}
	var tokcfg tokenizerConfig
	if err := readJSON(filepath.Join(dir, "tokenizer", "tokenizer_config.json"), &tokcfg); err != nil {
		return nil, err
	}
	if index.Tokenizer[1] != profile.Recognition.Tokenizer || tokcfg.ClassName != profile.Recognition.Tokenizer ||
		tokcfg.PadToken != profile.Conditioning.PadToken {
		return nil, fmt.Errorf("latentimage: tokenizer artifact disagrees with profile %q", profile.ID)
	}

	if len(tcfg.AxesDimsRope) != 3 {
		return nil, fmt.Errorf("latentimage: axes_dims_rope rank %d != 3", len(tcfg.AxesDimsRope))
	}
	if len(vcfg.DimMult) == 0 {
		return nil, fmt.Errorf("latentimage: empty dim_mult")
	}

	spec := &Spec{
		Profile:     profile,
		Pipeline:    index.ClassName,
		Family:      profile.Family,
		Scheduler:   index.Scheduler[1],
		ServingOnly: true,
		Distilled:   index.IsDistilled,
		PatchSize:   index.PatchSize,
		Tokenizer:   TokenizerKind(index.Tokenizer[1]),
		Transformer: TransformerSpec{
			Layers:              tcfg.NumLayers,
			Heads:               tcfg.NumAttentionHeads,
			KVHeads:             tcfg.NumKeyValueHeads,
			HeadDim:             tcfg.AttentionHeadDim,
			Hidden:              tcfg.NumAttentionHeads * tcfg.AttentionHeadDim,
			KVDim:               tcfg.NumKeyValueHeads * tcfg.AttentionHeadDim,
			InChannels:          tcfg.InChannels,
			Intermediate:        tcfg.IntermediateSize,
			RopeAxes:            [3]int{tcfg.AxesDimsRope[0], tcfg.AxesDimsRope[1], tcfg.AxesDimsRope[2]},
			RopeTheta:           tcfg.RopeTheta,
			NormEps:             tcfg.NormEps,
			TimestepEmbed:       tcfg.TimestepEmbedDim,
			ModFields:           0, // derived from tensor shape in VerifyCheckpoint
			TextLayers:          tcfg.NumTextLayers,
			TextHidden:          tcfg.TextHiddenDim,
			TextIntermediate:    tcfg.TextIntermediate,
			TextHeads:           tcfg.TextNumAttnHeads,
			TextKVHeads:         tcfg.TextNumKVHeads,
			LayerwiseTextBlocks: tcfg.NumLayerwiseText,
			RefinerTextBlocks:   tcfg.NumRefinerText,
		},
		VAE: VAESpec{
			ZDim:          vcfg.ZDim,
			BaseDim:       vcfg.BaseDim,
			DimMult:       vcfg.DimMult,
			InputChannels: vcfg.InputChannels,
			ResBlocks:     vcfg.NumResBlocks,
			DeepestDim:    vcfg.BaseDim * vcfg.DimMult[len(vcfg.DimMult)-1],
			QuantChannels: 2 * vcfg.ZDim,
			SpatialScale:  1 << (len(vcfg.DimMult) - 1),
			LatentsMean:   vcfg.LatentsMean,
			LatentsStd:    vcfg.LatentsStd,
		},
		TextEncoder: TextEncoderSpec{
			ModelType:    ecfg.ModelType,
			HiddenLayers: ecfg.TextConfig.NumHiddenLayers,
			Hidden:       ecfg.TextConfig.HiddenSize,
			Intermediate: ecfg.TextConfig.IntermediateSize,
			Heads:        ecfg.TextConfig.NumAttentionHead,
			KVHeads:      ecfg.TextConfig.NumKeyValueHeads,
			HeadDim:      ecfg.TextConfig.HeadDim,
			VocabSize:    ecfg.TextConfig.VocabSize,
			RopeTheta:    ecfg.TextConfig.RopeParameters.RopeTheta,
			RMSNormEps:   ecfg.TextConfig.RMSNormEps,
			SelectLayers: index.SelectLayers,
		},
	}

	if err := spec.crossCheckConfig(); err != nil {
		return nil, err
	}
	return spec, nil
}

// crossCheckConfig runs every derivation that needs only config values, so a
// bad config is rejected at recognition time (before any checkpoint read).
func (s *Spec) crossCheckConfig() error {
	t, v, e := &s.Transformer, &s.VAE, &s.TextEncoder

	// in_channels == latent_ch(z_dim) * patch^2  (transformer <-> vae <-> model_index)
	if want := v.ZDim * s.PatchSize * s.PatchSize; t.InChannels != want {
		return fmt.Errorf("latentimage: in_channels %d != z_dim*patch^2 %d (z=%d patch=%d)",
			t.InChannels, want, v.ZDim, s.PatchSize)
	}
	// sum(axes_dims_rope) == head_dim
	if sum := t.RopeAxes[0] + t.RopeAxes[1] + t.RopeAxes[2]; sum != t.HeadDim {
		return fmt.Errorf("latentimage: sum(axes_dims_rope)=%d != head_dim %d", sum, t.HeadDim)
	}
	// len(select_layers) == num_text_layers  (model_index <-> transformer)
	if len(e.SelectLayers) != t.TextLayers {
		return fmt.Errorf("latentimage: len(select_layers)=%d != num_text_layers %d", len(e.SelectLayers), t.TextLayers)
	}
	// Selected layers index hidden_states[N] captured AFTER decoder layer N-1
	// (adaptive convention; hidden_states[0] is the embedding). Valid range is
	// [1, HiddenLayers], matching captureSlots -- the code that executes the
	// capture -- so a config that passes derivation cannot fail at execution.
	for _, layer := range e.SelectLayers {
		if layer < 1 || layer > e.HiddenLayers {
			return fmt.Errorf("latentimage: select layer %d out of range [1,%d]", layer, e.HiddenLayers)
		}
	}
	// latents_mean/std length == z_dim
	if len(v.LatentsMean) != v.ZDim || len(v.LatentsStd) != v.ZDim {
		return fmt.Errorf("latentimage: latents_mean/std len (%d/%d) != z_dim %d",
			len(v.LatentsMean), len(v.LatentsStd), v.ZDim)
	}
	// fusion boundary: transformer text_hidden_dim == text encoder hidden_size
	if t.TextHidden != e.Hidden {
		return fmt.Errorf("latentimage: text_hidden_dim %d != text_encoder hidden_size %d", t.TextHidden, e.Hidden)
	}
	// GQA sanity
	if t.KVHeads <= 0 || t.Heads%t.KVHeads != 0 {
		return fmt.Errorf("latentimage: heads %d not a multiple of kv_heads %d", t.Heads, t.KVHeads)
	}
	return nil
}

func readJSON(path string, dst any) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("latentimage: read %s: %w", filepath.Base(filepath.Dir(path))+"/"+filepath.Base(path), err)
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return fmt.Errorf("latentimage: parse %s: %w", filepath.Base(path), err)
	}
	return nil
}
