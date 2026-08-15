package gemma4convert

// E4B tower layout: full vision transformer + conformer audio encoder.
// Every metadata value comes from config.json, processor_config.json, or
// source tensor shapes; every source tensor maps 1:1 into the output.

import (
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"

	"overgo/internal/gguf"
	"overgo/internal/safetensors"
)

type processorConfig struct {
	Audio struct {
		FeatureSize  uint32  `json:"feature_size"`
		FFTLength    uint32  `json:"fft_length"`
		FrameLength  uint32  `json:"frame_length"`
		HopLength    uint32  `json:"hop_length"`
		SampleRate   uint32  `json:"sampling_rate"`
		MinFrequency float32 `json:"min_frequency"`
		MaxFrequency float32 `json:"max_frequency"`
		MelFloor     float32 `json:"mel_floor"`
	} `json:"feature_extractor"`
	Image struct {
		Mean          []float32 `json:"image_mean"`
		Std           []float32 `json:"image_std"`
		MaxSoftTokens uint32    `json:"max_soft_tokens"`
	} `json:"image_processor"`
	Video struct {
		MaxSoftTokens uint32 `json:"max_soft_tokens"`
	} `json:"video_processor"`
}

// towerLayout: E4B checkpoints declare full towers via vision_config
// num_hidden_layers; the embedder layout (12B) has no such key.
func towerLayout(config modelConfig) bool {
	return config.Vision.HiddenLayers > 0
}

func validateTowerConfig(config modelConfig, processor processorConfig) error {
	vision, audio := config.Vision, config.Audio
	if vision.HiddenSize == 0 || vision.HiddenLayers == 0 || vision.AttentionHeads == 0 ||
		vision.IntermediateSize == 0 || vision.PatchSize == 0 || vision.PoolingSize == 0 ||
		vision.PositionEmbedding == 0 || vision.RMSEpsilon <= 0 || vision.Rope.Theta <= 0 ||
		vision.HiddenAct == "" {
		return fmt.Errorf("Gemma 4 converter: vision tower configuration is incomplete")
	}
	if audio.HiddenSize == 0 || audio.HiddenLayers == 0 || audio.AttentionHeads == 0 ||
		audio.ConvKernel == 0 || len(audio.SubChannels) == 0 || audio.ChunkSize == 0 ||
		audio.LogitCap <= 0 || audio.ResidualWeight <= 0 || audio.RMSEpsilon <= 0 ||
		audio.OutputProjDims == 0 || audio.HiddenAct == "" {
		return fmt.Errorf("Gemma 4 converter: audio tower configuration is incomplete")
	}
	feature := processor.Audio
	if feature.FeatureSize == 0 || feature.FFTLength == 0 || feature.FrameLength == 0 ||
		feature.HopLength == 0 || feature.SampleRate == 0 || feature.MaxFrequency <= 0 ||
		feature.MelFloor <= 0 || feature.MinFrequency < 0 {
		return fmt.Errorf("Gemma 4 converter: audio feature extractor configuration is incomplete")
	}
	if len(processor.Image.Mean) != rgbChannelCount || len(processor.Image.Std) != rgbChannelCount ||
		processor.Image.MaxSoftTokens == 0 || processor.Video.MaxSoftTokens == 0 {
		return fmt.Errorf("Gemma 4 converter: image processor configuration is incomplete")
	}
	return nil
}

// audioFeedForwardLength: audio_config carries no intermediate size; the
// ffw weight shape [intermediate, hidden] owns that fact.
func audioFeedForwardLength(source *safetensors.Source, config modelConfig) (uint32, error) {
	const name = "model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight"
	tensor, ok := source.Tensors[name]
	if !ok || len(tensor.Shape) != 2 || tensor.Shape[1] != uint64(config.Audio.HiddenSize) {
		return 0, fmt.Errorf("Gemma 4 converter: audio feed-forward tensor %q is missing or invalid", name)
	}
	return uint32(tensor.Shape[0]), nil
}

func towerProjectorMetadata(
	name string, config modelConfig, processor processorConfig, audioFeedForward uint32,
) []gguf.Metadata {
	vision, audio := config.Vision, config.Audio
	visionKVHeads := vision.KVHeads
	if visionKVHeads == 0 {
		visionKVHeads = vision.AttentionHeads
	}
	visionHeadDim := vision.HeadDim
	if visionHeadDim == 0 {
		visionHeadDim = vision.HiddenSize / vision.AttentionHeads
	}
	subChannels := make([]int32, len(audio.SubChannels))
	for index, channels := range audio.SubChannels {
		subChannels[index] = int32(channels)
	}
	return []gguf.Metadata{
		stringMetadata("general.architecture", "clip"),
		stringMetadata("general.name", name+" multimodal projector"),
		stringMetadata("clip.vision.projector_type", "gemma4vision"),
		boolMetadata("clip.has_vision_encoder", true),
		uint32Metadata("clip.vision.block_count", vision.HiddenLayers),
		uint32Metadata("clip.vision.embedding_length", vision.HiddenSize),
		uint32Metadata("clip.vision.feed_forward_length", vision.IntermediateSize),
		uint32Metadata("clip.vision.attention.head_count", vision.AttentionHeads),
		uint32Metadata("clip.vision.attention.head_count_kv", visionKVHeads),
		uint32Metadata("clip.vision.attention.key_length", visionHeadDim),
		float32Metadata("clip.vision.attention.layer_norm_epsilon", vision.RMSEpsilon),
		uint32Metadata("clip.vision.patch_size", vision.PatchSize),
		uint32Metadata("clip.vision.projector_scale_factor", vision.PoolingSize),
		uint32Metadata("clip.vision.position_embedding_size", vision.PositionEmbedding),
		uint32Metadata("clip.vision.projection_dim", config.Text.HiddenSize),
		float32Metadata("clip.vision.rope.freq_base", vision.Rope.Theta),
		arrayMetadata("clip.vision.input_scale", gguf.ValueTypeFloat32, []float32{2, 2, 2}),
		arrayMetadata("clip.vision.input_bias", gguf.ValueTypeFloat32, []float32{-1, -1, -1}),
		uint32Metadata("clip.vision.max_soft_tokens", processor.Image.MaxSoftTokens),
		uint32Metadata("clip.vision.video_max_soft_tokens", processor.Video.MaxSoftTokens),
		stringMetadata("clip.vision.hidden_activation", vision.HiddenAct),
		stringMetadata("clip.audio.projector_type", "gemma4audio"),
		boolMetadata("clip.has_audio_encoder", true),
		uint32Metadata("clip.audio.block_count", audio.HiddenLayers),
		uint32Metadata("clip.audio.embedding_length", audio.HiddenSize),
		uint32Metadata("clip.audio.feed_forward_length", audioFeedForward),
		uint32Metadata("clip.audio.attention.head_count", audio.AttentionHeads),
		float32Metadata("clip.audio.attention.layer_norm_epsilon", audio.RMSEpsilon),
		uint32Metadata("clip.audio.attention.chunk_size", audio.ChunkSize),
		uint32Metadata("clip.audio.attention.context_left", audio.ContextLeft),
		uint32Metadata("clip.audio.attention.context_right", audio.ContextRight),
		float32Metadata("clip.audio.attention.logit_softcapping", audio.LogitCap),
		uint32Metadata("clip.audio.conv_kernel_size", audio.ConvKernel),
		arrayMetadata("clip.audio.subsampling_conv_channels", gguf.ValueTypeInt32, subChannels),
		float32Metadata("clip.audio.residual_weight", audio.ResidualWeight),
		uint32Metadata("clip.audio.output_projection_dim", audio.OutputProjDims),
		uint32Metadata("clip.audio.projection_dim", config.Text.HiddenSize),
		stringMetadata("clip.audio.hidden_act", audio.HiddenAct),
		uint32Metadata("clip.audio.num_mel_bins", processor.Audio.FeatureSize),
		uint32Metadata("clip.audio.fft_length", processor.Audio.FFTLength),
		uint32Metadata("clip.audio.frame_length", processor.Audio.FrameLength),
		uint32Metadata("clip.audio.hop_length", processor.Audio.HopLength),
		uint32Metadata("clip.audio.sample_rate", processor.Audio.SampleRate),
		float32Metadata("clip.audio.min_frequency", processor.Audio.MinFrequency),
		float32Metadata("clip.audio.max_frequency", processor.Audio.MaxFrequency),
		float32Metadata("clip.audio.mel_floor", processor.Audio.MelFloor),
	}
}

var visionLayerPattern = regexp.MustCompile(`^model\.vision_tower\.encoder\.layers\.(\d+)\.(.+)$`)
var audioLayerPattern = regexp.MustCompile(`^model\.audio_tower\.layers\.(\d+)\.(.+)$`)
var audioSubsamplePattern = regexp.MustCompile(
	`^model\.audio_tower\.subsample_conv_projection\.layer(\d+)\.(conv|norm)\.weight$`)

// towerFixedNames: one entry per non-repeating tensor.
var towerFixedNames = map[string]string{
	"model.vision_tower.patch_embedder.input_proj.weight":                  "v.patch_embd.weight",
	"model.vision_tower.patch_embedder.position_embedding_table":           "v.position_embd.weight",
	"model.embed_vision.embedding_projection.weight":                       "mm.input_projection.weight",
	"model.audio_tower.subsample_conv_projection.input_proj_linear.weight": "a.input_proj.weight",
	"model.audio_tower.output_proj.weight":                                 "a.output_proj.weight",
	"model.audio_tower.output_proj.bias":                                   "a.output_proj.bias",
	"model.embed_audio.embedding_projection.weight":                        "mm.a.input_projection.weight",
}

// visionLinearModules: clipped-linear wrappers; suffix set is
// linear.weight + the four calibration scalars. Base names reuse the
// language-block vocabulary (identical block structure).
var visionLinearModules = map[string]string{
	"self_attn.q_proj": "attn_q",
	"self_attn.k_proj": "attn_k",
	"self_attn.v_proj": "attn_v",
	"self_attn.o_proj": "attn_output",
	"mlp.gate_proj":    "ffn_gate",
	"mlp.up_proj":      "ffn_up",
	"mlp.down_proj":    "ffn_down",
}

var visionPlainSuffixes = map[string]string{
	"input_layernorm.weight":            "attn_norm.weight",
	"post_attention_layernorm.weight":   "post_attention_norm.weight",
	"pre_feedforward_layernorm.weight":  "ffn_norm.weight",
	"post_feedforward_layernorm.weight": "post_ffw_norm.weight",
	"self_attn.q_norm.weight":           "attn_q_norm.weight",
	"self_attn.k_norm.weight":           "attn_k_norm.weight",
}

var audioLinearModules = map[string]string{
	"feed_forward1.ffw_layer_1": "ffn1_up",
	"feed_forward1.ffw_layer_2": "ffn1_down",
	"feed_forward2.ffw_layer_1": "ffn2_up",
	"feed_forward2.ffw_layer_2": "ffn2_down",
	"lconv1d.linear_start":      "conv_start",
	"lconv1d.linear_end":        "conv_end",
	"self_attn.q_proj":          "attn_q",
	"self_attn.k_proj":          "attn_k",
	"self_attn.v_proj":          "attn_v",
	"self_attn.post":            "attn_output",
}

var audioPlainSuffixes = map[string]string{
	"feed_forward1.pre_layer_norm.weight":  "ffn1_norm.weight",
	"feed_forward1.post_layer_norm.weight": "ffn1_post_norm.weight",
	"feed_forward2.pre_layer_norm.weight":  "ffn2_norm.weight",
	"feed_forward2.post_layer_norm.weight": "ffn2_post_norm.weight",
	"lconv1d.pre_layer_norm.weight":        "conv_pre_norm.weight",
	"lconv1d.conv_norm.weight":             "conv_norm.weight",
	"lconv1d.depthwise_conv1d.weight":      "conv_dw.weight",
	"norm_pre_attn.weight":                 "attn_norm.weight",
	"norm_post_attn.weight":                "post_attention_norm.weight",
	"norm_out.weight":                      "out_norm.weight",
	"self_attn.relative_k_proj.weight":     "attn_rel_k.weight",
	"self_attn.per_dim_scale":              "attn_per_dim_scale.weight",
}

var clippedLinearSuffixes = map[string]string{
	"linear.weight": "weight",
	"input_min":     "input_min",
	"input_max":     "input_max",
	"output_min":    "output_min",
	"output_max":    "output_max",
}

func mapLayerSuffix(remainder string, linear, plain map[string]string) (string, bool) {
	if mapped, ok := plain[remainder]; ok {
		return mapped, true
	}
	for module, base := range linear {
		if !strings.HasPrefix(remainder, module+".") {
			continue
		}
		mapped, ok := clippedLinearSuffixes[remainder[len(module)+1:]]
		if !ok {
			return "", false
		}
		return base + "." + mapped, true
	}
	return "", false
}

func towerTensorName(name string) (string, bool) {
	if mapped, ok := towerFixedNames[name]; ok {
		return mapped, true
	}
	if match := visionLayerPattern.FindStringSubmatch(name); match != nil {
		mapped, ok := mapLayerSuffix(match[2], visionLinearModules, visionPlainSuffixes)
		if !ok {
			return "", false
		}
		return "v.blk." + match[1] + "." + mapped, true
	}
	if match := audioLayerPattern.FindStringSubmatch(name); match != nil {
		mapped, ok := mapLayerSuffix(match[2], audioLinearModules, audioPlainSuffixes)
		if !ok {
			return "", false
		}
		return "a.blk." + match[1] + "." + mapped, true
	}
	if match := audioSubsamplePattern.FindStringSubmatch(name); match != nil {
		if match[2] == "conv" {
			return "a.conv." + match[1] + ".weight", true
		}
		return "a.conv." + match[1] + ".norm.weight", true
	}
	return "", false
}

func towerTensors(source *safetensors.Source, outputF32 bool) ([]gguf.TensorData, error) {
	tensors := make([]gguf.TensorData, 0, len(source.Tensors))
	for sourceName, tensor := range source.Tensors {
		if strings.HasPrefix(sourceName, "model.language_model.") {
			continue
		}
		destinationName, ok := towerTensorName(sourceName)
		if !ok {
			return nil, fmt.Errorf("Gemma 4 converter: tower tensor %q has no mapping", sourceName)
		}
		if tensor.DType != "BF16" {
			return nil, fmt.Errorf("Gemma 4 converter: tower tensor %q uses %s", sourceName, tensor.DType)
		}
		shape := gguf.ReverseShape(tensor.Shape)
		var reader io.Reader = tensor.Reader()
		dataType := gguf.DTypeBF16
		if len(shape) == 0 {
			// rank-0 calibration scalar: shape [1], stored F32 (layer_scalar rule)
			shape = []uint64{1}
			promoted, err := safetensors.F32Reader(tensor)
			if err != nil {
				return nil, fmt.Errorf("Gemma 4 converter: tensor %q: %w", sourceName, err)
			}
			reader, dataType = promoted, gguf.DTypeF32
		} else if outputF32 {
			promoted, err := safetensors.PromoteF32Reader(reader, tensor.DType)
			if err != nil {
				return nil, fmt.Errorf("Gemma 4 converter: tensor %q: %w", sourceName, err)
			}
			reader, dataType = promoted, gguf.DTypeF32
		}
		tensors = append(tensors, gguf.TensorData{
			Name: destinationName, Shape: shape, Type: dataType, Data: reader,
		})
	}
	sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })
	if len(tensors) == 0 {
		return nil, fmt.Errorf("Gemma 4 converter: no tower tensors found")
	}
	return tensors, nil
}
