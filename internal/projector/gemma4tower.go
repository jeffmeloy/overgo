package projector

// Gemma 4 tower projector (E4B layout): full vision transformer plus
// conformer audio encoder. This file owns the mmproj contract — metadata
// keys, tensor names, and shapes — that the converter must satisfy.
// Encoding runs in a later serving step; opening validates the catalog.

import (
	"errors"
	"fmt"

	"overgo/internal/gguf"
)

const (
	gemma4VisionTowerProjectorType = "gemma4vision"
	gemma4AudioTowerProjectorType  = "gemma4audio"
)

type Gemma4VisionTowerSpec struct {
	Layers           int
	Hidden           int
	Heads            int
	KVHeads          int
	HeadDim          int
	Intermediate     int
	PatchSize        int
	PoolKernel       int
	PositionCount    int
	ProjectionDim    int
	MaxImageTokens   int
	MaxVideoTokens   int
	RMSNormEpsilon   float32
	RopeFreqBase     float32
	InputScale       [3]float32
	InputBias        [3]float32
	HiddenActivation string
}

type Gemma4AudioTowerSpec struct {
	Layers           int
	Hidden           int
	Heads            int
	HeadDim          int
	Intermediate     int
	ConvKernel       int
	SubChannels      []int
	ChunkSize        int
	ContextLeft      int
	ContextRight     int
	OutputProjDim    int
	ProjectionDim    int
	MelBins          int
	FFTLength        int
	FrameLength      int
	HopLength        int
	SampleRate       int
	LogitSoftcap     float32
	ResidualWeight   float32
	RMSNormEpsilon   float32
	MinFrequency     float32
	MaxFrequency     float32
	MelFloor         float32
	HiddenActivation string
}

type Gemma4TowerSpec struct {
	Vision Gemma4VisionTowerSpec
	Audio  Gemma4AudioTowerSpec
}

type Gemma4TowerRunner struct {
	projectorResources
	spec      Gemma4TowerSpec
	audioPlan *gemma4AudioFrontendPlan
}

func (r *Gemma4TowerRunner) Spec() Gemma4TowerSpec {
	if r == nil {
		return Gemma4TowerSpec{}
	}
	return r.spec
}

func ReadGemma4TowerSpec(file *gguf.File) (Gemma4TowerSpec, error) {
	var spec Gemma4TowerSpec
	if err := validateVisionProjector(file, "clip.vision.projector_type", gemma4VisionTowerProjectorType); err != nil {
		return spec, err
	}
	if err := validateProjector(
		file, "clip.audio.projector_type", "clip.has_audio_encoder", gemma4AudioTowerProjectorType, "audio",
	); err != nil {
		return spec, err
	}
	vision := &spec.Vision
	audio := &spec.Audio
	if err := readMetadataIntFields(file,
		metadataIntField{"clip.vision.block_count", &vision.Layers},
		metadataIntField{"clip.vision.embedding_length", &vision.Hidden},
		metadataIntField{"clip.vision.feed_forward_length", &vision.Intermediate},
		metadataIntField{"clip.vision.attention.head_count", &vision.Heads},
		metadataIntField{"clip.vision.attention.head_count_kv", &vision.KVHeads},
		metadataIntField{"clip.vision.attention.key_length", &vision.HeadDim},
		metadataIntField{"clip.vision.patch_size", &vision.PatchSize},
		metadataIntField{"clip.vision.projector_scale_factor", &vision.PoolKernel},
		metadataIntField{"clip.vision.position_embedding_size", &vision.PositionCount},
		metadataIntField{"clip.vision.projection_dim", &vision.ProjectionDim},
		metadataIntField{"clip.vision.max_soft_tokens", &vision.MaxImageTokens},
		metadataIntField{"clip.vision.video_max_soft_tokens", &vision.MaxVideoTokens},
		metadataIntField{"clip.audio.block_count", &audio.Layers},
		metadataIntField{"clip.audio.embedding_length", &audio.Hidden},
		metadataIntField{"clip.audio.feed_forward_length", &audio.Intermediate},
		metadataIntField{"clip.audio.attention.head_count", &audio.Heads},
		metadataIntField{"clip.audio.attention.chunk_size", &audio.ChunkSize},
		metadataIntField{"clip.audio.attention.context_left", &audio.ContextLeft},
		metadataIntField{"clip.audio.attention.context_right", &audio.ContextRight},
		metadataIntField{"clip.audio.conv_kernel_size", &audio.ConvKernel},
		metadataIntField{"clip.audio.output_projection_dim", &audio.OutputProjDim},
		metadataIntField{"clip.audio.projection_dim", &audio.ProjectionDim},
		metadataIntField{"clip.audio.num_mel_bins", &audio.MelBins},
		metadataIntField{"clip.audio.fft_length", &audio.FFTLength},
		metadataIntField{"clip.audio.frame_length", &audio.FrameLength},
		metadataIntField{"clip.audio.hop_length", &audio.HopLength},
		metadataIntField{"clip.audio.sample_rate", &audio.SampleRate},
	); err != nil {
		return spec, err
	}
	floatFields := []struct {
		key    string
		target *float32
	}{
		{"clip.vision.attention.layer_norm_epsilon", &vision.RMSNormEpsilon},
		{"clip.vision.rope.freq_base", &vision.RopeFreqBase},
		{"clip.audio.attention.layer_norm_epsilon", &audio.RMSNormEpsilon},
		{"clip.audio.attention.logit_softcapping", &audio.LogitSoftcap},
		{"clip.audio.residual_weight", &audio.ResidualWeight},
		{"clip.audio.min_frequency", &audio.MinFrequency},
		{"clip.audio.max_frequency", &audio.MaxFrequency},
		{"clip.audio.mel_floor", &audio.MelFloor},
	}
	for _, field := range floatFields {
		value, err := metadataFloat32(file, field.key)
		if err != nil {
			return spec, err
		}
		*field.target = value
	}
	stringFields := []struct {
		key    string
		target *string
	}{
		{"clip.vision.hidden_activation", &vision.HiddenActivation},
		{"clip.audio.hidden_act", &audio.HiddenActivation},
	}
	for _, field := range stringFields {
		value, err := metadataString(file, field.key)
		if err != nil {
			return spec, err
		}
		*field.target = value
	}
	inputScale, err := metadataFloat32Array(file, "clip.vision.input_scale", 3)
	if err != nil {
		return spec, err
	}
	inputBias, err := metadataFloat32Array(file, "clip.vision.input_bias", 3)
	if err != nil {
		return spec, err
	}
	copy(vision.InputScale[:], inputScale)
	copy(vision.InputBias[:], inputBias)
	subChannels, err := metadataInt32ArrayValues(file, "clip.audio.subsampling_conv_channels")
	if err != nil {
		return spec, err
	}
	audio.SubChannels = subChannels
	if err := spec.validate(); err != nil {
		return spec, err
	}
	return spec, nil
}

func metadataInt32ArrayValues(file *gguf.File, key string) ([]int, error) {
	value, ok := file.MetadataValue(key)
	if !ok || value.Type != gguf.ValueTypeArray || value.ArrayType != gguf.ValueTypeInt32 {
		return nil, fmt.Errorf("projector: metadata %q must be int32 array", key)
	}
	encoded, ok := value.Data.([]int32)
	if !ok || len(encoded) == 0 {
		return nil, fmt.Errorf("projector: metadata %q has invalid storage", key)
	}
	result := make([]int, len(encoded))
	for index, item := range encoded {
		if item <= 0 {
			return nil, fmt.Errorf("projector: metadata %q entry %d is not positive", key, index)
		}
		result[index] = int(item)
	}
	return result, nil
}

func (s Gemma4TowerSpec) validate() error {
	vision, audio := s.Vision, s.Audio
	if err := vision.validate(); err != nil {
		return err
	}
	if audio.Layers <= 0 || audio.Hidden <= 0 || audio.Heads <= 0 || audio.Hidden%audio.Heads != 0 ||
		audio.Intermediate <= 0 || audio.ConvKernel <= 0 || len(audio.SubChannels) == 0 ||
		audio.ChunkSize <= 0 || audio.ContextLeft < 0 || audio.ContextRight < 0 ||
		audio.OutputProjDim <= 0 || audio.ProjectionDim <= 0 || audio.MelBins <= 0 ||
		audio.FFTLength <= 0 || audio.FrameLength <= 0 || audio.HopLength <= 0 ||
		audio.SampleRate <= 0 || audio.LogitSoftcap <= 0 || audio.ResidualWeight <= 0 ||
		audio.RMSNormEpsilon <= 0 || audio.MinFrequency < 0 || audio.MaxFrequency <= audio.MinFrequency ||
		audio.MelFloor <= 0 || audio.HiddenActivation == "" {
		return fmt.Errorf("projector: invalid Gemma 4 audio tower metadata: %+v", audio)
	}
	return nil
}

func (vision Gemma4VisionTowerSpec) validate() error {
	if vision.Layers <= 0 || vision.Hidden <= 0 || vision.Heads <= 0 || vision.KVHeads <= 0 ||
		vision.HeadDim <= 0 || vision.HeadDim%4 != 0 || vision.Heads%vision.KVHeads != 0 ||
		vision.Intermediate <= 0 || vision.PatchSize <= 0 ||
		vision.PoolKernel <= 0 || vision.PositionCount <= 0 || vision.ProjectionDim <= 0 ||
		vision.MaxImageTokens <= 0 || vision.MaxVideoTokens <= 0 ||
		vision.RMSNormEpsilon <= 0 || vision.RopeFreqBase <= 0 || vision.HiddenActivation == "" {
		return fmt.Errorf("projector: invalid Gemma 4 vision tower metadata: %+v", vision)
	}
	for channel := range vision.InputScale {
		if vision.InputScale[channel] == 0 || !finite32(vision.InputScale[channel]) || !finite32(vision.InputBias[channel]) {
			return fmt.Errorf("projector: invalid Gemma 4 vision input affine channel %d", channel)
		}
	}
	return nil
}

// gemma4TowerClippedLinear: clipped-linear weights carry four rank-1 F32
// calibration scalars beside the BF16 weight.
func gemma4TowerClippedLinear(required map[string][]uint64, base string, shape []uint64) {
	required[base+".weight"] = shape
	for _, scalar := range []string{"input_min", "input_max", "output_min", "output_max"} {
		required[base+"."+scalar] = []uint64{1}
	}
}

func validateGemma4TowerCatalog(file *gguf.File, spec Gemma4TowerSpec) ([]string, error) {
	vision, audio := spec.Vision, spec.Audio
	hidden := uint64(vision.Hidden)
	patchPixels := uint64(vision.PatchSize * vision.PatchSize * 3)
	queryWidth := uint64(vision.Heads * vision.HeadDim)
	kvWidth := uint64(vision.KVHeads * vision.HeadDim)
	inter := uint64(vision.Intermediate)
	required := map[string][]uint64{
		visionPatchWeightTensor:      {patchPixels, hidden},
		visionPositionWeightTensor:   {hidden, uint64(vision.PositionCount), 2},
		"mm.input_projection.weight": {hidden, uint64(vision.ProjectionDim)},
	}
	for layer := 0; layer < vision.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for _, norm := range []string{"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"} {
			required[prefix+norm+".weight"] = []uint64{hidden}
		}
		required[prefix+"attn_q_norm.weight"] = []uint64{uint64(vision.HeadDim)}
		required[prefix+"attn_k_norm.weight"] = []uint64{uint64(vision.HeadDim)}
		gemma4TowerClippedLinear(required, prefix+"attn_q", []uint64{hidden, queryWidth})
		gemma4TowerClippedLinear(required, prefix+"attn_k", []uint64{hidden, kvWidth})
		gemma4TowerClippedLinear(required, prefix+"attn_v", []uint64{hidden, kvWidth})
		gemma4TowerClippedLinear(required, prefix+"attn_output", []uint64{queryWidth, hidden})
		gemma4TowerClippedLinear(required, prefix+"ffn_gate", []uint64{hidden, inter})
		gemma4TowerClippedLinear(required, prefix+"ffn_up", []uint64{hidden, inter})
		gemma4TowerClippedLinear(required, prefix+"ffn_down", []uint64{inter, hidden})
	}
	audioHidden := uint64(audio.Hidden)
	audioInter := uint64(audio.Intermediate)
	for layer := 0; layer < audio.Layers; layer++ {
		prefix := fmt.Sprintf("a.blk.%d.", layer)
		for _, norm := range []string{
			"attn_norm", "post_attention_norm", "out_norm",
			"ffn1_norm", "ffn1_post_norm", "ffn2_norm", "ffn2_post_norm",
			"conv_pre_norm", "conv_norm",
		} {
			required[prefix+norm+".weight"] = []uint64{audioHidden}
		}
		required[prefix+"attn_per_dim_scale.weight"] = []uint64{uint64(audio.Hidden / audio.Heads)}
		required[prefix+"attn_rel_k.weight"] = []uint64{audioHidden, audioHidden}
		required[prefix+"conv_dw.weight"] = []uint64{uint64(audio.ConvKernel), 1, audioHidden}
		gemma4TowerClippedLinear(required, prefix+"attn_q", []uint64{audioHidden, audioHidden})
		gemma4TowerClippedLinear(required, prefix+"attn_k", []uint64{audioHidden, audioHidden})
		gemma4TowerClippedLinear(required, prefix+"attn_v", []uint64{audioHidden, audioHidden})
		gemma4TowerClippedLinear(required, prefix+"attn_output", []uint64{audioHidden, audioHidden})
		gemma4TowerClippedLinear(required, prefix+"ffn1_up", []uint64{audioHidden, audioInter})
		gemma4TowerClippedLinear(required, prefix+"ffn1_down", []uint64{audioInter, audioHidden})
		gemma4TowerClippedLinear(required, prefix+"ffn2_up", []uint64{audioHidden, audioInter})
		gemma4TowerClippedLinear(required, prefix+"ffn2_down", []uint64{audioInter, audioHidden})
		gemma4TowerClippedLinear(required, prefix+"conv_start", []uint64{audioHidden, 2 * audioHidden})
		gemma4TowerClippedLinear(required, prefix+"conv_end", []uint64{audioHidden, audioHidden})
	}
	inputChannels := uint64(1)
	for index, channels := range audio.SubChannels {
		name := fmt.Sprintf("a.conv.%d.weight", index)
		info, ok := file.Tensor(name)
		// kernel spatial dims are tensor-owned; channel dims come from config
		if !ok || info.Dimensions != 4 || info.Shape[2] != inputChannels || info.Shape[3] != uint64(channels) {
			return nil, fmt.Errorf(
				"projector: Gemma 4 audio subsample tensor %q is unavailable or has invalid channels", name)
		}
		required[name] = []uint64{info.Shape[0], info.Shape[1], inputChannels, uint64(channels)}
		required[fmt.Sprintf("a.conv.%d.norm.weight", index)] = []uint64{uint64(channels)}
		inputChannels = uint64(channels)
	}
	inputProj, ok := file.Tensor("a.input_proj.weight")
	// flattened subsample width is tensor-owned; output width is the tower hidden
	if !ok || inputProj.Dimensions != 2 || inputProj.Shape[1] != audioHidden || inputProj.Shape[0] == 0 {
		return nil, errors.New("projector: Gemma 4 audio input projection is unavailable or invalid")
	}
	required["a.input_proj.weight"] = []uint64{inputProj.Shape[0], audioHidden}
	required["a.output_proj.weight"] = []uint64{audioHidden, uint64(audio.OutputProjDim)}
	required["a.output_proj.bias"] = []uint64{uint64(audio.OutputProjDim)}
	required["mm.a.input_projection.weight"] = []uint64{uint64(audio.OutputProjDim), uint64(audio.ProjectionDim)}
	return validateProjectorTensorCatalog(file, required)
}
