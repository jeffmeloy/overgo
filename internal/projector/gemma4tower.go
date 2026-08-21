package projector

// Tower projector contract: metadata and tensor inventory.

import (
	"errors"
	"fmt"

	"overgo/internal/checked"
	"overgo/internal/gguf"
	"overgo/internal/media"
	"overgo/internal/tensor"
	"overgo/internal/tensorcatalog"
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
	InputScale       [media.RGBChannels]float32
	InputBias        [media.RGBChannels]float32
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
	audioPlan *audioFrontendPlan
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
	inputScale, err := metadataFloat32Array(file, "clip.vision.input_scale", media.RGBChannels)
	if err != nil {
		return spec, err
	}
	inputBias, err := metadataFloat32Array(file, "clip.vision.input_bias", media.RGBChannels)
	if err != nil {
		return spec, err
	}
	copy(vision.InputScale[:], inputScale)
	copy(vision.InputBias[:], inputBias)
	subChannels, err := metadataInts(file, "clip.audio.subsampling_conv_channels", tensorRequired, true)
	if err != nil {
		return spec, err
	}
	audio.SubChannels = subChannels
	if err := spec.validate(); err != nil {
		return spec, err
	}
	return spec, nil
}

func (s Gemma4TowerSpec) validate() error {
	vision, audio := s.Vision, s.Audio
	if err := vision.validate(); err != nil {
		return err
	}
	_, audioHeadsOK := checked.DivExactInt(audio.Hidden, audio.Heads)
	if !checked.PositiveInts(
		audio.Layers, audio.Hidden, audio.Heads, audio.Intermediate, audio.ConvKernel, len(audio.SubChannels),
		audio.ChunkSize, audio.OutputProjDim, audio.ProjectionDim, audio.MelBins,
		audio.FFTLength, audio.FrameLength, audio.HopLength, audio.SampleRate,
	) || !checked.NonNegativeInts(audio.ContextLeft, audio.ContextRight) || !audioHeadsOK ||
		audio.FFTLength < audio.FrameLength ||
		!checked.PositiveFinite32(audio.LogitSoftcap) || !checked.PositiveFinite32(audio.ResidualWeight) ||
		!checked.PositiveFinite32(audio.RMSNormEpsilon) || !checked.NonNegativeFinite32(audio.MinFrequency) ||
		!checked.Finite32(audio.MaxFrequency) || audio.MaxFrequency <= audio.MinFrequency ||
		audio.MaxFrequency > float32(audio.SampleRate)/tensor.PairedExtent ||
		!checked.PositiveFinite32(audio.MelFloor) || audio.HiddenActivation == "" {
		return fmt.Errorf("projector: invalid Gemma 4 audio tower metadata: %+v", audio)
	}
	return nil
}

func (vision Gemma4VisionTowerSpec) validate() error {
	_, headAxesOK := checked.DivExactInt(vision.HeadDim, tensor.PairedExtent*tensor.PairedExtent)
	_, groupedHeadsOK := checked.DivExactInt(vision.Heads, vision.KVHeads)
	if !checked.PositiveInts(
		vision.Layers, vision.Hidden, vision.Heads, vision.KVHeads, vision.HeadDim, vision.Intermediate,
		vision.PatchSize, vision.PoolKernel, vision.PositionCount, vision.ProjectionDim,
		vision.MaxImageTokens, vision.MaxVideoTokens,
	) || !headAxesOK || !groupedHeadsOK || !checked.PositiveFinite32(vision.RMSNormEpsilon) ||
		!checked.PositiveFinite32(vision.RopeFreqBase) || vision.HiddenActivation == "" {
		return fmt.Errorf("projector: invalid Gemma 4 vision tower metadata: %+v", vision)
	}
	for channel := range vision.InputScale {
		if !checked.Nonzero(vision.InputScale[channel]) || !checked.Finite32(vision.InputScale[channel]) || !checked.Finite32(vision.InputBias[channel]) {
			return fmt.Errorf("projector: invalid Gemma 4 vision input affine channel %d", channel)
		}
	}
	return nil
}

// addClippedLinearCatalog: weight plus scalar clamp bounds.
func addClippedLinearCatalog(required map[string][]uint64, base string, shape []uint64) {
	required[base+".weight"] = shape
	for _, scalar := range []string{"input_min", "input_max", "output_min", "output_max"} {
		required[base+"."+scalar] = []uint64{tensor.SingletonExtent}
	}
}

func validateGemma4TowerCatalog(file *gguf.File, spec Gemma4TowerSpec) ([]string, error) {
	vision, audio := spec.Vision, spec.Audio
	hidden := uint64(vision.Hidden)
	patchPixels := uint64(vision.PatchSize * vision.PatchSize * media.RGBChannels)
	queryWidth := uint64(vision.Heads * vision.HeadDim)
	kvWidth := uint64(vision.KVHeads * vision.HeadDim)
	inter := uint64(vision.Intermediate)
	required := map[string][]uint64{
		visionPatchWeightTensor:    {patchPixels, hidden},
		visionPositionWeightTensor: {hidden, uint64(vision.PositionCount), tensor.PairedExtent},
		multimodalInputProjection:  {hidden, uint64(vision.ProjectionDim)},
	}
	for layer := 0; layer < vision.Layers; layer++ {
		prefix := fmt.Sprintf("v.blk.%d.", layer)
		for _, norm := range []string{"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"} {
			required[prefix+norm+".weight"] = []uint64{hidden}
		}
		required[prefix+"attn_q_norm.weight"] = []uint64{uint64(vision.HeadDim)}
		required[prefix+"attn_k_norm.weight"] = []uint64{uint64(vision.HeadDim)}
		addClippedLinearCatalog(required, prefix+"attn_q", []uint64{hidden, queryWidth})
		addClippedLinearCatalog(required, prefix+"attn_k", []uint64{hidden, kvWidth})
		addClippedLinearCatalog(required, prefix+"attn_v", []uint64{hidden, kvWidth})
		addClippedLinearCatalog(required, prefix+"attn_output", []uint64{queryWidth, hidden})
		addClippedLinearCatalog(required, prefix+"ffn_gate", []uint64{hidden, inter})
		addClippedLinearCatalog(required, prefix+"ffn_up", []uint64{hidden, inter})
		addClippedLinearCatalog(required, prefix+"ffn_down", []uint64{inter, hidden})
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
		required[prefix+"conv_dw.weight"] = []uint64{uint64(audio.ConvKernel), tensor.SingletonExtent, audioHidden}
		addClippedLinearCatalog(required, prefix+"attn_q", []uint64{audioHidden, audioHidden})
		addClippedLinearCatalog(required, prefix+"attn_k", []uint64{audioHidden, audioHidden})
		addClippedLinearCatalog(required, prefix+"attn_v", []uint64{audioHidden, audioHidden})
		addClippedLinearCatalog(required, prefix+"attn_output", []uint64{audioHidden, audioHidden})
		addClippedLinearCatalog(required, prefix+"ffn1_up", []uint64{audioHidden, audioInter})
		addClippedLinearCatalog(required, prefix+"ffn1_down", []uint64{audioInter, audioHidden})
		addClippedLinearCatalog(required, prefix+"ffn2_up", []uint64{audioHidden, audioInter})
		addClippedLinearCatalog(required, prefix+"ffn2_down", []uint64{audioInter, audioHidden})
		addClippedLinearCatalog(required, prefix+"conv_start", []uint64{audioHidden, tensor.PairedExtent * audioHidden})
		addClippedLinearCatalog(required, prefix+"conv_end", []uint64{audioHidden, audioHidden})
	}
	inputChannels := uint64(tensor.SingletonExtent)
	for index, channels := range audio.SubChannels {
		name := fmt.Sprintf("a.conv.%d.weight", index)
		info, ok := file.Tensor(name)
		if !ok {
			return nil, fmt.Errorf("projector: Gemma 4 audio subsample tensor %q is unavailable", name)
		}
		if err := tensorcatalog.ValidateInfo(info, tensorcatalog.Requirement{Rank: tensor.MaxDimensions, NonEmpty: true}); err != nil {
			return nil, fmt.Errorf("projector: Gemma 4 audio subsample tensor: %w", err)
		}
		if err := tensorcatalog.ValidateRelations(info, nil, []tensorcatalog.FixedAxis{
			{Axis: tensor.PairedExtent, Extent: inputChannels}, {Axis: tensor.TripleExtent, Extent: uint64(channels)},
		}); err != nil {
			return nil, fmt.Errorf(
				"projector: Gemma 4 audio subsample tensor %q: %w", name, err)
		}
		required[name] = []uint64{info.Shape[tensor.FirstOffset], info.Shape[tensor.SingletonExtent], inputChannels, uint64(channels)}
		required[fmt.Sprintf("a.conv.%d.norm.weight", index)] = []uint64{uint64(channels)}
		inputChannels = uint64(channels)
	}
	inputProj, ok := file.Tensor("a.input_proj.weight")
	if !ok {
		return nil, errors.New("projector: Gemma 4 audio input projection is unavailable")
	}
	if err := tensorcatalog.ValidateInfo(inputProj, tensorcatalog.Requirement{Rank: tensor.PairedExtent, NonEmpty: true}); err != nil {
		return nil, fmt.Errorf("projector: Gemma 4 audio input projection: %w", err)
	}
	if err := tensorcatalog.ValidateRelations(inputProj, nil,
		[]tensorcatalog.FixedAxis{{Axis: tensor.SingletonExtent, Extent: audioHidden}}); err != nil {
		return nil, fmt.Errorf("projector: Gemma 4 audio input projection: %w", err)
	}
	required["a.input_proj.weight"] = []uint64{inputProj.Shape[tensor.FirstOffset], audioHidden}
	required["a.output_proj.weight"] = []uint64{audioHidden, uint64(audio.OutputProjDim)}
	required["a.output_proj.bias"] = []uint64{uint64(audio.OutputProjDim)}
	required["mm.a.input_projection.weight"] = []uint64{uint64(audio.OutputProjDim), uint64(audio.ProjectionDim)}
	return validateProjectorTensorCatalog(file, required)
}
