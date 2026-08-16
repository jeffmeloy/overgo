// Package organ classifies tensors into the eight-axis organ vocabulary:
// modality, role, space, dtype, layout, training role, and objective under one
// schema version. It is classification only -- no retrieval index, no
// persistence; the component catalog and promotion wrappers arrive with their
// consuming rows.
//
// Ported from adaptive_new go/extmodel/organ_schema.go (classifyOrganTensor,
// classifyOrganSpace, classifyOrganModality, validateOrganContract) at schema
// version 2.0. Token rules match the source except two GGUF-vocabulary
// additions the source never sees: "embd" (GGUF token_embd) classifies as
// embedding, and "output" (GGUF attn_output) joins "o"/"out" for attention
// output; both extend recall without changing any source-classifiable name.
package organ

import (
	"slices"
	"strings"
)

const SchemaVersion = "2.0"

type Modality string

const (
	ModalityUnknown    Modality = "unknown"
	ModalityText       Modality = "text"
	ModalityVision     Modality = "vision"
	ModalityAudio      Modality = "audio"
	ModalityVideo      Modality = "video"
	ModalityTabular    Modality = "tabular"
	ModalityTimeSeries Modality = "time_series"
	ModalityMultimodal Modality = "multimodal"
)

type Role string

const (
	RoleUnknown      Role = "unknown"
	RoleEmbedding    Role = "embedding"
	RoleEncoder      Role = "encoder"
	RoleDecoder      Role = "decoder"
	RoleQKV          Role = "qkv"
	RoleAttentionOut Role = "attention_out"
	RoleMLPGate      Role = "mlp_gate"
	RoleMLPUp        Role = "mlp_up"
	RoleMLPDown      Role = "mlp_down"
	RoleNorm         Role = "norm"
	RoleProjector    Role = "projector"
	RoleVAEEncoder   Role = "vae_encoder"
	RoleVAEDecoder   Role = "vae_decoder"
	RoleDenoiser     Role = "denoiser"
	RoleScheduler    Role = "scheduler"
	RoleAdapter      Role = "adapter"
	RoleHead         Role = "head"
)

type Space string

const (
	SpaceUnknown      Space = "unknown"
	SpaceTokenIDs     Space = "token_ids"
	SpaceTextHidden   Space = "text_hidden"
	SpaceVisionHidden Space = "vision_hidden"
	SpaceAudioLatent  Space = "audio_latent"
	SpaceImageLatent  Space = "image_latent"
	SpaceVideoLatent  Space = "video_latent"
	SpaceLogits       Space = "logits"
	SpaceMel          Space = "mel"
	SpacePCM          Space = "pcm"
)

type DType string

const (
	DTypeUnknown DType = "unknown"
	DTypeFP32    DType = "fp32"
	DTypeBF16    DType = "bf16"
	DTypeFP16    DType = "fp16"
	DTypeFP8     DType = "fp8"
	DTypeInt8    DType = "int8"
	DTypeInt4    DType = "int4"
)

type Layout string

const (
	LayoutUnknown      Layout = "unknown"
	LayoutRowMajor     Layout = "row_major"
	LayoutColMajor     Layout = "col_major"
	LayoutQKVPacked    Layout = "qkv_packed"
	LayoutGatedMLPPair Layout = "gated_mlp_pair"
	LayoutSharded      Layout = "sharded"
)

type TrainingRole string

const (
	TrainingRoleUnknown        TrainingRole = "unknown"
	TrainingRoleFrozen         TrainingRole = "frozen"
	TrainingRoleTrainable      TrainingRole = "trainable"
	TrainingRoleAdapter        TrainingRole = "adapter"
	TrainingRoleOptimizerState TrainingRole = "optimizer_state"
	TrainingRoleMomentum       TrainingRole = "momentum"
	TrainingRoleMasterWeight   TrainingRole = "master_weight"
)

type Objective string

const (
	ObjectiveUnknown            Objective = "unknown"
	ObjectiveLMCE               Objective = "lm_ce"
	ObjectiveLatentL2           Objective = "latent_l2"
	ObjectiveFlowMatching       Objective = "flow_matching"
	ObjectiveDiffusionNoisePred Objective = "diffusion_noise_pred"
	ObjectiveCTC                Objective = "ctc"
	ObjectiveContrastive        Objective = "contrastive"
	ObjectiveReconstruction     Objective = "reconstruction"
)

// Contract is the eight-axis classification for one tensor. Empty fields mean
// no typed fact was supplied; explicit "unknown" means classification was
// attempted and intentionally declined.
type Contract struct {
	Modality     Modality     `json:"modality"`
	Role         Role         `json:"role"`
	Space        Space        `json:"space"`
	DType        DType        `json:"dtype"`
	Layout       Layout       `json:"layout"`
	TrainingRole TrainingRole `json:"training_role"`
	Objective    Objective    `json:"objective"`
}

type ContractIssue struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

var (
	modalities = [...]Modality{
		"", ModalityUnknown, ModalityText, ModalityVision, ModalityAudio,
		ModalityVideo, ModalityTabular, ModalityTimeSeries, ModalityMultimodal,
	}
	roles = [...]Role{
		"", RoleUnknown, RoleEmbedding, RoleEncoder, RoleDecoder, RoleQKV,
		RoleAttentionOut, RoleMLPGate, RoleMLPUp, RoleMLPDown, RoleNorm,
		RoleProjector, RoleVAEEncoder, RoleVAEDecoder, RoleDenoiser,
		RoleScheduler, RoleAdapter, RoleHead,
	}
	spaces = [...]Space{
		"", SpaceUnknown, SpaceTokenIDs, SpaceTextHidden, SpaceVisionHidden,
		SpaceAudioLatent, SpaceImageLatent, SpaceVideoLatent, SpaceLogits,
		SpaceMel, SpacePCM,
	}
	dtypes = [...]DType{
		"", DTypeUnknown, DTypeFP32, DTypeBF16, DTypeFP16, DTypeFP8, DTypeInt8, DTypeInt4,
	}
	layouts = [...]Layout{
		"", LayoutUnknown, LayoutRowMajor, LayoutColMajor, LayoutQKVPacked,
		LayoutGatedMLPPair, LayoutSharded,
	}
	trainingRoles = [...]TrainingRole{
		"", TrainingRoleUnknown, TrainingRoleFrozen, TrainingRoleTrainable,
		TrainingRoleAdapter, TrainingRoleOptimizerState, TrainingRoleMomentum,
		TrainingRoleMasterWeight,
	}
	objectives = [...]Objective{
		"", ObjectiveUnknown, ObjectiveLMCE, ObjectiveLatentL2, ObjectiveFlowMatching,
		ObjectiveDiffusionNoisePred, ObjectiveCTC, ObjectiveContrastive, ObjectiveReconstruction,
	}
)

// Classify derives a tensor's contract from its name, dtype, model family,
// declared modality, and optimizer role. Objective stays unknown: it is a
// training-plan fact, not a tensor-name fact.
func Classify(name, dtype, family, modality, optimizerRole string) Contract {
	nameTokens := orderedNameTokens(name)
	role, layout := classifyTensor(nameTokens)
	classifiedModality := classifyModality(modality, append(orderedNameTokens(family), nameTokens...))
	return Contract{
		Modality:     classifiedModality,
		Role:         role,
		Space:        classifySpace(nameTokens, classifiedModality, role),
		DType:        classifyDType(dtype),
		Layout:       layout,
		TrainingRole: classifyTrainingRole(append(orderedNameTokens(optimizerRole), nameTokens...)),
		Objective:    ObjectiveUnknown,
	}
}

func classifyDType(dtype string) DType {
	value := strings.ToLower(strings.TrimSpace(dtype))
	switch {
	case value == "fp32" || value == "f32" || value == "float32":
		return DTypeFP32
	case value == "bf16" || value == "bfloat16":
		return DTypeBF16
	case value == "fp16" || value == "f16" || value == "float16" || value == "half":
		return DTypeFP16
	case strings.HasPrefix(value, "fp8") || strings.HasPrefix(value, "f8") ||
		strings.Contains(value, "e4m3") || strings.Contains(value, "e5m2"):
		return DTypeFP8
	case value == "int8" || value == "i8":
		return DTypeInt8
	case value == "int4" || value == "i4":
		return DTypeInt4
	default:
		return DTypeUnknown
	}
}

func classifyModality(modality string, tokens []string) Modality {
	if typed := parseModality(modality); typed != ModalityUnknown {
		return typed
	}
	switch {
	case tokenHasAny(tokens, "multimodal", "omni"):
		return ModalityMultimodal
	case tokenHasAny(tokens, "audio", "speech", "voice", "tts", "mimi", "codec", "acoustic", "vocoder"):
		return ModalityAudio
	case tokenHasAny(tokens, "video", "frame", "frames", "temporal"):
		return ModalityVideo
	case tokenHasAny(tokens, "vision", "visual", "image", "images", "pixel", "pixels", "patch", "ocr", "sam"):
		return ModalityVision
	case tokenHasAny(tokens, "forecast", "timeseries", "time_series"):
		return ModalityTimeSeries
	case tokenHasAny(tokens, "tabular", "table"):
		return ModalityTabular
	case tokenHasAny(tokens, "text", "token", "tokens", "embed", "language", "lm", "dense", "causal"):
		return ModalityText
	default:
		return ModalityUnknown
	}
}

func classifyTensor(tokens []string) (Role, Layout) {
	layout := LayoutUnknown
	switch {
	case tokenHasAny(tokens, "shard", "sharded"):
		layout = LayoutSharded
	case slices.Contains(tokens, "qkv"):
		layout = LayoutQKVPacked
	case (slices.Contains(tokens, "gate") && slices.Contains(tokens, "up")) ||
		tokenHasAny(tokens, "gateup", "upgate"):
		layout = LayoutGatedMLPPair
	}
	switch {
	case tokenHasAny(tokens, "lora", "adapter"):
		return RoleAdapter, layout
	case slices.Contains(tokens, "vae") && tokenHasAny(tokens, "encoder", "encode"):
		return RoleVAEEncoder, layout
	case slices.Contains(tokens, "vae") && tokenHasAny(tokens, "decoder", "decode"):
		return RoleVAEDecoder, layout
	case tokenHasAny(tokens, "denoiser", "dit", "transformer") && tokenHasAny(tokens, "noise", "diffusion", "flow"):
		return RoleDenoiser, layout
	case tokenHasAny(tokens, "scheduler", "timestep", "timesteps"):
		return RoleScheduler, layout
	case tokenHasAny(tokens, "embed", "embedding", "embeddings", "embd"):
		return RoleEmbedding, layout
	case tokenHasAny(tokens, "qkv") ||
		(tokenHasAny(tokens, "q", "k", "v", "query", "key", "value") && tokenHasAny(tokens, "attn", "attention", "proj", "projection")):
		return RoleQKV, layout
	case tokenHasAny(tokens, "o", "out", "output") && tokenHasAny(tokens, "attn", "attention", "proj", "projection"):
		return RoleAttentionOut, layout
	case slices.Contains(tokens, "gate") && tokenHasAny(tokens, "mlp", "ffn", "proj", "projection"):
		return RoleMLPGate, layout
	case slices.Contains(tokens, "up") && tokenHasAny(tokens, "mlp", "ffn", "proj", "projection"):
		return RoleMLPUp, layout
	case slices.Contains(tokens, "down") && tokenHasAny(tokens, "mlp", "ffn", "proj", "projection"):
		return RoleMLPDown, layout
	case tokenHasAny(tokens, "norm", "ln", "rmsnorm", "layernorm"):
		return RoleNorm, layout
	case tokenHasAny(tokens, "projector", "projection") ||
		(slices.Contains(tokens, "proj") && !tokenHasAny(tokens, "q", "k", "v")):
		return RoleProjector, layout
	case tokenHasAny(tokens, "lmhead", "head", "classifier"):
		return RoleHead, layout
	case tokenHasAny(tokens, "encoder"):
		return RoleEncoder, layout
	case tokenHasAny(tokens, "decoder"):
		return RoleDecoder, layout
	default:
		return RoleUnknown, layout
	}
}

func classifySpace(tokens []string, modality Modality, role Role) Space {
	switch {
	case tokenHasAny(tokens, "pcm", "waveform"):
		return SpacePCM
	case tokenHasAny(tokens, "mel", "spectrogram"):
		return SpaceMel
	case tokenHasAny(tokens, "logits") || role == RoleHead:
		return SpaceLogits
	case tokenHasAny(tokens, "inputids", "tokenids", "input_ids", "token_ids"):
		return SpaceTokenIDs
	case tokenHasAny(tokens, "latent", "latents"):
		switch modality {
		case ModalityAudio:
			return SpaceAudioLatent
		case ModalityVideo:
			return SpaceVideoLatent
		case ModalityVision:
			return SpaceImageLatent
		}
	case modality == ModalityAudio && role == RoleProjector:
		return SpaceAudioLatent
	case modality == ModalityVideo:
		return SpaceVideoLatent
	case modality == ModalityVision:
		return SpaceVisionHidden
	case modality == ModalityText:
		return SpaceTextHidden
	}
	return SpaceUnknown
}

func classifyTrainingRole(tokens []string) TrainingRole {
	switch {
	case tokenHasAny(tokens, "master", "masterweight", "master_weight"):
		return TrainingRoleMasterWeight
	case tokenHasAny(tokens, "momentum", "velocity"):
		return TrainingRoleMomentum
	case tokenHasAny(tokens, "optimizer", "optstate", "optimizer_state"):
		return TrainingRoleOptimizerState
	case tokenHasAny(tokens, "lora", "adapter"):
		return TrainingRoleAdapter
	case tokenHasAny(tokens, "muon", "sign", "trainable"):
		return TrainingRoleTrainable
	case slices.Contains(tokens, "frozen"):
		return TrainingRoleFrozen
	default:
		return TrainingRoleUnknown
	}
}

// Validate reports every contract violation; an empty slice is a valid
// contract. Promotion refusal wrappers belong to the composition rows that
// consume them.
func Validate(c Contract) []ContractIssue {
	var issues []ContractIssue
	if !slices.Contains(modalities[:], c.Modality) || !slices.Contains(roles[:], c.Role) ||
		!slices.Contains(spaces[:], c.Space) || !slices.Contains(dtypes[:], c.DType) ||
		!slices.Contains(layouts[:], c.Layout) || !slices.Contains(trainingRoles[:], c.TrainingRole) ||
		!slices.Contains(objectives[:], c.Objective) {
		issues = append(issues, ContractIssue{Code: "unknown-enum-value", Message: "contract contains a value outside organ schema " + SchemaVersion})
	}
	if c.Space == SpaceAudioLatent && c.Modality != ModalityAudio && c.Modality != ModalityMultimodal {
		issues = append(issues, ContractIssue{Code: "audio-latent-modality", Message: "audio_latent requires audio or multimodal modality"})
	}
	if c.Layout == LayoutQKVPacked && c.Role != RoleQKV {
		issues = append(issues, ContractIssue{Code: "qkv-layout-role", Message: "qkv_packed requires qkv role"})
	}
	if c.Space == SpaceLogits && c.Role != RoleHead && c.Role != RoleDecoder {
		issues = append(issues, ContractIssue{Code: "logits-role", Message: "logits requires head or decoder role"})
	}
	if c.TrainingRole == TrainingRoleOptimizerState && c.Role != "" && c.Role != RoleUnknown {
		issues = append(issues, ContractIssue{Code: "optimizer-state-role", Message: "optimizer_state cannot claim a model operator role"})
	}
	return issues
}

func parseModality(value string) Modality {
	modality := Modality(strings.ToLower(strings.TrimSpace(value)))
	if modality != "" && modality != ModalityUnknown && slices.Contains(modalities[:], modality) {
		return modality
	}
	return ModalityUnknown
}

func orderedNameTokens(name string) []string {
	return strings.FieldsFunc(strings.ToLower(name), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	})
}

func tokenHasAny(tokens []string, values ...string) bool {
	for _, value := range values {
		if slices.Contains(tokens, value) {
			return true
		}
	}
	return false
}
