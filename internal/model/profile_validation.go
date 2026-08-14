package model

import (
	"fmt"
	"math"

	"overgo/internal/tensor"
)

type profileOrdinal interface {
	~uint8 | ~uint16 | ~uint32 | ~uint64
}

const (
	allArchitectureCapabilities = ((ArchitectureDiscreteImageTokens << 1) - 1) &^
		(architectureReservedGEGLU | architectureReservedAltUp)
	allExpertSupplements = (expertSupplementDenseBranch << 1) - 1
)

// ValidateArchitectureProfile checks persisted policy domains and relationships.
func ValidateArchitectureProfile(profile ArchitectureProfile) error {
	if err := ValidateArchitectureName(profile.Name); err != nil {
		return err
	}
	checks := []error{
		validateProfileOrdinal("DraftKind", profile.DraftKind, DraftOptionalSingleCatalog),
		validateProfileOrdinal("OutputNorm", profile.OutputNorm, OutputNormTokenEmbedding),
		validateProfileOrdinal("Normalization", profile.Normalization, NormalizationWeightOnlyLayer),
		validateProfileOrdinal("Position", profile.Position, PositionNormal),
		validateProfileOrdinal("Residual", profile.Residual, ResidualParallel),
		validateProfileOrdinal("FeedForward", profile.FeedForward, FeedForwardGEGLU),
		validateProfileOrdinal("Attention", profile.Attention, AttentionShortConvolution),
		validateProfileOrdinal("Overrides", profile.Overrides, EmbeddingOverrideMappedBase),
		validateProfileOrdinal("Deepstack", profile.Deepstack, DeepstackSequentialAfter),
		validateProfileOrdinal("AttentionBlocks", profile.AttentionBlocks, AttentionBlocksUncached),
		validateProfileOrdinal("Auxiliary", profile.Auxiliary, AuxiliarySparseTopK),
		validateProfileOrdinal("Temperature", profile.Temperature, AttentionTemperatureNoRoPE),
		validateProfileOrdinal("PostNormLayout", profile.PostNormLayout, PostNormLayoutGrok),
		validateProfileOrdinal("FFNNormLayout", profile.FFNNormLayout, FeedForwardNormLayoutPostAttention),
		validateProfileOrdinal("Cache", profile.Cache, CacheCompressedAttention),
		validateProfileOrdinal("RecurrentCache", profile.RecurrentCache, CacheCompressedAttention),
		validateProfileOrdinal("CacheFallback", profile.CacheFallback, CacheFallbackMissingKV),
		validateProfileOrdinal("LayerTopology", profile.LayerTopology, LayerTopologyCompressedHyper),
		validateProfileOrdinal("RecurrentMixer", profile.RecurrentMixer, recurrentMixerKeyedDelta),
		validateProfileOrdinal("DenseStages.QK.Projection", profile.DenseStages.QK.Projection, qkNormRMS),
		validateProfileOrdinal("DenseStages.QK.Heads", profile.DenseStages.QK.Heads, qkNormRMS),
		validateProfileOrdinal("DenseStages.QK.PostRotary", profile.DenseStages.QK.PostRotary, qkNormRMS),
		validateProfileOrdinal("DenseStages.AttentionGate", profile.DenseStages.AttentionGate, attentionGateSigmoid),
		validateProfileOrdinal("DenseStages.Residual", profile.DenseStages.Residual, residualGPTOSS),
		validateProfileOrdinal("DenseStages.QueryScale", profile.DenseStages.QueryScale, queryScalePolicyEmbeddingHead),
		validateProfileOrdinal("DenseWeights.BiasCatalog", profile.DenseWeights.BiasCatalog, denseBiasCatalogJais),
		validateProfileOrdinal("ModelCatalog.PositionEmbedding", profile.ModelCatalog.PositionEmbedding, positionEmbeddingOptional),
		validateProfileOrdinal("ModelCatalog.TokenNorm", profile.ModelCatalog.TokenNorm, tokenNormAffine),
		validateProfileOrdinal("ModelCatalog.Weights", profile.ModelCatalog.Weights, WeightCatalogEncoderDecoder),
		validateProfileOrdinal("Rotary.Kind", profile.Rotary.Kind, rotaryPolicySlidingLinearReset),
		validateProfileOrdinal("Rotary.MultiAxis", profile.Rotary.MultiAxis, multiAxisRotaryWithSections),
		validateProfileOrdinal("Rotary.Usage", profile.Rotary.Usage, RotaryUsageSlidingOnly),
		validateProfileOrdinal("AttentionGraph.GatedDelta", profile.AttentionGraph.GatedDelta, gatedDeltaInterleavedProjections),
		validateProfileOrdinal("Experts.Composition", profile.Experts.Composition, expertDenseRoutedSeparateNorm),
		validateProfileOrdinal("Experts.Condition", profile.Experts.Condition, expertCompositionUnlessSigmoidWithoutShared),
		validateProfileOrdinal("Experts.Normalization", profile.Experts.Normalization, expertNormalizeNever),
		validateProfileOrdinal("Experts.Routing", profile.Experts.Routing, expertRouteSelectedSoftmax),
		validateProfileOrdinal("Experts.Activation", profile.Experts.Activation, tensor.MoEActivationReLUSquared),
		validateProfileOrdinal("Experts.Catalog", profile.Experts.Catalog, expertCatalogAfterDenseExceptDraft),
		validateProfileOrdinal("Experts.BiasCatalog", profile.Experts.BiasCatalog, expertBiasCatalogRequiredF32),
		validateProfileOrdinal("Experts.SharedCatalog", profile.Experts.SharedCatalog, sharedExpertCatalogGated),
		validateProfileOrdinal("Metadata.FeedForward", profile.Metadata.FeedForward, metadataHybridLayers),
		validateProfileOrdinal("Metadata.Heads", profile.Metadata.Heads, metadataHybridLayers),
		validateProfileOrdinal("Metadata.KVHeads", profile.Metadata.KVHeads, metadataHybridLayers),
		validateProfileOrdinal("Validation.BaseRotary", profile.Validation.BaseRotary, BaseRotaryValidationHalfWidth),
		validateProfileOrdinal("Validation.ExpertMetadata", profile.Validation.ExpertMetadata, ExpertMetadataWhenDeclared),
		validateProfileOrdinal("Validation.Encoder", profile.Validation.Encoder, EncoderValidationNomicBERTMoE),
		validateProfileOrdinal("Validation.Attention", profile.Validation.Attention, AttentionValidationRefact),
		validateProfileOrdinal("Validation.MLA", profile.Validation.MLA, MLAValidationMiniCPM3),
		validateProfileOrdinal("Validation.Recurrent", profile.Validation.Recurrent, RecurrentValidationNemotronHMoE),
		validateProfileOrdinal("Validation.Hybrid", profile.Validation.Hybrid, HybridValidationLFM2MoE),
		validateProfileOrdinal("EncoderOperator", profile.EncoderOperator, encoderOperatorRelativeEncoder),
		validateProfileOrdinal("LatentAttention", profile.LatentAttention, latentAttentionNoRoPE),
		validateProfileOrdinal("Cadence.Recurrent", profile.Cadence.Recurrent, recurrentCadenceAttentionInterval),
		validateProfileOrdinal("Cadence.MoE", profile.Cadence.MoE, moeCadenceAfterDense),
		validateProfileOrdinal("Cadence.Sliding", profile.Cadence.Sliding, slidingCadenceExceptLast),
		validateProfileOrdinal("Runtime.EmbeddingScale", profile.Runtime.EmbeddingScale, EmbeddingScaleSqrtWidth),
		validateProfileOrdinal("Runtime.LogitScale", profile.Runtime.LogitScale, LogitScaleDirect),
		validateProfileOrdinal("Runtime.NormalizationPlacement", profile.Runtime.NormalizationPlacement, NormalizationPlacementPostOnly),
		validateProfileOrdinal("Runtime.NormalizationBias", profile.Runtime.NormalizationBias, NormalizationBiasAlways),
		validateProfileOrdinal("Runtime.NormalizationFallback", profile.Runtime.NormalizationFallback, NormalizationFallbackRMSWithoutLayerEpsilon),
		validateProfileOrdinal("MetadataDefaults.RopeDimension", profile.MetadataDefaults.RopeDimension, RopeDimensionDefaultKeyLengthOverride),
	}
	for _, err := range checks {
		if err != nil {
			return fmt.Errorf("architecture profile %q: %w", profile.Name, err)
		}
	}
	if !profile.Forward.valid() {
		return fmt.Errorf("architecture profile %q: invalid forward program", profile.Name)
	}
	if unknown := profile.Capabilities &^ allArchitectureCapabilities; unknown != 0 {
		return fmt.Errorf("architecture profile %q: Capabilities has unknown bits %#x", profile.Name, unknown)
	}
	if unknown := profile.Experts.SupplementalCatalog &^ allExpertSupplements; unknown != 0 {
		return fmt.Errorf("architecture profile %q: Experts.SupplementalCatalog has unknown bits %#x", profile.Name, unknown)
	}
	if unknown := profile.MetadataRead &^ allMetadataReadPolicies; unknown != 0 {
		return fmt.Errorf("architecture profile %q: MetadataRead has unknown bits %#x", profile.Name, unknown)
	}
	for _, scalar := range []struct {
		name  string
		value float32
	}{
		{"MetadataDefaults.AttentionSoftcap", profile.MetadataDefaults.AttentionSoftcap},
		{"MetadataDefaults.AttentionOutputScale", profile.MetadataDefaults.AttentionOutputScale},
		{"MetadataDefaults.EmbeddingScale", profile.MetadataDefaults.EmbeddingScale},
		{"MetadataDefaults.LogitScale", profile.MetadataDefaults.LogitScale},
		{"MetadataDefaults.LayerNormEpsilon", profile.MetadataDefaults.LayerNormEpsilon},
		{"MetadataDefaults.QKNormEpsilon", profile.MetadataDefaults.QKNormEpsilon},
		{"MetadataDefaults.MaxALiBiBias", profile.MetadataDefaults.MaxALiBiBias},
		{"MetadataDefaults.SparsityStdMultiplier", profile.MetadataDefaults.SparsityStdMultiplier},
		{"Runtime.Recurrent.HeadNormEpsilon", profile.Runtime.Recurrent.HeadNormEpsilon},
		{"Runtime.Recurrent.KeyNormEpsilon", profile.Runtime.Recurrent.KeyNormEpsilon},
		{"Runtime.Recurrent.PeriodicResidualScale", profile.Runtime.Recurrent.PeriodicResidualScale},
	} {
		if scalar.value < 0 || math.IsNaN(float64(scalar.value)) || math.IsInf(float64(scalar.value), 0) {
			return fmt.Errorf("architecture profile %q: %s must be finite and nonnegative", profile.Name, scalar.name)
		}
	}
	recurrent := profile.Runtime.Recurrent
	switch profile.Validation.Recurrent {
	case RecurrentValidationRWKV6:
		if recurrent.TokenShiftCount == 0 || recurrent.HeadNormEpsilon <= 0 || recurrent.PeriodicResidualScale <= 0 {
			return fmt.Errorf("architecture profile %q: RWKV6 runtime policy is incomplete", profile.Name)
		}
	case RecurrentValidationRWKV6Qwen2:
		if recurrent.TokenShiftCount == 0 || recurrent.PeriodicResidualScale <= 0 {
			return fmt.Errorf("architecture profile %q: RWKV6-Qwen2 runtime policy is incomplete", profile.Name)
		}
	case RecurrentValidationRWKV7, RecurrentValidationARWKV7:
		if recurrent.TokenShiftCount == 0 || recurrent.HeadNormEpsilon <= 0 || recurrent.KeyNormEpsilon <= 0 {
			return fmt.Errorf("architecture profile %q: RWKV7 runtime policy is incomplete", profile.Name)
		}
	}
	if profile.readsMetadata(MetadataReadVisualSections) &&
		!profile.Has(ArchitectureMultiAxisPositions) {
		return fmt.Errorf("architecture profile %q: visual metadata requires multi-axis positions", profile.Name)
	}
	if profile.readsMetadata(MetadataReadQwen3VLDeepstack) &&
		!profile.readsMetadata(MetadataReadVisualSections) {
		return fmt.Errorf("architecture profile %q: Qwen3-VL deepstack metadata requires visual sections", profile.Name)
	}
	if profile.readsMetadata(MetadataReadZeroALiBiDefault) &&
		!profile.readsMetadata(MetadataReadALiBi) {
		return fmt.Errorf("architecture profile %q: zero ALiBi default requires ALiBi metadata", profile.Name)
	}
	if profile.readsMetadata(MetadataReadGLMDSAGating) && profile.Attention != AttentionSparseLatent {
		return fmt.Errorf("architecture profile %q: GLM-DSA metadata requires DSA attention", profile.Name)
	}
	if profile.Validation.QLoRARankOptional && !profile.Has(ArchitectureLatentKVLayout) {
		return fmt.Errorf("architecture profile %q: optional Q-LoRA rank requires latent KV layout", profile.Name)
	}
	if profile.MetadataDefaults.MaxALiBiBias > 0 && !profile.MetadataDefaults.RopeDisabled {
		return fmt.Errorf("architecture profile %q: ALiBi default requires disabled RoPE", profile.Name)
	}
	if profile.MetadataDefaults.SlidingWindow > 0 && profile.MetadataDefaults.SlidingPattern < 2 {
		return fmt.Errorf("architecture profile %q: sliding-window default requires a period", profile.Name)
	}
	if profile.AttentionGraph.GatedDelta != gatedDeltaNone && profile.Attention != AttentionGatedDelta {
		return fmt.Errorf("architecture profile %q: Qwen GDN graph requires Qwen GDN attention", profile.Name)
	}
	defaults := profile.MetadataDefaults
	if profile.Validation.Attention == AttentionValidationGemma3N &&
		(defaults.AlternateStateCount == 0 || defaults.LowRankResidualWidth == 0 ||
			defaults.PerLayerEmbeddingWidth == 0 || defaults.SharedKVStartLayer == 0 ||
			defaults.SparseLayerCount == 0 || defaults.SparsityStdMultiplier <= 0 ||
			profile.Validation.RequiredBlockCount == 0 || profile.Validation.AlternateBlockCount == 0 ||
			profile.Validation.RequiredBlockCount == profile.Validation.AlternateBlockCount ||
			profile.Validation.SlidingPeriod < 2) {
		return fmt.Errorf("architecture profile %q: alternate-state metadata defaults are incomplete", profile.Name)
	}
	if profile.Validation.Recurrent == RecurrentValidationDFlash && defaults.DraftBlockSize == 0 {
		return fmt.Errorf("architecture profile %q: paired-feature draft block default is absent", profile.Name)
	}
	indexer := profile.Cadence
	if indexer.FullIndexerEveryLayer &&
		(indexer.FullIndexerContext != 0 || indexer.FullIndexerPrefix != 0 || indexer.FullIndexerPeriod != 0) ||
		!indexer.FullIndexerEveryLayer && indexer.FullIndexerContext > 0 && indexer.FullIndexerPeriod == 0 {
		return fmt.Errorf("architecture profile %q: invalid full-indexer cadence", profile.Name)
	}
	if profile.Rotary.MultiAxis != multiAxisRotaryNone && !profile.Has(ArchitectureMultiAxisPositions) {
		return fmt.Errorf("architecture profile %q: multi-axis rotary policy requires multi-axis positions", profile.Name)
	}
	fusedRequirements := ArchitectureRequiresFusedQKV | ArchitectureRequiresFusedQKVBias |
		ArchitectureRejectsOrphanFusedQKVBias
	if profile.Capabilities&fusedRequirements != 0 && !profile.Has(ArchitectureFusedQKV) {
		return fmt.Errorf("architecture profile %q: fused QKV requirements require fused QKV", profile.Name)
	}
	return nil
}

func validateProfileOrdinal[T profileOrdinal](field string, value, last T) error {
	if value > last {
		return fmt.Errorf("%s value %d exceeds supported maximum %d", field, value, last)
	}
	return nil
}
