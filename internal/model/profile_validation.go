package model

import (
	"fmt"

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
	if (profile.DraftKind == DraftSingleCatalog) != (profile.DraftQueryCopies > tensor.FirstOffset) {
		return fmt.Errorf("architecture profile %q: draft query packing is inconsistent", profile.Name)
	}
	if err := ValidateArchitectureName(profile.Name); err != nil {
		return err
	}
	checks := []error{
		validateProfileOrdinal("DraftKind", profile.DraftKind, DraftOptionalSingleCatalog),
		validateProfileOrdinal("OutputNorm", profile.OutputNorm, OutputNormTokenEmbedding),
		validateProfileOrdinal("Normalization", profile.Normalization, NormalizationMAD),
		validateProfileOrdinal("Position", profile.Position, PositionLearnedAbsolute),
		validateProfileOrdinal("Residual", profile.Residual, ResidualParallel),
		validateProfileOrdinal("FeedForward", profile.FeedForward, FeedForwardReLU),
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
		validateProfileOrdinal("MetadataDefaults.SharedExpert", profile.MetadataDefaults.SharedExpert, SharedExpertDefaultExpertProduct),
		validateProfileOrdinal("Validation.BaseRotary", profile.Validation.BaseRotary, BaseRotaryValidationPaired),
		validateProfileOrdinal("Validation.ExpertMetadata", profile.Validation.ExpertMetadata, ExpertMetadataWhenDeclared),
		validateProfileOrdinal("Validation.Encoder", profile.Validation.Encoder, EncoderValidationRotaryPeriodicExperts),
		validateProfileOrdinal("Validation.Attention", profile.Validation.Attention, AttentionValidationOptionalExperts),
		validateProfileOrdinal("Validation.MLA", profile.Validation.MLA, MLAValidationScaledLatent),
		validateProfileOrdinal("Validation.Recurrent", profile.Validation.Recurrent, RecurrentValidationScheduledStateSpaceExperts),
		validateProfileOrdinal("Validation.Hybrid", profile.Validation.Hybrid, HybridValidationAlternatingShortConvolutionExperts),
		validateProfileOrdinal("EncoderOperator", profile.EncoderOperator, encoderOperatorRelativeEncoder),
		validateProfileOrdinal("LatentAttention", profile.LatentAttention, latentAttentionNoRoPE),
		validateProfileOrdinal("Cadence.Recurrent", profile.Cadence.Recurrent, recurrentCadenceAttentionInterval),
		validateProfileOrdinal("Cadence.MoE", profile.Cadence.MoE, moeCadenceAfterDense),
		validateProfileOrdinal("Cadence.Sliding", profile.Cadence.Sliding, slidingCadenceExceptLast),
		validateProfileOrdinal("Runtime.EmbeddingScale", profile.Runtime.EmbeddingScale, EmbeddingScaleSqrtWidth),
		validateProfileOrdinal("Runtime.LogitScale", profile.Runtime.LogitScale, LogitScaleDirect),
		validateProfileOrdinal("Runtime.NormalizationPlacement", profile.Runtime.NormalizationPlacement, NormalizationPlacementPostOnly),
		validateProfileOrdinal("Runtime.NormalizationBias", profile.Runtime.NormalizationBias, NormalizationBiasAlways),
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
	if unknown := profile.Capabilities &^ allArchitectureCapabilities; unknown != tensor.FirstOffset {
		return fmt.Errorf("architecture profile %q: Capabilities has unknown bits %#x", profile.Name, unknown)
	}
	if unknown := profile.Experts.SupplementalCatalog &^ allExpertSupplements; unknown != tensor.FirstOffset {
		return fmt.Errorf("architecture profile %q: Experts.SupplementalCatalog has unknown bits %#x", profile.Name, unknown)
	}
	if unknown := profile.MetadataRead &^ allMetadataReadPolicies; unknown != tensor.FirstOffset {
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
		{"MetadataDefaults.ResidualScalePerSqrtBlock", profile.MetadataDefaults.ResidualScalePerSqrtBlock},
		{"MetadataDefaults.LogitScaleNumerator", profile.MetadataDefaults.LogitScaleNumerator},
		{"MetadataDefaults.AttentionTemperatureScale", profile.MetadataDefaults.AttentionTemperatureScale},
		{"MetadataDefaults.LayerNormEpsilon", profile.MetadataDefaults.LayerNormEpsilon},
		{"MetadataDefaults.QKNormEpsilon", profile.MetadataDefaults.QKNormEpsilon},
		{"MetadataDefaults.MaxALiBiBias", profile.MetadataDefaults.MaxALiBiBias},
		{"MetadataDefaults.SparsityStdMultiplier", profile.MetadataDefaults.SparsityStdMultiplier},
		{"Runtime.Recurrent.HeadNormEpsilon", profile.Runtime.Recurrent.HeadNormEpsilon},
		{"Runtime.Recurrent.KeyNormEpsilon", profile.Runtime.Recurrent.KeyNormEpsilon},
		{"Runtime.Recurrent.PeriodicResidualScale", profile.Runtime.Recurrent.PeriodicResidualScale},
	} {
		if scalar.value < tensor.FirstOffset || !finite(scalar.value) {
			return fmt.Errorf("architecture profile %q: %s must be finite and nonnegative", profile.Name, scalar.name)
		}
	}
	recurrent := profile.Runtime.Recurrent
	switch profile.Validation.Recurrent {
	case RecurrentValidationUngroupedStateSpace, RecurrentValidationStateSpaceAttentionExperts,
		RecurrentValidationGroupedStateSpaceOptionalExperts:
		if recurrent.InnerWidthMultiplier == tensor.FirstOffset {
			return fmt.Errorf("architecture profile %q: recurrent inner-width policy is incomplete", profile.Name)
		}
	case RecurrentValidationTimeMixV6:
		if recurrent.TokenShiftCount == tensor.FirstOffset || !positiveFinite(recurrent.HeadNormEpsilon) || !positiveFinite(recurrent.PeriodicResidualScale) {
			return fmt.Errorf("architecture profile %q: RWKV6 runtime policy is incomplete", profile.Name)
		}
	case RecurrentValidationTimeMixV6SharedKV:
		if recurrent.TokenShiftCount == tensor.FirstOffset || !positiveFinite(recurrent.PeriodicResidualScale) {
			return fmt.Errorf("architecture profile %q: RWKV6-Qwen2 runtime policy is incomplete", profile.Name)
		}
	case RecurrentValidationTimeMixV7Gated, RecurrentValidationTimeMixV7:
		if recurrent.TokenShiftCount == tensor.FirstOffset || !positiveFinite(recurrent.HeadNormEpsilon) || !positiveFinite(recurrent.KeyNormEpsilon) {
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
	if (profile.readsMetadata(MetadataReadALiBi) && !profile.readsMetadata(MetadataReadZeroALiBiDefault) ||
		profile.EncoderOperator.usesALiBiQKNorm()) &&
		!positiveFinite(profile.MetadataDefaults.MaxALiBiBias) {
		return fmt.Errorf("architecture profile %q: ALiBi default is absent", profile.Name)
	}
	if (profile.readsMetadata(MetadataReadSmolLM3NoRoPE) ||
		profile.Validation.Hybrid == HybridValidationSlidingSigmoidExperts) &&
		profile.MetadataDefaults.NoRopeLayerStep == tensor.FirstOffset {
		return fmt.Errorf("architecture profile %q: no-RoPE cadence default is absent", profile.Name)
	}
	if profile.readsMetadata(MetadataReadGLMDSAGating) && profile.Attention != AttentionSparseLatent {
		return fmt.Errorf("architecture profile %q: GLM-DSA metadata requires DSA attention", profile.Name)
	}
	if profile.Validation.QLoRARankOptional && !profile.Has(ArchitectureLatentKVLayout) {
		return fmt.Errorf("architecture profile %q: optional Q-LoRA rank requires latent KV layout", profile.Name)
	}
	if positiveFinite(profile.MetadataDefaults.MaxALiBiBias) && !profile.MetadataDefaults.RopeDisabled &&
		!profile.readsMetadata(MetadataReadALiBi) && !profile.EncoderOperator.usesALiBiQKNorm() {
		return fmt.Errorf("architecture profile %q: ALiBi default requires disabled RoPE", profile.Name)
	}
	if profile.MetadataDefaults.SlidingWindow > tensor.FirstOffset && profile.MetadataDefaults.SlidingPattern < tensor.PairedExtent {
		return fmt.Errorf("architecture profile %q: sliding-window default requires a period", profile.Name)
	}
	if profile.Validation.optionalRopeBase() && !positiveFinite(profile.MetadataDefaults.RopeFrequencyBase) {
		return fmt.Errorf("architecture profile %q: optional RoPE base default is absent", profile.Name)
	}
	if profile.AttentionGraph.GatedDelta != gatedDeltaNone && profile.Attention != AttentionGatedDelta {
		return fmt.Errorf("architecture profile %q: Qwen GDN graph requires Qwen GDN attention", profile.Name)
	}
	defaults := profile.MetadataDefaults
	if (profile.Validation.Hybrid == HybridValidationExtendedRotary || profile.Validation.MLA == MLAValidationScaledLatent) &&
		(!positiveFinite(defaults.EmbeddingScale) || !positiveFinite(defaults.ResidualScalePerSqrtBlock) ||
			!positiveFinite(defaults.LogitScaleNumerator)) {
		return fmt.Errorf("architecture profile %q: scaled runtime defaults are incomplete", profile.Name)
	}
	if profile.Validation.MLA == MLAValidationScaledLatent && (!defaults.OriginalContext || !positiveFinite(defaults.RopeAttentionFactor)) {
		return fmt.Errorf("architecture profile %q: scaled latent position defaults are incomplete", profile.Name)
	}
	if profile.Validation.Hybrid == HybridValidationChunkedExperts &&
		(defaults.SlidingWindow == tensor.FirstOffset || !positiveFinite(defaults.AttentionTemperatureScale) ||
			!finite(defaults.AttentionTemperatureOffset)) {
		return fmt.Errorf("architecture profile %q: chunked attention defaults are incomplete", profile.Name)
	}
	if profile.Forward.Session == ForwardSessionEncoderDecoder && !defaults.DecoderBlocksFromModel {
		return fmt.Errorf("architecture profile %q: decoder block default is absent", profile.Name)
	}
	if profile.Validation.hybridOneOf(
		HybridValidationAlternatingGatedDelta, HybridValidationAlternatingGatedDeltaHybrid, HybridValidationAlternatingGatedDeltaExperts,
	) && defaults.FullAttentionInterval == tensor.FirstOffset {
		return fmt.Errorf("architecture profile %q: full-attention cadence default is absent", profile.Name)
	}
	if profile.Validation.Hybrid == HybridValidationCompressedHyperDraft && defaults.MoELayerStep == tensor.FirstOffset {
		return fmt.Errorf("architecture profile %q: MoE cadence default is absent", profile.Name)
	}
	if profile.Validation.Hybrid == HybridValidationSparseSharedExperts && !defaults.ExpertChunkFromKey {
		return fmt.Errorf("architecture profile %q: expert chunk-width relationship is absent", profile.Name)
	}
	if profile.Validation.hybridOneOf(HybridValidationAlternatingGatedDelta, HybridValidationAlternatingGatedDeltaExperts, HybridValidationSharedExpertProduct) &&
		defaults.SharedExpert != SharedExpertDefaultModelFeedForward {
		return fmt.Errorf("architecture profile %q: model-width shared expert relationship is absent", profile.Name)
	}
	if profile.Validation.Hybrid == HybridValidationMultiHeadDraft && defaults.SharedExpert != SharedExpertDefaultExpertFeedForward {
		return fmt.Errorf("architecture profile %q: expert-width shared expert relationship is absent", profile.Name)
	}
	if profile.Validation.Attention == AttentionValidationRequiredSlidingRotaryExperts && defaults.SharedExpert != SharedExpertDefaultExpertProduct {
		return fmt.Errorf("architecture profile %q: repeated shared expert relationship is absent", profile.Name)
	}
	if profile.Validation.Attention == AttentionValidationSharedKVAlternatingState &&
		(defaults.AlternateStateCount == tensor.FirstOffset || defaults.LowRankResidualWidth == tensor.FirstOffset ||
			defaults.PerLayerEmbeddingWidth == tensor.FirstOffset || defaults.SharedKVStartLayer == tensor.FirstOffset ||
			defaults.SparseLayerCount == tensor.FirstOffset || !positiveFinite(defaults.SparsityStdMultiplier) ||
			profile.Validation.RequiredBlockCount == tensor.FirstOffset || profile.Validation.AlternateBlockCount == tensor.FirstOffset ||
			profile.Validation.RequiredBlockCount == profile.Validation.AlternateBlockCount ||
			profile.Validation.SlidingPeriod < tensor.PairedExtent || !positiveFinite(profile.Runtime.ActivationResidualScale)) {
		return fmt.Errorf("architecture profile %q: alternate-state metadata defaults are incomplete", profile.Name)
	}
	if profile.Has(ArchitecturePerLayerEmbeddings) && !positiveFinite(profile.Runtime.ActivationResidualScale) {
		return fmt.Errorf("architecture profile %q: per-layer residual scale is absent", profile.Name)
	}
	if profile.Validation.Recurrent == RecurrentValidationTargetLayerBlock && defaults.DraftBlockSize == tensor.FirstOffset {
		return fmt.Errorf("architecture profile %q: paired-feature draft block default is absent", profile.Name)
	}
	indexer := profile.Cadence
	if indexer.FullIndexerEveryLayer &&
		(indexer.FullIndexerContext != tensor.FirstOffset || indexer.FullIndexerPrefix != tensor.FirstOffset || indexer.FullIndexerPeriod != tensor.FirstOffset) ||
		!indexer.FullIndexerEveryLayer && indexer.FullIndexerContext > tensor.FirstOffset && indexer.FullIndexerPeriod == tensor.FirstOffset {
		return fmt.Errorf("architecture profile %q: invalid full-indexer cadence", profile.Name)
	}
	if profile.Rotary.MultiAxis != multiAxisRotaryNone && !profile.Has(ArchitectureMultiAxisPositions) {
		return fmt.Errorf("architecture profile %q: multi-axis rotary policy requires multi-axis positions", profile.Name)
	}
	fusedRequirements := ArchitectureRequiresFusedQKV | ArchitectureRequiresFusedQKVBias |
		ArchitectureRejectsOrphanFusedQKVBias
	if profile.Capabilities&fusedRequirements != tensor.FirstOffset && !profile.Has(ArchitectureFusedQKV) {
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
