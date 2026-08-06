package model

import (
	"fmt"

	"llamacpp2go/internal/tensor"
)

type profileOrdinal interface {
	~uint8 | ~uint16 | ~uint32 | ~uint64
}

const (
	allArchitectureCapabilities = (ArchitectureDiscreteImageTokens << 1) - 1
	allExpertSupplements        = (expertSupplementDenseFFN << 1) - 1
)

// ValidateArchitectureProfile checks persisted policy domains and relationships.
func ValidateArchitectureProfile(profile ArchitectureProfile) error {
	if err := ValidateArchitectureName(profile.Name); err != nil {
		return err
	}
	checks := []error{
		validateProfileOrdinal("Family", profile.Family, ArchitectureFamilyDraft),
		validateProfileOrdinal("GraphFamily", profile.GraphFamily, ArchitectureFamilyDraft),
		validateProfileOrdinal("CatalogFamily", profile.CatalogFamily, ArchitectureFamilyDraft),
		validateProfileOrdinal("DraftKind", profile.DraftKind, DraftCohere2MTP),
		validateProfileOrdinal("Forward", profile.Forward, ForwardT5),
		validateProfileOrdinal("OutputNorm", profile.OutputNorm, OutputNormTokenEmbedding),
		validateProfileOrdinal("Normalization", profile.Normalization, NormalizationWeightOnlyLayer),
		validateProfileOrdinal("Position", profile.Position, PositionNormal),
		validateProfileOrdinal("Residual", profile.Residual, ResidualParallel),
		validateProfileOrdinal("FeedForward", profile.FeedForward, FeedForwardXIELU),
		validateProfileOrdinal("Attention", profile.Attention, AttentionLFM2),
		validateProfileOrdinal("Overrides", profile.Overrides, EmbeddingOverrideDeepstackBase),
		validateProfileOrdinal("Deepstack", profile.Deepstack, DeepstackSequentialAfter),
		validateProfileOrdinal("AttentionBlocks", profile.AttentionBlocks, AttentionBlocksUncached),
		validateProfileOrdinal("Auxiliary", profile.Auxiliary, AuxiliaryDSATopK),
		validateProfileOrdinal("Temperature", profile.Temperature, AttentionTemperatureNoRoPE),
		validateProfileOrdinal("PostNormLayout", profile.PostNormLayout, PostNormLayoutGrok),
		validateProfileOrdinal("FFNNormLayout", profile.FFNNormLayout, FeedForwardNormLayoutPostAttention),
		validateProfileOrdinal("Block", profile.Block, BlockDeepSeek4),
		validateProfileOrdinal("RecurrentBlock", profile.RecurrentBlock, BlockDeepSeek4),
		validateProfileOrdinal("Cache", profile.Cache, CacheDeepSeek4),
		validateProfileOrdinal("RecurrentCache", profile.RecurrentCache, CacheDeepSeek4),
		validateProfileOrdinal("CacheFallback", profile.CacheFallback, CacheFallbackMissingKV),
		validateProfileOrdinal("DenseGraph", profile.DenseGraph, DenseGraphRWKV7),
		validateProfileOrdinal("DenseStages.QK.Projection", profile.DenseStages.QK.Projection, qkNormRMS),
		validateProfileOrdinal("DenseStages.QK.Heads", profile.DenseStages.QK.Heads, qkNormRMS),
		validateProfileOrdinal("DenseStages.QK.PostRotary", profile.DenseStages.QK.PostRotary, qkNormRMS),
		validateProfileOrdinal("DenseStages.AttentionGate", profile.DenseStages.AttentionGate, attentionGateSigmoid),
		validateProfileOrdinal("DenseStages.Residual", profile.DenseStages.Residual, residualGPTOSS),
		validateProfileOrdinal("DenseStages.QueryScale", profile.DenseStages.QueryScale, queryScalePolicyGemma),
		validateProfileOrdinal("DenseWeights.BiasCatalog", profile.DenseWeights.BiasCatalog, denseBiasCatalogJais),
		validateProfileOrdinal("ModelCatalog.PositionEmbedding", profile.ModelCatalog.PositionEmbedding, positionEmbeddingOptional),
		validateProfileOrdinal("ModelCatalog.TokenNorm", profile.ModelCatalog.TokenNorm, tokenNormAffine),
		validateProfileOrdinal("Rotary.Kind", profile.Rotary.Kind, rotaryPolicyGemma),
		validateProfileOrdinal("Rotary.MultiAxis", profile.Rotary.MultiAxis, multiAxisRotaryWithSections),
		validateProfileOrdinal("Rotary.Usage", profile.Rotary.Usage, RotaryUsageSlidingOnly),
		validateProfileOrdinal("AttentionGraph.QwenGDN", profile.AttentionGraph.QwenGDN, qwenGDNRepeatInterleave),
		validateProfileOrdinal("Experts.Composition", profile.Experts.Composition, expertArctic),
		validateProfileOrdinal("Experts.Condition", profile.Experts.Condition, expertCompositionUnlessSigmoidWithoutShared),
		validateProfileOrdinal("Experts.Normalization", profile.Experts.Normalization, expertNormalizeNever),
		validateProfileOrdinal("Experts.Routing", profile.Experts.Routing, expertRouteSelectedSoftmax),
		validateProfileOrdinal("Experts.Activation", profile.Experts.Activation, tensor.MoEActivationReLUSquared),
		validateProfileOrdinal("Experts.Catalog", profile.Experts.Catalog, expertCatalogAfterDenseExceptNextN),
		validateProfileOrdinal("Experts.BiasCatalog", profile.Experts.BiasCatalog, expertBiasCatalogRequiredF32),
		validateProfileOrdinal("Experts.SharedCatalog", profile.Experts.SharedCatalog, sharedExpertCatalogGated),
		validateProfileOrdinal("Metadata.FeedForward", profile.Metadata.FeedForward, metadataHybridLayers),
		validateProfileOrdinal("Metadata.Heads", profile.Metadata.Heads, metadataHybridLayers),
		validateProfileOrdinal("Metadata.KVHeads", profile.Metadata.KVHeads, metadataHybridLayers),
		validateProfileOrdinal("Validation.BaseRotary", profile.Validation.BaseRotary, BaseRotaryValidationHalfWidth),
		validateProfileOrdinal("Validation.Encoder", profile.Validation.Encoder, EncoderValidationNomicBERTMoE),
		validateProfileOrdinal("Validation.Attention", profile.Validation.Attention, AttentionValidationRefact),
		validateProfileOrdinal("Validation.MLA", profile.Validation.MLA, MLAValidationMiniCPM3),
		validateProfileOrdinal("Validation.Recurrent", profile.Validation.Recurrent, RecurrentValidationNemotronHMoE),
		validateProfileOrdinal("Validation.Hybrid", profile.Validation.Hybrid, HybridValidationLFM2MoE),
		validateProfileOrdinal("EncoderGraph.Kind", profile.EncoderGraph.Kind, encoderGraphT5Encoder),
		validateProfileOrdinal("MLAVariant", profile.MLAVariant, mlaVariantKimi),
		validateProfileOrdinal("Cadence.Recurrent", profile.Cadence.Recurrent, recurrentCadenceAttentionInterval),
		validateProfileOrdinal("Cadence.MoE", profile.Cadence.MoE, moeCadenceAfterDense),
		validateProfileOrdinal("Cadence.Sliding", profile.Cadence.Sliding, slidingCadenceExceptLast),
		validateProfileOrdinal("Runtime.EmbeddingScale", profile.Runtime.EmbeddingScale, EmbeddingScaleSqrtWidth),
		validateProfileOrdinal("Runtime.LogitScale", profile.Runtime.LogitScale, LogitScaleDirect),
		validateProfileOrdinal("Runtime.NormalizationPlacement", profile.Runtime.NormalizationPlacement, NormalizationPlacementPostOnly),
		validateProfileOrdinal("Runtime.NormalizationBias", profile.Runtime.NormalizationBias, NormalizationBiasAlways),
		validateProfileOrdinal("Runtime.NormalizationFallback", profile.Runtime.NormalizationFallback, NormalizationFallbackRMSWithoutLayerEpsilon),
	}
	for _, err := range checks {
		if err != nil {
			return fmt.Errorf("architecture profile %q: %w", profile.Name, err)
		}
	}
	if unknown := profile.Capabilities &^ allArchitectureCapabilities; unknown != 0 {
		return fmt.Errorf("architecture profile %q: Capabilities has unknown bits %#x", profile.Name, unknown)
	}
	if unknown := profile.Experts.SupplementalCatalog &^ allExpertSupplements; unknown != 0 {
		return fmt.Errorf("architecture profile %q: Experts.SupplementalCatalog has unknown bits %#x", profile.Name, unknown)
	}
	if profile.AttentionGraph.QwenGDN != qwenGDNNone && profile.Attention != AttentionQwenGDN {
		return fmt.Errorf("architecture profile %q: Qwen GDN graph requires Qwen GDN attention", profile.Name)
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
