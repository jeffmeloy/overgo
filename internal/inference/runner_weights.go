package inference

import (
	"slices"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
)

func f32RequiredModelTensors(weights model.Weights) map[string]struct{} {
	result := make(map[string]struct{})
	layers := slices.Clone(weights.Layers)
	for _, draft := range weights.DraftCatalogs() {
		layers = append(layers, draft.Layer)
	}
	for _, layer := range layers {
		for _, info := range []*gguf.TensorInfo{
			layer.SSMConv1D,
			layer.SSMQueryConv,
			layer.SSMKeyConv,
			layer.SSMValueConv,
			layer.ShortConvKernel,
		} {
			if info != nil {
				result[info.Name] = struct{}{}
			}
		}
	}
	return result
}

func selectedModelTensors(file *gguf.File, weights model.Weights) []gguf.TensorInfo {
	names := map[string]struct{}{weights.TokenEmbedding.Name: {}}
	for _, mtp := range weights.DraftCatalogs() {
		for _, info := range []gguf.TensorInfo{mtp.EHProjection, mtp.EmbeddingNorm, mtp.HiddenNorm} {
			names[info.Name] = struct{}{}
		}
		for _, info := range []*gguf.TensorInfo{mtp.TokenEmbedding, mtp.LayerOutputNorm, mtp.OutputNorm, mtp.Output} {
			if info != nil {
				names[info.Name] = struct{}{}
			}
		}
	}
	if weights.PositionEmbedding != nil {
		names[weights.PositionEmbedding.Name] = struct{}{}
	}
	if weights.TokenTypeEmbedding != nil {
		names[weights.TokenTypeEmbedding.Name] = struct{}{}
	}
	if weights.TokenEmbeddingNorm != nil {
		names[weights.TokenEmbeddingNorm.Name] = struct{}{}
	}
	if weights.TokenEmbeddingNormBias != nil {
		names[weights.TokenEmbeddingNormBias.Name] = struct{}{}
	}
	if weights.OutputNorm.Name != "" {
		names[weights.OutputNorm.Name] = struct{}{}
	}
	if weights.EncoderOutputNorm != nil {
		names[weights.EncoderOutputNorm.Name] = struct{}{}
	}
	if weights.OutputNormBias != nil {
		names[weights.OutputNormBias.Name] = struct{}{}
	}
	if weights.Output != nil {
		names[weights.Output.Name] = struct{}{}
	}
	if weights.OutputBias != nil {
		names[weights.OutputBias.Name] = struct{}{}
	}
	if weights.Dense2Output != nil {
		names[weights.Dense2Output.Name] = struct{}{}
	}
	if weights.Dense3Output != nil {
		names[weights.Dense3Output.Name] = struct{}{}
	}
	if weights.ClassifierOutput != nil {
		names[weights.ClassifierOutput.Name] = struct{}{}
	}
	for _, pointer := range []*gguf.TensorInfo{
		weights.PerLayerTokenEmbedding,
		weights.PerLayerModelProjection,
		weights.PerLayerProjectionNorm,
		weights.AltUpProjection,
		weights.AltUpUnembedding,
		weights.FeatureProjection,
		weights.FeatureProjectionPost,
		weights.DraftToTarget,
	} {
		if pointer != nil {
			names[pointer.Name] = struct{}{}
		}
	}
	if wav := weights.WavTokenizer; wav != nil {
		infos := []gguf.TensorInfo{
			wav.InputConv, wav.InputConvBias, wav.TokenNorm, wav.TokenNormBias,
			wav.OutputNorm, wav.OutputNormBias, wav.Output, wav.OutputBias,
		}
		for _, layer := range wav.PosNet {
			infos = append(infos,
				layer.Norm1, layer.Norm1Bias, layer.Conv1, layer.Conv1Bias,
				layer.Norm2, layer.Norm2Bias, layer.Conv2, layer.Conv2Bias,
				layer.AttentionNorm, layer.AttentionNormBias,
				layer.AttentionQ, layer.AttentionQBias, layer.AttentionK, layer.AttentionKBias,
				layer.AttentionV, layer.AttentionVBias, layer.AttentionOutput, layer.AttentionOutBias,
			)
		}
		for _, layer := range wav.ConvNext {
			infos = append(infos,
				layer.Depthwise, layer.DepthwiseBias, layer.Norm, layer.NormBias,
				layer.Pointwise1, layer.Pointwise1Bias, layer.Pointwise2, layer.Pointwise2Bias, layer.Gamma,
			)
		}
		for _, info := range infos {
			if info.Name != "" {
				names[info.Name] = struct{}{}
			}
		}
	}
	allLayers := weights.LayerCatalog()
	for _, layer := range allLayers {
		infos := []gguf.TensorInfo{
			layer.AttentionNorm,
			layer.FeedForwardNorm,
			layer.FeedForwardGate,
			layer.FeedForwardUp,
			layer.FeedForwardDown,
		}
		if layer.Recurrent {
			if layer.SSMQueryConv != nil {
				infos = append(infos, layer.AttentionQ, layer.AttentionK, layer.AttentionV, layer.AttentionOutput)
			}
			for _, pointer := range []*gguf.TensorInfo{
				layer.SSMInput,
				layer.AttentionQKV,
				layer.AttentionGate,
				layer.SSMConv1D,
				layer.SSMConv1DBias,
				layer.SSMX,
				layer.SSMTimeStepWeight,
				layer.SSMTimeStep,
				layer.SSMTimeStepNorm,
				layer.SSMA,
				layer.SSMD,
				layer.SSMBNorm,
				layer.SSMCNorm,
				layer.SSMBeta,
				layer.SSMAlpha,
				layer.SSMBetaAlpha,
				layer.SSMNorm,
				layer.SSMOutput,
				layer.SSMQueryConv,
				layer.SSMKeyConv,
				layer.SSMValueConv,
				layer.SSMForgetA,
				layer.SSMForgetB,
				layer.SSMOutputGateA,
				layer.SSMOutputGateB,
				layer.ShortConvKernel,
				layer.ShortConvInput,
				layer.ShortConvOutput,
				layer.TimeMixW1,
				layer.TimeMixW2,
				layer.TimeMixW0,
				layer.TimeMixA0,
				layer.TimeMixA1,
				layer.TimeMixA2,
				layer.TimeMixV0,
				layer.TimeMixV1,
				layer.TimeMixV2,
				layer.TimeMixG1,
				layer.TimeMixG2,
				layer.TimeMixKK,
				layer.TimeMixKA,
				layer.TimeMixRK,
				layer.TimeMixLerpX,
				layer.TimeMixLerpFused,
				layer.TimeMixLerpW,
				layer.TimeMixLerpK,
				layer.TimeMixLerpV,
				layer.TimeMixLerpR,
				layer.TimeMixLerpG,
				layer.TimeMixFirst,
				layer.TimeMixDecay,
				layer.TimeMixDecayW1,
				layer.TimeMixDecayW2,
				layer.TimeMixKey,
				layer.TimeMixValue,
				layer.TimeMixReceptance,
				layer.TimeMixGate,
				layer.TimeMixLN,
				layer.TimeMixLNBias,
				layer.TimeMixOutput,
				layer.ChannelMixLerpK,
				layer.ChannelMixLerpR,
				layer.ChannelMixKey,
				layer.ChannelMixValue,
				layer.ChannelMixReceptance,
			} {
				if pointer != nil {
					infos = append(infos, *pointer)
				}
			}
		} else {
			if layer.AttentionQKV != nil {
				infos = append(infos, *layer.AttentionQKV)
			} else {
				infos = append(infos, layer.AttentionQ, layer.AttentionK, layer.AttentionV)
			}
			infos = append(infos, layer.AttentionOutput)
		}
		for _, info := range infos {
			if info.Name != "" {
				names[info.Name] = struct{}{}
			}
		}
		if layer.AttentionQNorm != nil {
			names[layer.AttentionQNorm.Name] = struct{}{}
		}
		for _, pointer := range []*gguf.TensorInfo{
			layer.AttentionQB,
			layer.AttentionNormBias,
			layer.AttentionNorm2,
			layer.AttentionNorm2Bias,
			layer.AttentionQScale,
			layer.AttentionKScale,
			layer.AttentionVScale,
			layer.AttentionOutputScale,
			layer.AttentionSubNorm,
			layer.AttentionOutputGate,
			layer.AttentionSinks,
			layer.AttentionQKVBias,
			layer.AttentionQNormBias,
			layer.AttentionKNormBias,
			layer.AttentionQBias,
			layer.AttentionKBias,
			layer.AttentionVBias,
			layer.AttentionOutputBias,
			layer.FeedForwardGateBias,
			layer.FeedForwardUpBias,
			layer.FeedForwardDownBias,
			layer.FeedForwardNormBias,
			layer.FeedForwardExpertNorm,
			layer.FeedForwardGateScale,
			layer.FeedForwardUpScale,
			layer.FeedForwardDownScale,
			layer.FeedForwardActivationScale,
			layer.FeedForwardSubNorm,
			layer.FeedForwardRouter,
			layer.FeedForwardRouterBias,
			layer.FeedForwardGateUpExperts,
			layer.FeedForwardGateExperts,
			layer.FeedForwardUpExperts,
			layer.FeedForwardDownExperts,
			layer.FeedForwardDownExpertsScale,
			layer.FeedForwardGateChunkExperts,
			layer.FeedForwardUpChunkExperts,
			layer.FeedForwardDownChunkExperts,
			layer.FeedForwardExpertBias,
			layer.FeedForwardLatentDown,
			layer.FeedForwardLatentUp,
			layer.FeedForwardSharedGate,
			layer.FeedForwardSharedUp,
			layer.FeedForwardSharedDown,
			layer.FeedForwardSharedRouter,
			layer.LayerOutputScale,
			layer.FeedForwardPreNorm2,
			layer.FeedForwardPostNorm1,
			layer.FeedForwardPostNorm2,
			layer.FeedForwardRouterScale,
			layer.PerLayerInputGate,
			layer.PerLayerProjection,
			layer.PerLayerPostNorm,
			layer.AltUpCorrectCoefficient,
			layer.AltUpCorrectScale,
			layer.AltUpPredictCoefficient,
			layer.AltUpRouter,
			layer.AltUpRouterNorm,
			layer.LaurelLeft,
			layer.LaurelRight,
			layer.LaurelPostNorm,
			layer.AttentionKVAMQA,
			layer.AttentionKVANorm,
			layer.AttentionKVB,
			layer.AttentionKB,
			layer.AttentionVB,
			layer.CrossAttentionNorm,
			layer.CrossAttentionQ,
			layer.CrossAttentionK,
			layer.CrossAttentionV,
			layer.CrossAttentionOutput,
			layer.IndexerKNorm,
			layer.IndexerKNormBias,
			layer.IndexerProjection,
			layer.IndexerAttentionK,
			layer.IndexerAttentionQB,
			layer.AttentionOutputA,
			layer.AttentionCompressorKV,
			layer.AttentionCompressorGate,
			layer.AttentionCompressorAPE,
			layer.AttentionCompressorNorm,
			layer.IndexerCompressorKV,
			layer.IndexerCompressorGate,
			layer.IndexerCompressorAPE,
			layer.IndexerCompressorNorm,
			layer.HyperAttentionFN,
			layer.HyperAttentionBase,
			layer.HyperAttentionScale,
			layer.HyperFeedForwardFN,
			layer.HyperFeedForwardBase,
			layer.HyperFeedForwardScale,
			layer.HyperHeadFN,
			layer.HyperHeadBase,
			layer.HyperHeadScale,
			layer.FeedForwardHashExperts,
			layer.VisualAttentionQKV,
			layer.VisualAttentionOutput,
			layer.VisualFeedForwardGate,
			layer.VisualFeedForwardUp,
			layer.VisualFeedForwardDown,
			layer.SSMInput,
			layer.SSMConv1D,
			layer.SSMConv1DBias,
			layer.SSMTimeStep,
			layer.SSMA,
			layer.SSMD,
			layer.SSMNorm,
			layer.SSMOutput,
		} {
			if pointer != nil {
				names[pointer.Name] = struct{}{}
			}
		}
		if layer.AttentionKNorm != nil {
			names[layer.AttentionKNorm.Name] = struct{}{}
		}
		if layer.AttentionPostNorm != nil {
			names[layer.AttentionPostNorm.Name] = struct{}{}
		}
		if layer.AttentionPostNormBias != nil {
			names[layer.AttentionPostNormBias.Name] = struct{}{}
		}
		if layer.AttentionRelativeBias != nil {
			names[layer.AttentionRelativeBias.Name] = struct{}{}
		}
		if layer.RopeFactors != nil {
			names[layer.RopeFactors.Name] = struct{}{}
		}
		if layer.FeedForwardPostNorm != nil {
			names[layer.FeedForwardPostNorm.Name] = struct{}{}
		}
		if layer.FeedForwardPostNormBias != nil {
			names[layer.FeedForwardPostNormBias.Name] = struct{}{}
		}
	}
	result := make([]gguf.TensorInfo, 0, len(names))
	for _, info := range file.Tensors {
		if _, ok := names[info.Name]; ok {
			result = append(result, info)
		}
	}
	return result
}
