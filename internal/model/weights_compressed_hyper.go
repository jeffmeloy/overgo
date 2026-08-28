package model

import (
	"fmt"

	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func readCompressedHyperWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	width := uint64(spec.EmbeddingLength)
	headWidth := uint64(spec.KeyLength)
	hyper := uint64(spec.HyperConnectionCount)
	hyperWidth := hyper * width
	mixWidth := (tensor.PairedExtent + hyper) * hyper
	result := Weights{Layers: make([]LayerWeights, spec.BlockCount)}
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensor(tokenEmbeddingWeightTensor, &result.TokenEmbedding, width, uint64(spec.VocabularySize)),
		requiredTensor(outputNormWeightTensor, &result.OutputNorm, width),
		requiredTensorPointer(outputWeightTensor, &result.Output, width, uint64(spec.VocabularySize)),
	}); err != nil {
		return Weights{}, err
	}
	for block := uint32(tensor.FirstOffset); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		layer := &result.Layers[block]
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer(attentionNormWeightTensor, &layer.AttentionNorm, width),
			requiredTensorPointer("attn_q_a.weight", &layer.AttentionQ, width, uint64(spec.QLoRARank)),
			requiredTensorPointer("attn_kv.weight", &layer.AttentionK, width, headWidth),
			requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, uint64(spec.AttentionOutputGroups*spec.AttentionOutputRank), width),
			requiredTensorPointer(feedForwardNormWeightTensor, &layer.FeedForwardNorm, width),
			requiredTensorPointer("attn_sinks.weight", &layer.AttentionSinks, uint64(spec.HeadCount)),
			requiredTensorPointer("attn_q_a_norm.weight", &layer.AttentionQNorm, uint64(spec.QLoRARank)),
			requiredTensorPointer("attn_q_b.weight", &layer.AttentionQB, uint64(spec.QLoRARank), uint64(spec.HeadCount)*headWidth),
			requiredTensorPointer("attn_kv_a_norm.weight", &layer.AttentionKNorm, headWidth),
			requiredTensorPointer("attn_output_a.weight", &layer.AttentionOutputA, uint64(spec.HeadCount)*headWidth/uint64(spec.AttentionOutputGroups), uint64(spec.AttentionOutputRank*spec.AttentionOutputGroups)),
			requiredTensorPointer("hc_attn_fn.weight", &layer.HyperAttentionFN, hyperWidth, mixWidth),
			requiredTensorPointer("hc_attn_base.weight", &layer.HyperAttentionBase, mixWidth),
			requiredTensorPointer("hc_attn_scale.weight", &layer.HyperAttentionScale, tensor.TripleExtent),
			requiredTensorPointer("hc_ffn_fn.weight", &layer.HyperFeedForwardFN, hyperWidth, mixWidth),
			requiredTensorPointer("hc_ffn_base.weight", &layer.HyperFeedForwardBase, mixWidth),
			requiredTensorPointer("hc_ffn_scale.weight", &layer.HyperFeedForwardScale, tensor.TripleExtent),
			requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, width, uint64(spec.ExpertCount)),
			requiredTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, uint64(spec.ExpertFeedForward), width, uint64(spec.ExpertCount)),
		}); err != nil {
			return Weights{}, err
		}
		if err := bindTensorProgram(catalog, prefix,
			sharedExpertBindings(width, spec, layer, sharedExpertCatalogAlways)); err != nil {
			return Weights{}, err
		}
		if block < spec.HashLayerCount {
			if err := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredStoredTensorPointer("ffn_gate_tid2eid.weight", &layer.FeedForwardHashExperts,
					dtype.I32, uint64(spec.ExpertUsedCount), uint64(spec.VocabularySize)),
			}); err != nil {
				return Weights{}, err
			}
		} else {
			if err := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer("exp_probs_b.bias", &layer.FeedForwardRouterBias, uint64(spec.ExpertCount)),
			}); err != nil {
				return Weights{}, err
			}
		}
		if ratio := tensor.CompressionRatio(spec.CompressRatios[block]); ratio.Enabled() {
			coefficient := ratio.KVWidthMultiplier()
			if err := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer("attn_compressor_kv.weight", &layer.AttentionCompressorKV, width, coefficient*headWidth),
				requiredTensorPointer("attn_compressor_gate.weight", &layer.AttentionCompressorGate, width, coefficient*headWidth),
				requiredTensorPointer("attn_compressor_ape.weight", &layer.AttentionCompressorAPE, coefficient*headWidth, uint64(ratio)),
				requiredTensorPointer("attn_compressor_norm.weight", &layer.AttentionCompressorNorm, headWidth),
			}); err != nil {
				return Weights{}, err
			}
		}
		if tensor.CompressionRatio(spec.CompressRatios[block]).UsesIndexer() {
			indexerWidth := uint64(spec.IndexerKeyLength)
			if err := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer("indexer.proj.weight", &layer.IndexerProjection, width, uint64(spec.IndexerHeadCount)),
				requiredTensorPointer("indexer.attn_q_b.weight", &layer.IndexerAttentionQB, uint64(spec.QLoRARank), uint64(spec.IndexerHeadCount)*indexerWidth),
				requiredTensorPointer("indexer_compressor_kv.weight", &layer.IndexerCompressorKV,
					width, tensor.PairedExtent*indexerWidth),
				requiredTensorPointer("indexer_compressor_gate.weight", &layer.IndexerCompressorGate,
					width, tensor.PairedExtent*indexerWidth),
				requiredTensorPointer(
					"indexer_compressor_ape.weight",
					&layer.IndexerCompressorAPE,
					tensor.PairedExtent*indexerWidth,
					uint64(tensor.CompressionOverlap),
				),
				requiredTensorPointer("indexer_compressor_norm.weight", &layer.IndexerCompressorNorm, indexerWidth),
			}); err != nil {
				return Weights{}, err
			}
		}
	}
	last := &result.Layers[len(result.Layers)-tensor.SingletonExtent]
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensorPointer("output_hc_fn.weight", &last.HyperHeadFN, hyperWidth, hyper),
		requiredTensorPointer("output_hc_base.weight", &last.HyperHeadBase, hyper),
		requiredTensorPointer("output_hc_scale.weight", &last.HyperHeadScale, tensor.SingletonExtent),
	}); err != nil {
		return Weights{}, err
	}
	return result, nil
}
