package model

import "overgo/internal/tensor"

func bindGatedHyperWeights(catalog weightCatalog, prefix string, spec Spec, layer *LayerWeights, block uint32) error {
	width := uint64(spec.EmbeddingLength) * uint64(spec.HyperConnectionCount)
	rank, streams := uint64(spec.HyperConnectionRank), uint64(spec.HyperConnectionCount)
	if n := spec.NGram; n != nil && n.Layer == block {
		embed := uint64(spec.EmbeddingLength)
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer("ple_key.weight", &layer.NGramKey, embed, width),
			requiredTensorPointer("ple_value.weight", &layer.NGramValue, embed, embed),
			requiredTensorPointer("ple_norm_key.weight", &layer.NGramKeyNorm, width),
			requiredTensorPointer("ple_norm_query.weight", &layer.NGramQueryNorm, width),
			requiredTensorPointer("ple_norm_conv.weight", &layer.NGramConvNorm, width),
			requiredTensorPointer("ple_conv1d.weight", &layer.NGramConv, uint64(n.ConvKernel), width),
		}); err != nil {
			return err
		}
	}
	if !spec.IsRecurrentLayer(block) {
		indexWidth := uint64(spec.IndexerKeyLength)
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer("indexer.q_proj.weight", &layer.IndexerAttentionQB, uint64(spec.EmbeddingLength), indexWidth*uint64(spec.IndexerHeadCount)),
			requiredTensorPointer("indexer.k_proj.weight", &layer.IndexerAttentionK, uint64(spec.EmbeddingLength), indexWidth),
			requiredTensorPointer("indexer.q_norm.weight", &layer.IndexerQNorm, indexWidth),
			requiredTensorPointer("indexer.k_norm.weight", &layer.IndexerKNorm, indexWidth),
		}); err != nil {
			return err
		}
	}
	if err := bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer("hc_attn_norm.weight", &layer.HyperAttentionNorm, width),
		requiredTensorPointer("hc_attn_down.weight", &layer.HyperAttentionDown, width, rank),
		requiredTensorPointer("hc_attn_up.weight", &layer.HyperAttentionUp, rank, width),
		requiredTensorPointer("hc_attn_inject.weight", &layer.HyperAttentionInject, width, streams),
		requiredTensorPointer("hc_ffn_norm.weight", &layer.HyperFeedForwardNorm, width),
		requiredTensorPointer("hc_ffn_down.weight", &layer.HyperFeedForwardDown, width, rank),
		requiredTensorPointer("hc_ffn_up.weight", &layer.HyperFeedForwardUp, rank, width),
		requiredTensorPointer("hc_ffn_inject.weight", &layer.HyperFeedForwardInject, width, streams),
	}); err != nil {
		return err
	}
	if block+tensor.SingletonExtent == spec.BlockCount {
		return bindTensorProgram(catalog, "", []tensorBinding{
			requiredTensorPointer("output_hc_norm.weight", &layer.HyperHeadNorm, width),
			requiredTensorPointer("output_hc_down.weight", &layer.HyperHeadDown, width, rank),
			requiredTensorPointer("output_hc_up.weight", &layer.HyperHeadUp, rank, width),
		})
	}
	return nil
}
