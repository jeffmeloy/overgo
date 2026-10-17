package model

import "overgo/internal/tensor"

func bindGatedHyperWeights(catalog weightCatalog, prefix string, spec Spec, layer *LayerWeights, block uint32) error {
	width := uint64(spec.EmbeddingLength) * uint64(spec.HyperConnectionCount)
	rank, streams := uint64(spec.HyperConnectionRank), uint64(spec.HyperConnectionCount)
	if !spec.IsRecurrentLayer(block) {
		indexWidth := uint64(spec.IndexerKeyLength)
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer("index_q_proj.weight", &layer.IndexerAttentionQB, uint64(spec.EmbeddingLength), indexWidth*uint64(spec.IndexerHeadCount)),
			requiredTensorPointer("index_k_proj.weight", &layer.IndexerAttentionK, uint64(spec.EmbeddingLength), indexWidth),
			requiredTensorPointer("index_q_norm.weight", &layer.IndexerQNorm, indexWidth),
			requiredTensorPointer("index_k_norm.weight", &layer.IndexerKNorm, indexWidth),
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
