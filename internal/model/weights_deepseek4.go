package model

import (
	"fmt"

	"llamacpp2go/internal/tensor"
	"llamacpp2go/internal/tensor/dtype"
)

func readDeepSeek4WeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	required, tensors := catalog.required, catalog.tensors
	width := uint64(spec.EmbeddingLength)
	headWidth := uint64(spec.KeyLength)
	hyper := uint64(spec.HyperConnectionCount)
	hyperWidth := hyper * width
	mixWidth := (2 + hyper) * hyper
	result := Weights{Layers: make([]LayerWeights, spec.BlockCount)}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensor("token_embd.weight", &result.TokenEmbedding, width, uint64(spec.VocabularySize)),
		requiredTensor("output_norm.weight", &result.OutputNorm, width),
		requiredTensorPointer("output.weight", &result.Output, width, uint64(spec.VocabularySize)),
	}); err != nil {
		return Weights{}, err
	}
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		layer := &result.Layers[block]
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensor("attn_norm.weight", &layer.AttentionNorm, width),
			requiredTensor("attn_q_a.weight", &layer.AttentionQ, width, uint64(spec.QLoRARank)),
			requiredTensor("attn_kv.weight", &layer.AttentionK, width, headWidth),
			requiredTensor("attn_output.weight", &layer.AttentionOutput, uint64(spec.AttentionOutputGroups*spec.AttentionOutputRank), width),
			requiredTensor("ffn_norm.weight", &layer.FeedForwardNorm, width),
			requiredTensorPointer("attn_sinks.weight", &layer.AttentionSinks, uint64(spec.HeadCount)),
			requiredTensorPointer("attn_q_a_norm.weight", &layer.AttentionQNorm, uint64(spec.QLoRARank)),
			requiredTensorPointer("attn_q_b.weight", &layer.AttentionQB, uint64(spec.QLoRARank), uint64(spec.HeadCount)*headWidth),
			requiredTensorPointer("attn_kv_a_norm.weight", &layer.AttentionKNorm, headWidth),
			requiredTensorPointer("attn_output_a.weight", &layer.AttentionOutputA, uint64(spec.HeadCount)*headWidth/uint64(spec.AttentionOutputGroups), uint64(spec.AttentionOutputRank*spec.AttentionOutputGroups)),
			requiredTensorPointer("hc_attn_fn.weight", &layer.HyperAttentionFN, hyperWidth, mixWidth),
			requiredTensorPointer("hc_attn_base.weight", &layer.HyperAttentionBase, mixWidth),
			requiredTensorPointer("hc_attn_scale.weight", &layer.HyperAttentionScale, 3),
			requiredTensorPointer("hc_ffn_fn.weight", &layer.HyperFeedForwardFN, hyperWidth, mixWidth),
			requiredTensorPointer("hc_ffn_base.weight", &layer.HyperFeedForwardBase, mixWidth),
			requiredTensorPointer("hc_ffn_scale.weight", &layer.HyperFeedForwardScale, 3),
			requiredTensorPointer("ffn_gate_inp.weight", &layer.FeedForwardRouter, width, uint64(spec.ExpertCount)),
			requiredTensorPointer("ffn_gate_exps.weight", &layer.FeedForwardGateExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredTensorPointer("ffn_up_exps.weight", &layer.FeedForwardUpExperts, width, uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)),
			requiredTensorPointer("ffn_down_exps.weight", &layer.FeedForwardDownExperts, uint64(spec.ExpertFeedForward), width, uint64(spec.ExpertCount)),
		}); err != nil {
			return Weights{}, err
		}
		if err := loadSharedExpertWeightsForWidth(required, prefix, width, spec, layer, false); err != nil {
			return Weights{}, err
		}
		if block < spec.HashLayerCount {
			loaded, err := required(prefix+"ffn_gate_tid2eid.weight", uint64(spec.ExpertUsedCount), uint64(spec.VocabularySize))
			if err != nil {
				return Weights{}, err
			}
			if loaded.Type != dtype.I32 {
				return Weights{}, fmt.Errorf("tensor %q must use I32 storage", loaded.Name)
			}
			layer.FeedForwardHashExperts = &loaded
		} else {
			loaded, err := required(prefix+"exp_probs_b.bias", uint64(spec.ExpertCount))
			if err != nil {
				return Weights{}, err
			}
			layer.FeedForwardRouterBias = &loaded
		}
		if ratio := tensor.DeepSeek4CompressionRatio(spec.CompressRatios[block]); ratio.Enabled() {
			coefficient := ratio.KVWidthMultiplier()
			if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("attn_compressor_kv.weight", &layer.AttentionCompressorKV, width, coefficient*headWidth),
				requiredTensorPointer("attn_compressor_gate.weight", &layer.AttentionCompressorGate, width, coefficient*headWidth),
				requiredTensorPointer("attn_compressor_ape.weight", &layer.AttentionCompressorAPE, coefficient*headWidth, uint64(ratio)),
				requiredTensorPointer("attn_compressor_norm.weight", &layer.AttentionCompressorNorm, headWidth),
			}); err != nil {
				return Weights{}, err
			}
		}
		if tensor.DeepSeek4CompressionRatio(spec.CompressRatios[block]).UsesIndexer() {
			indexerWidth := uint64(spec.IndexerKeyLength)
			if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("indexer.proj.weight", &layer.IndexerProjection, width, uint64(spec.IndexerHeadCount)),
				requiredTensorPointer("indexer.attn_q_b.weight", &layer.IndexerAttentionQB, uint64(spec.QLoRARank), uint64(spec.IndexerHeadCount)*indexerWidth),
				requiredTensorPointer("indexer_compressor_kv.weight", &layer.IndexerCompressorKV, width, 2*indexerWidth),
				requiredTensorPointer("indexer_compressor_gate.weight", &layer.IndexerCompressorGate, width, 2*indexerWidth),
				requiredTensorPointer(
					"indexer_compressor_ape.weight",
					&layer.IndexerCompressorAPE,
					2*indexerWidth,
					uint64(tensor.DeepSeek4CompressionOverlap),
				),
				requiredTensorPointer("indexer_compressor_norm.weight", &layer.IndexerCompressorNorm, indexerWidth),
			}); err != nil {
				return Weights{}, err
			}
		}
	}
	last := &result.Layers[len(result.Layers)-1]
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensorPointer("output_hc_fn.weight", &last.HyperHeadFN, hyperWidth, hyper),
		requiredTensorPointer("output_hc_base.weight", &last.HyperHeadBase, hyper),
		requiredTensorPointer("output_hc_scale.weight", &last.HyperHeadScale, 1),
	}); err != nil {
		return Weights{}, err
	}
	return result, nil
}
