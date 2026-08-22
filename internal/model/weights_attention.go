package model

import (
	"fmt"

	"overgo/internal/tensor"
)

func loadQLoRAQuery(catalog weightCatalog, prefix string, spec Spec, queryLength uint64, layer *LayerWeights) error {
	return bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer("attn_q_a.weight", &layer.AttentionQ,
			uint64(spec.EmbeddingLength), uint64(spec.QLoRARank)),
		requiredTensorPointer("attn_q_a_norm.weight", &layer.AttentionQNorm,
			uint64(spec.QLoRARank)),
		requiredTensorPointer("attn_q_b.weight", &layer.AttentionQB,
			uint64(spec.QLoRARank), queryLength),
	})
}

func loadLatentAttentionCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	plan LayerPlan,
	queryLength, outputLength uint64,
) error {
	expandQuery := plan.LatentAttention == latentAttentionNeoXResidualScale ||
		((plan.LatentYaRNQuery || plan.Attention == AttentionSparseLatent ||
			plan.LatentAttention == latentAttentionNoRoPE || plan.Mixer == recurrentMixerKeyedDelta) &&
			spec.QLoRARank > tensor.FirstOffset)
	if expandQuery {
		if err := loadQLoRAQuery(catalog, prefix, spec, queryLength, layer); err != nil {
			return err
		}
	} else if err := bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ,
			uint64(spec.EmbeddingLength), queryLength),
	}); err != nil {
		return err
	}
	nope := uint64(spec.KeyLength - spec.RopeDimensionCount)
	program := []tensorBinding{
		requiredTensorPointer("attn_kv_a_mqa.weight", &layer.AttentionKVAMQA,
			uint64(spec.EmbeddingLength), uint64(spec.KVLoRARank+spec.RopeDimensionCount)),
		requiredTensorPointer("attn_kv_a_norm.weight", &layer.AttentionKVANorm, uint64(spec.KVLoRARank)),
	}
	if _, modern := catalog.tensors[prefix+"attn_k_b.weight"]; modern {
		program = append(program,
			requiredTensorPointer("attn_k_b.weight", &layer.AttentionKB,
				nope, uint64(spec.KVLoRARank), uint64(spec.HeadCount)),
			requiredTensorPointer("attn_v_b.weight", &layer.AttentionVB,
				uint64(spec.KVLoRARank), uint64(spec.ValueLength), uint64(spec.HeadCount)),
		)
	} else {
		program = append(program, requiredTensorPointer(
			"attn_kv_b.weight", &layer.AttentionKVB, uint64(spec.KVLoRARank),
			uint64(spec.HeadCount)*(nope+uint64(spec.ValueLength)),
		))
	}
	if plan.Attention == AttentionSparseLatent && spec.LayerHasFullIndexer(plan.Layer) {
		program = append(program,
			requiredTensorPointer("indexer.k_norm.weight", &layer.IndexerKNorm, uint64(spec.IndexerKeyLength)),
			requiredTensorPointer("indexer.k_norm.bias", &layer.IndexerKNormBias, uint64(spec.IndexerKeyLength)),
			requiredTensorPointer("indexer.proj.weight", &layer.IndexerProjection, uint64(spec.EmbeddingLength), uint64(spec.IndexerHeadCount)),
			requiredTensorPointer("indexer.attn_k.weight", &layer.IndexerAttentionK, uint64(spec.EmbeddingLength), uint64(spec.IndexerKeyLength)),
			requiredTensorPointer("indexer.attn_q_b.weight", &layer.IndexerAttentionQB, uint64(spec.QLoRARank), uint64(spec.IndexerHeadCount)*uint64(spec.IndexerKeyLength)),
		)
	}
	program = append(program, requiredTensorPointer(
		attentionOutputWeightTensor, &layer.AttentionOutput, outputLength, uint64(spec.EmbeddingLength),
	))
	return bindTensorProgram(catalog, prefix, program)
}

func loadStandardAttentionCatalog(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	queryLength, keyLength, valueLength, outputLength uint64,
) error {
	profile := spec.Profile()
	if profile.Has(ArchitectureFusedQKV) {
		_, present := catalog.tensors[prefix+"attn_qkv.weight"]
		if present || profile.Has(ArchitectureRequiresFusedQKV) {
			bias := optionalF32TensorPointer(
				"attn_qkv.bias", &layer.AttentionQKVBias, queryLength+keyLength+valueLength,
			)
			bias.optional = !profile.Has(ArchitectureRequiresFusedQKVBias)
			if err := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer("attn_qkv.weight", &layer.AttentionQKV,
					uint64(spec.EmbeddingLength), queryLength+keyLength+valueLength),
				bias,
			}); err != nil {
				return err
			}
		}
	}
	if layer.AttentionQKV == nil {
		if profile.Has(ArchitectureRejectsOrphanFusedQKVBias) {
			if _, present := catalog.tensors[prefix+"attn_qkv.bias"]; present {
				return fmt.Errorf("%s fused QKV bias has no fused weight", spec.Architecture)
			}
		}
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, uint64(spec.EmbeddingLength), queryLength),
			requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK, uint64(spec.EmbeddingLength), keyLength),
			requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV, uint64(spec.EmbeddingLength), valueLength),
		}); err != nil {
			return err
		}
	}
	return bindTensorProgram(catalog, prefix, []tensorBinding{
		requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput,
			outputLength, uint64(spec.EmbeddingLength)),
	})
}
