package model

import (
	"fmt"
)

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
			if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{
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
		if err := loadTensorRequirements(catalog, prefix, []tensorRequirement{
			requiredTensor("attn_q.weight", &layer.AttentionQ, uint64(spec.EmbeddingLength), queryLength),
			requiredTensor("attn_k.weight", &layer.AttentionK, uint64(spec.EmbeddingLength), keyLength),
			requiredTensor("attn_v.weight", &layer.AttentionV, uint64(spec.EmbeddingLength), valueLength),
		}); err != nil {
			return err
		}
	}
	output, err := catalog.required(
		prefix+"attn_output.weight", outputLength, uint64(spec.EmbeddingLength),
	)
	if err != nil {
		return err
	}
	layer.AttentionOutput = output
	return nil
}
