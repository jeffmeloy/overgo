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
