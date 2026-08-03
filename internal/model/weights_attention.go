package model

import (
	"fmt"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

func loadStandardAttentionCatalog(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	queryLength, keyLength, valueLength, outputLength uint64,
) error {
	profile := spec.Profile()
	if profile.Has(ArchitectureFusedQKV) {
		_, present := tensors[prefix+"attn_qkv.weight"]
		if present || profile.Has(ArchitectureRequiresFusedQKV) {
			qkv, err := required(
				prefix+"attn_qkv.weight", uint64(spec.EmbeddingLength),
				queryLength+keyLength+valueLength,
			)
			if err != nil {
				return err
			}
			layer.AttentionQKV = &qkv
			if _, present := tensors[prefix+"attn_qkv.bias"]; present {
				bias, err := required(prefix+"attn_qkv.bias", queryLength+keyLength+valueLength)
				if err != nil {
					return err
				}
				if bias.Type != dtype.F32 {
					return fmt.Errorf("tensor %q must use F32 bias storage", bias.Name)
				}
				layer.AttentionQKVBias = &bias
			}
			if profile.Has(ArchitectureRequiresFusedQKVBias) && layer.AttentionQKVBias == nil {
				return fmt.Errorf("required tensor %q is missing", prefix+"attn_qkv.bias")
			}
		}
	}
	if layer.AttentionQKV == nil {
		if profile.Has(ArchitectureRejectsOrphanFusedQKVBias) {
			if _, present := tensors[prefix+"attn_qkv.bias"]; present {
				return fmt.Errorf("%s fused QKV bias has no fused weight", spec.Architecture)
			}
		}
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensor("attn_q.weight", &layer.AttentionQ, uint64(spec.EmbeddingLength), queryLength),
			requiredTensor("attn_k.weight", &layer.AttentionK, uint64(spec.EmbeddingLength), keyLength),
			requiredTensor("attn_v.weight", &layer.AttentionV, uint64(spec.EmbeddingLength), valueLength),
		}); err != nil {
			return err
		}
	}
	output, err := required(
		prefix+"attn_output.weight", outputLength, uint64(spec.EmbeddingLength),
	)
	if err != nil {
		return err
	}
	layer.AttentionOutput = output
	return nil
}
