package model

import "fmt"

func readWavTokenizerWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	required, tensors := catalog.required, catalog.tensors
	width := uint64(spec.PosNetEmbeddingLength)
	ffn := uint64(spec.FeedForwardLength)
	result := Weights{}
	wav := &WavTokenizerWeights{
		PosNet:   make([]WavPosNetWeights, spec.PosNetBlockCount),
		ConvNext: make([]WavConvNextWeights, spec.ConvNextBlockCount),
	}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensor("token_embd.weight", &result.TokenEmbedding, uint64(spec.EmbeddingLength), uint64(spec.VocabularySize)),
		requiredTensor("conv1d.weight", &wav.InputConv, 7, uint64(spec.EmbeddingLength), width),
		requiredTensor("conv1d.bias", &wav.InputConvBias, 1, width),
	}); err != nil {
		return Weights{}, err
	}
	for block := uint32(0); block < spec.PosNetBlockCount; block++ {
		prefix := fmt.Sprintf("posnet.%d.", block)
		layer := &wav.PosNet[block]
		var requirements []tensorRequirement
		switch wavTokenizerResidualProgram[block] {
		case sequenceResidualConvolution:
			requirements = []tensorRequirement{
				requiredTensor("norm1.weight", &layer.Norm1, 1, width),
				requiredTensor("norm1.bias", &layer.Norm1Bias, 1, width),
				requiredTensor("conv1.weight", &layer.Conv1, 3, width, width),
				requiredTensor("conv1.bias", &layer.Conv1Bias, 1, width),
				requiredTensor("norm2.weight", &layer.Norm2, 1, width),
				requiredTensor("norm2.bias", &layer.Norm2Bias, 1, width),
				requiredTensor("conv2.weight", &layer.Conv2, 3, width, width),
				requiredTensor("conv2.bias", &layer.Conv2Bias, 1, width),
			}
		case sequenceResidualAttention:
			requirements = []tensorRequirement{
				requiredTensor("attn_norm.weight", &layer.AttentionNorm, 1, width),
				requiredTensor("attn_norm.bias", &layer.AttentionNormBias, 1, width),
				requiredTensor("attn_q.bias", &layer.AttentionQBias, 1, width),
				requiredTensor("attn_k.bias", &layer.AttentionKBias, 1, width),
				requiredTensor("attn_v.bias", &layer.AttentionVBias, 1, width),
				requiredTensor("attn_output.bias", &layer.AttentionOutBias, 1, width),
				requiredTensor("attn_q.weight", &layer.AttentionQ, 1, width, width),
				requiredTensor("attn_k.weight", &layer.AttentionK, 1, width, width),
				requiredTensor("attn_v.weight", &layer.AttentionV, 1, width, width),
				requiredTensor("attn_output.weight", &layer.AttentionOutput, 1, width, width),
			}
		case sequenceResidualNormalization:
			requirements = []tensorRequirement{
				requiredTensor("attn_norm.weight", &layer.AttentionNorm, 1, width),
				requiredTensor("attn_norm.bias", &layer.AttentionNormBias, 1, width),
			}
		}
		if err := loadTensorRequirements(required, tensors, prefix, requirements); err != nil {
			return Weights{}, err
		}
	}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensor("token_embd_norm.weight", &wav.TokenNorm, width),
		requiredTensor("token_embd_norm.bias", &wav.TokenNormBias, width),
	}); err != nil {
		return Weights{}, err
	}
	for block := uint32(0); block < spec.ConvNextBlockCount; block++ {
		prefix := fmt.Sprintf("convnext.%d.", block)
		layer := &wav.ConvNext[block]
		if err := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensor("dw.weight", &layer.Depthwise, 7, 1, width),
			requiredTensor("dw.bias", &layer.DepthwiseBias, 1, width),
			requiredTensor("norm.weight", &layer.Norm, width),
			requiredTensor("norm.bias", &layer.NormBias, width),
			requiredTensor("pw1.weight", &layer.Pointwise1, width, ffn),
			requiredTensor("pw1.bias", &layer.Pointwise1Bias, ffn),
			requiredTensor("pw2.weight", &layer.Pointwise2, ffn, width),
			requiredTensor("pw2.bias", &layer.Pointwise2Bias, width),
			requiredTensor("gamma.weight", &layer.Gamma, width),
		}); err != nil {
			return Weights{}, err
		}
	}
	if err := loadTensorRequirements(required, tensors, "", []tensorRequirement{
		requiredTensor("output_norm.weight", &wav.OutputNorm, width),
		requiredTensor("output_norm.bias", &wav.OutputNormBias, width),
		requiredTensor("output.weight", &wav.Output, width, uint64(spec.OutputEmbeddingLength)),
		requiredTensor("output.bias", &wav.OutputBias, uint64(spec.OutputEmbeddingLength)),
	}); err != nil {
		return Weights{}, err
	}
	result.WavTokenizer = wav
	result.Output = &wav.Output
	result.OutputBias = &wav.OutputBias
	return result, nil
}
