package model

import (
	"fmt"

	"overgo/internal/tensor"
)

func readAudioDecoderWeightCatalog(catalog weightCatalog, spec Spec) (Weights, error) {
	width := uint64(spec.PosNetEmbeddingLength)
	ffn := uint64(spec.FeedForwardLength)
	forward := spec.Profile().Forward
	residuals := forward.SequenceResiduals
	if len(residuals) != int(spec.PosNetBlockCount) {
		return Weights{}, fmt.Errorf("sequence-output residual program has %d stages for %d blocks", len(residuals), spec.PosNetBlockCount)
	}
	result := Weights{}
	wav := &AudioDecoderWeights{
		PosNet:   make([]WavPosNetWeights, spec.PosNetBlockCount),
		ConvNext: make([]WavConvNextWeights, spec.ConvNextBlockCount),
	}
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensor(tokenEmbeddingWeightTensor, &result.TokenEmbedding, uint64(spec.EmbeddingLength), uint64(spec.VocabularySize)),
		requiredTensor("conv1d.weight", &wav.InputConv,
			uint64(forward.SequenceInputKernel), uint64(spec.EmbeddingLength), width),
		requiredTensor("conv1d.bias", &wav.InputConvBias, tensor.SingletonExtent, width),
	}); err != nil {
		return Weights{}, err
	}
	for block := uint32(tensor.FirstOffset); block < spec.PosNetBlockCount; block++ {
		prefix := fmt.Sprintf("posnet.%d.", block)
		layer := &wav.PosNet[block]
		var requirements []tensorBinding
		switch residuals[block] {
		case sequenceResidualConvolution:
			requirements = []tensorBinding{
				requiredTensor("norm1.weight", &layer.Norm1, tensor.SingletonExtent, width),
				requiredTensor("norm1.bias", &layer.Norm1Bias, tensor.SingletonExtent, width),
				requiredTensor("conv1.weight", &layer.Conv1,
					uint64(forward.SequenceResidualKernel), width, width),
				requiredTensor("conv1.bias", &layer.Conv1Bias, tensor.SingletonExtent, width),
				requiredTensor("norm2.weight", &layer.Norm2, tensor.SingletonExtent, width),
				requiredTensor("norm2.bias", &layer.Norm2Bias, tensor.SingletonExtent, width),
				requiredTensor("conv2.weight", &layer.Conv2,
					uint64(forward.SequenceResidualKernel), width, width),
				requiredTensor("conv2.bias", &layer.Conv2Bias, tensor.SingletonExtent, width),
			}
		case sequenceResidualAttention:
			requirements = []tensorBinding{
				requiredTensor(attentionNormWeightTensor, &layer.AttentionNorm, tensor.SingletonExtent, width),
				requiredTensor("attn_norm.bias", &layer.AttentionNormBias, tensor.SingletonExtent, width),
				requiredTensor("attn_q.bias", &layer.AttentionQBias, tensor.SingletonExtent, width),
				requiredTensor("attn_k.bias", &layer.AttentionKBias, tensor.SingletonExtent, width),
				requiredTensor("attn_v.bias", &layer.AttentionVBias, tensor.SingletonExtent, width),
				requiredTensor("attn_output.bias", &layer.AttentionOutBias, tensor.SingletonExtent, width),
				requiredTensor(attentionQueryWeightTensor, &layer.AttentionQ, tensor.SingletonExtent, width, width),
				requiredTensor(attentionKeyWeightTensor, &layer.AttentionK, tensor.SingletonExtent, width, width),
				requiredTensor(attentionValueWeightTensor, &layer.AttentionV, tensor.SingletonExtent, width, width),
				requiredTensor(attentionOutputWeightTensor, &layer.AttentionOutput, tensor.SingletonExtent, width, width),
			}
		case sequenceResidualNormalization:
			requirements = []tensorBinding{
				requiredTensor(attentionNormWeightTensor, &layer.AttentionNorm, tensor.SingletonExtent, width),
				requiredTensor("attn_norm.bias", &layer.AttentionNormBias, tensor.SingletonExtent, width),
			}
		}
		if err := bindTensorProgram(catalog, prefix, requirements); err != nil {
			return Weights{}, err
		}
	}
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensor("token_embd_norm.weight", &wav.TokenNorm, width),
		requiredTensor("token_embd_norm.bias", &wav.TokenNormBias, width),
	}); err != nil {
		return Weights{}, err
	}
	for block := uint32(tensor.FirstOffset); block < spec.ConvNextBlockCount; block++ {
		prefix := fmt.Sprintf("convnext.%d.", block)
		layer := &wav.ConvNext[block]
		if err := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensor("dw.weight", &layer.Depthwise,
				uint64(forward.SequenceDepthwiseKernel), tensor.SingletonExtent, width),
			requiredTensor("dw.bias", &layer.DepthwiseBias, tensor.SingletonExtent, width),
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
	if err := bindTensorProgram(catalog, "", []tensorBinding{
		requiredTensor(outputNormWeightTensor, &wav.OutputNorm, width),
		requiredTensor("output_norm.bias", &wav.OutputNormBias, width),
		requiredTensor(outputWeightTensor, &wav.Output, width, uint64(spec.OutputEmbeddingLength)),
		requiredTensor("output.bias", &wav.OutputBias, uint64(spec.OutputEmbeddingLength)),
	}); err != nil {
		return Weights{}, err
	}
	result.AudioDecoder = wav
	result.Output = &wav.Output
	result.OutputBias = &wav.OutputBias
	return result, nil
}
