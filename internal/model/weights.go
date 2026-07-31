package model

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

// LayerWeights is the initial dense transformer weight set.
type LayerWeights struct {
	Recurrent             bool
	AttentionNorm         gguf.TensorInfo
	AttentionQ            gguf.TensorInfo
	AttentionK            gguf.TensorInfo
	AttentionV            gguf.TensorInfo
	AttentionOutput       gguf.TensorInfo
	AttentionQBias        *gguf.TensorInfo
	AttentionKBias        *gguf.TensorInfo
	AttentionVBias        *gguf.TensorInfo
	AttentionOutputBias   *gguf.TensorInfo
	AttentionQNorm        *gguf.TensorInfo
	AttentionKNorm        *gguf.TensorInfo
	AttentionPostNorm     *gguf.TensorInfo
	AttentionRelativeBias *gguf.TensorInfo
	RopeFactors           *gguf.TensorInfo
	FeedForwardNorm       gguf.TensorInfo
	FeedForwardGate       gguf.TensorInfo
	FeedForwardUp         gguf.TensorInfo
	FeedForwardDown       gguf.TensorInfo
	FeedForwardGateBias   *gguf.TensorInfo
	FeedForwardUpBias     *gguf.TensorInfo
	FeedForwardDownBias   *gguf.TensorInfo
	FeedForwardPostNorm   *gguf.TensorInfo

	AttentionQKV  *gguf.TensorInfo
	AttentionGate *gguf.TensorInfo
	SSMConv1D     *gguf.TensorInfo
	SSMTimeStep   *gguf.TensorInfo
	SSMA          *gguf.TensorInfo
	SSMBeta       *gguf.TensorInfo
	SSMAlpha      *gguf.TensorInfo
	SSMNorm       *gguf.TensorInfo
	SSMOutput     *gguf.TensorInfo
}

// Weights is a validated initial Llama/Qwen3 tensor catalog.
type Weights struct {
	TokenEmbedding gguf.TensorInfo
	OutputNorm     gguf.TensorInfo
	Output         *gguf.TensorInfo
	OutputBias     *gguf.TensorInfo
	Layers         []LayerWeights
}

// ReadWeights validates names and shapes without loading tensor bytes.
func ReadWeights(file *gguf.File, spec Spec) (Weights, error) {
	if file == nil {
		return Weights{}, errors.New("model file is nil")
	}
	tensors := make(map[string]gguf.TensorInfo, len(file.Tensors))
	for _, item := range file.Tensors {
		if _, exists := tensors[item.Name]; exists {
			return Weights{}, fmt.Errorf("duplicate tensor %q", item.Name)
		}
		tensors[item.Name] = item
	}
	required := func(name string, shape ...uint64) (gguf.TensorInfo, error) {
		item, ok := tensors[name]
		if !ok {
			return gguf.TensorInfo{}, fmt.Errorf("required tensor %q is missing", name)
		}
		if item.Dimensions != uint32(len(shape)) {
			return gguf.TensorInfo{}, fmt.Errorf(
				"tensor %q has rank %d, need %d",
				name,
				item.Dimensions,
				len(shape),
			)
		}
		for index, dimension := range shape {
			if item.Shape[index] != dimension {
				return gguf.TensorInfo{}, fmt.Errorf(
					"tensor %q dimension %d is %d, need %d",
					name,
					index,
					item.Shape[index],
					dimension,
				)
			}
		}
		return item, nil
	}

	var result Weights
	var err error
	if result.TokenEmbedding, err = required(
		"token_embd.weight",
		uint64(spec.EmbeddingLength),
		uint64(spec.VocabularySize),
	); err != nil {
		return Weights{}, err
	}
	outputNormName := "output_norm.weight"
	if spec.Architecture == "t5encoder" {
		outputNormName = "enc.output_norm.weight"
	}
	if result.OutputNorm, err = required(outputNormName, uint64(spec.EmbeddingLength)); err != nil {
		return Weights{}, err
	}
	if spec.Architecture == "t5encoder" {
		result.Layers = make([]LayerWeights, spec.BlockCount)
		queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
		keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
		valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
		attentionOutputLength := uint64(spec.HeadCount) * uint64(spec.ValueLength)
		for block := uint32(0); block < spec.BlockCount; block++ {
			prefix := fmt.Sprintf("enc.blk.%d.", block)
			layer := &result.Layers[block]
			if layer.AttentionNorm, err = required(prefix+"attn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if layer.AttentionQ, err = required(prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength); err != nil {
				return Weights{}, err
			}
			if layer.AttentionK, err = required(prefix+"attn_k.weight", uint64(spec.EmbeddingLength), keyLength); err != nil {
				return Weights{}, err
			}
			if layer.AttentionV, err = required(prefix+"attn_v.weight", uint64(spec.EmbeddingLength), valueLength); err != nil {
				return Weights{}, err
			}
			if layer.AttentionOutput, err = required(prefix+"attn_o.weight", attentionOutputLength, uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			relativeBias, biasErr := required(
				prefix+"attn_rel_b.weight",
				uint64(spec.HeadCount),
				uint64(spec.RelativeBuckets),
			)
			if biasErr != nil {
				return Weights{}, biasErr
			}
			layer.AttentionRelativeBias = &relativeBias
			if layer.FeedForwardNorm, err = required(prefix+"ffn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if layer.FeedForwardGate, err = required(prefix+"ffn_gate.weight", uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); err != nil {
				return Weights{}, err
			}
			if layer.FeedForwardUp, err = required(prefix+"ffn_up.weight", uint64(spec.EmbeddingLength), uint64(spec.FeedForwardLength)); err != nil {
				return Weights{}, err
			}
			if layer.FeedForwardDown, err = required(prefix+"ffn_down.weight", uint64(spec.FeedForwardLength), uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
		}
		return result, nil
	}
	if output, ok := tensors["output.weight"]; ok {
		if output.Dimensions != 2 ||
			output.Shape[0] != uint64(spec.EmbeddingLength) ||
			output.Shape[1] != uint64(spec.VocabularySize) {
			return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", output.Name, output.Shape)
		}
		result.Output = &output
	}
	if (spec.Architecture == "internlm2" || spec.Architecture == "xverse") &&
		result.Output == nil {
		return Weights{}, errors.New(`required tensor "output.weight" is missing`)
	}
	if outputBias, ok := tensors["output.bias"]; ok {
		if outputBias.Type != dtype.F32 ||
			outputBias.Dimensions != 1 ||
			outputBias.Shape[0] != uint64(spec.VocabularySize) {
			return Weights{}, fmt.Errorf(
				"tensor %q has incompatible shape %v",
				outputBias.Name,
				outputBias.Shape,
			)
		}
		result.OutputBias = &outputBias
	}

	queryLength := uint64(spec.HeadCount) * uint64(spec.KeyLength)
	keyLength := uint64(spec.HeadCountKV) * uint64(spec.KeyLength)
	valueLength := uint64(spec.HeadCountKV) * uint64(spec.ValueLength)
	attentionOutputLength := uint64(spec.HeadCount) * uint64(spec.ValueLength)
	result.Layers = make([]LayerWeights, spec.BlockCount)
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		biasNames := []string{
			"attn_q.bias",
			"attn_k.bias",
			"attn_v.bias",
			"attn_output.bias",
			"ffn_gate.bias",
			"ffn_up.bias",
			"ffn_down.bias",
		}
		if spec.Architecture == "qwen35" {
			for _, name := range biasNames {
				if _, ok := tensors[prefix+name]; ok {
					return Weights{}, fmt.Errorf(
						"tensor %q requires unsupported Qwen3.5 projection biases",
						prefix+name,
					)
				}
			}
		}
		for _, unsupported := range []string{
			"rope_factors_long.weight",
			"rope_factors_short.weight",
		} {
			if _, ok := tensors[prefix+unsupported]; ok {
				return Weights{}, fmt.Errorf(
					"tensor %q requires an unsupported dense-decoder feature",
					prefix+unsupported,
				)
			}
		}
		layer := &result.Layers[block]
		if ropeFactors, ok := tensors[prefix+"rope_freqs.weight"]; ok {
			if spec.Architecture == "qwen35" {
				return Weights{}, fmt.Errorf(
					"tensor %q requires unsupported multi-axis RoPE factors",
					ropeFactors.Name,
				)
			}
			if spec.KeyLength%2 != 0 ||
				ropeFactors.Type != dtype.F32 ||
				ropeFactors.Dimensions != 1 ||
				ropeFactors.Shape[0] != uint64(spec.KeyLength/2) {
				return Weights{}, fmt.Errorf(
					"tensor %q has incompatible shape %v",
					ropeFactors.Name,
					ropeFactors.Shape,
				)
			}
			layer.RopeFactors = &ropeFactors
		}
		if layer.AttentionNorm, err = required(prefix+"attn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
			return Weights{}, err
		}
		if spec.Architecture == "qwen35" {
			layer.Recurrent = spec.IsRecurrentLayer(block)
			if layer.Recurrent {
				keyDimension := uint64(spec.SSMStateSize) * uint64(spec.SSMGroupCount)
				valueDimension := uint64(spec.SSMInnerSize)
				qkv, qkvErr := required(
					prefix+"attn_qkv.weight",
					uint64(spec.EmbeddingLength),
					keyDimension*2+valueDimension,
				)
				if qkvErr != nil {
					return Weights{}, qkvErr
				}
				layer.AttentionQKV = &qkv
				attentionGate, gateErr := required(
					prefix+"attn_gate.weight",
					uint64(spec.EmbeddingLength),
					valueDimension,
				)
				if gateErr != nil {
					return Weights{}, gateErr
				}
				layer.AttentionGate = &attentionGate
				for name, shape := range map[string][]uint64{
					"ssm_conv1d.weight": {
						uint64(spec.SSMConvKernel),
						keyDimension*2 + valueDimension,
					},
					"ssm_dt.bias":      {uint64(spec.SSMTimeStepRank)},
					"ssm_a":            {uint64(spec.SSMTimeStepRank)},
					"ssm_beta.weight":  {uint64(spec.EmbeddingLength), uint64(spec.SSMTimeStepRank)},
					"ssm_alpha.weight": {uint64(spec.EmbeddingLength), uint64(spec.SSMTimeStepRank)},
					"ssm_norm.weight":  {uint64(spec.SSMStateSize)},
					"ssm_out.weight":   {valueDimension, uint64(spec.EmbeddingLength)},
				} {
					item, itemErr := required(prefix+name, shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					switch name {
					case "ssm_conv1d.weight":
						layer.SSMConv1D = &item
					case "ssm_dt.bias":
						layer.SSMTimeStep = &item
					case "ssm_a":
						layer.SSMA = &item
					case "ssm_beta.weight":
						layer.SSMBeta = &item
					case "ssm_alpha.weight":
						layer.SSMAlpha = &item
					case "ssm_norm.weight":
						layer.SSMNorm = &item
					case "ssm_out.weight":
						layer.SSMOutput = &item
					}
				}
			} else {
				if layer.AttentionQ, err = required(
					prefix+"attn_q.weight",
					uint64(spec.EmbeddingLength),
					queryLength*2,
				); err != nil {
					return Weights{}, err
				}
				if layer.AttentionK, err = required(
					prefix+"attn_k.weight",
					uint64(spec.EmbeddingLength),
					keyLength,
				); err != nil {
					return Weights{}, err
				}
				if layer.AttentionV, err = required(
					prefix+"attn_v.weight",
					uint64(spec.EmbeddingLength),
					valueLength,
				); err != nil {
					return Weights{}, err
				}
				if layer.AttentionOutput, err = required(
					prefix+"attn_output.weight",
					attentionOutputLength,
					uint64(spec.EmbeddingLength),
				); err != nil {
					return Weights{}, err
				}
			}
		} else {
			if layer.AttentionQ, err = required(
				prefix+"attn_q.weight",
				uint64(spec.EmbeddingLength),
				queryLength,
			); err != nil {
				return Weights{}, err
			}
			if layer.AttentionK, err = required(
				prefix+"attn_k.weight",
				uint64(spec.EmbeddingLength),
				keyLength,
			); err != nil {
				return Weights{}, err
			}
			if layer.AttentionV, err = required(
				prefix+"attn_v.weight",
				uint64(spec.EmbeddingLength),
				valueLength,
			); err != nil {
				return Weights{}, err
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight",
				attentionOutputLength,
				uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		}
		if spec.Architecture == "qwen3" || spec.Architecture == "gemma3" ||
			(spec.Architecture == "qwen35" && !layer.Recurrent) {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", uint64(spec.KeyLength))
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(prefix+"attn_k_norm.weight", uint64(spec.KeyLength))
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if !layer.Recurrent {
			for name, shapeAndDestination := range map[string]struct {
				shape       uint64
				destination **gguf.TensorInfo
			}{
				"attn_q.bias":      {layer.AttentionQ.Shape[1], &layer.AttentionQBias},
				"attn_k.bias":      {layer.AttentionK.Shape[1], &layer.AttentionKBias},
				"attn_v.bias":      {layer.AttentionV.Shape[1], &layer.AttentionVBias},
				"attn_output.bias": {uint64(spec.EmbeddingLength), &layer.AttentionOutputBias},
			} {
				if item, ok := tensors[prefix+name]; ok {
					if item.Type != dtype.F32 ||
						item.Dimensions != 1 ||
						item.Shape[0] != shapeAndDestination.shape {
						return Weights{}, fmt.Errorf(
							"tensor %q has incompatible shape %v",
							item.Name,
							item.Shape,
						)
					}
					*shapeAndDestination.destination = &item
				}
			}
		}
		if hasGemmaPostNorm(spec.Architecture) {
			attentionPostNorm, normErr := required(
				prefix+"post_attention_norm.weight",
				uint64(spec.EmbeddingLength),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			feedForwardPostNorm, normErr := required(
				prefix+"post_ffw_norm.weight",
				uint64(spec.EmbeddingLength),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionPostNorm = &attentionPostNorm
			layer.FeedForwardPostNorm = &feedForwardPostNorm
		}
		feedForwardNormName := "ffn_norm.weight"
		if spec.Architecture == "qwen35" {
			feedForwardNormName = "post_attention_norm.weight"
		}
		if layer.FeedForwardNorm, err = required(prefix+feedForwardNormName, uint64(spec.EmbeddingLength)); err != nil {
			return Weights{}, err
		}
		if layer.FeedForwardGate, err = required(
			prefix+"ffn_gate.weight",
			uint64(spec.EmbeddingLength),
			uint64(spec.FeedForwardLength),
		); err != nil {
			return Weights{}, err
		}
		if layer.FeedForwardUp, err = required(
			prefix+"ffn_up.weight",
			uint64(spec.EmbeddingLength),
			uint64(spec.FeedForwardLength),
		); err != nil {
			return Weights{}, err
		}
		if layer.FeedForwardDown, err = required(
			prefix+"ffn_down.weight",
			uint64(spec.FeedForwardLength),
			uint64(spec.EmbeddingLength),
		); err != nil {
			return Weights{}, err
		}
		for name, shapeAndDestination := range map[string]struct {
			shape       uint64
			destination **gguf.TensorInfo
		}{
			"ffn_gate.bias": {uint64(spec.FeedForwardLength), &layer.FeedForwardGateBias},
			"ffn_up.bias":   {uint64(spec.FeedForwardLength), &layer.FeedForwardUpBias},
			"ffn_down.bias": {uint64(spec.EmbeddingLength), &layer.FeedForwardDownBias},
		} {
			if item, ok := tensors[prefix+name]; ok {
				if item.Type != dtype.F32 ||
					item.Dimensions != 1 ||
					item.Shape[0] != shapeAndDestination.shape {
					return Weights{}, fmt.Errorf(
						"tensor %q has incompatible shape %v",
						item.Name,
						item.Shape,
					)
				}
				*shapeAndDestination.destination = &item
			}
		}
	}
	return result, nil
}
