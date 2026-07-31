package model

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tensor/dtype"
)

// LayerWeights: initial dense transformer weight set
type LayerWeights struct {
	Recurrent               bool
	AttentionNorm           gguf.TensorInfo
	AttentionNormBias       *gguf.TensorInfo
	AttentionNorm2          *gguf.TensorInfo
	AttentionNorm2Bias      *gguf.TensorInfo
	AttentionQ              gguf.TensorInfo
	AttentionK              gguf.TensorInfo
	AttentionV              gguf.TensorInfo
	AttentionOutput         gguf.TensorInfo
	AttentionQScale         *gguf.TensorInfo
	AttentionKScale         *gguf.TensorInfo
	AttentionVScale         *gguf.TensorInfo
	AttentionOutputScale    *gguf.TensorInfo
	AttentionSubNorm        *gguf.TensorInfo
	AttentionQBias          *gguf.TensorInfo
	AttentionKBias          *gguf.TensorInfo
	AttentionVBias          *gguf.TensorInfo
	AttentionOutputBias     *gguf.TensorInfo
	AttentionQNorm          *gguf.TensorInfo
	AttentionKNorm          *gguf.TensorInfo
	AttentionQNormBias      *gguf.TensorInfo
	AttentionKNormBias      *gguf.TensorInfo
	AttentionPostNorm       *gguf.TensorInfo
	AttentionRelativeBias   *gguf.TensorInfo
	AttentionOutputGate     *gguf.TensorInfo
	RopeFactors             *gguf.TensorInfo
	FeedForwardNorm         gguf.TensorInfo
	FeedForwardNormBias     *gguf.TensorInfo
	FeedForwardExpertNorm   *gguf.TensorInfo
	FeedForwardGate         gguf.TensorInfo
	FeedForwardUp           gguf.TensorInfo
	FeedForwardDown         gguf.TensorInfo
	FeedForwardGateScale    *gguf.TensorInfo
	FeedForwardUpScale      *gguf.TensorInfo
	FeedForwardDownScale    *gguf.TensorInfo
	FeedForwardSubNorm      *gguf.TensorInfo
	FeedForwardGateBias     *gguf.TensorInfo
	FeedForwardUpBias       *gguf.TensorInfo
	FeedForwardDownBias     *gguf.TensorInfo
	FeedForwardPostNorm     *gguf.TensorInfo
	FeedForwardRouter       *gguf.TensorInfo
	FeedForwardGateExperts  *gguf.TensorInfo
	FeedForwardUpExperts    *gguf.TensorInfo
	FeedForwardDownExperts  *gguf.TensorInfo
	FeedForwardExpertBias   *gguf.TensorInfo
	FeedForwardSharedGate   *gguf.TensorInfo
	FeedForwardSharedUp     *gguf.TensorInfo
	FeedForwardSharedDown   *gguf.TensorInfo
	FeedForwardSharedRouter *gguf.TensorInfo
	ShortConvKernel         *gguf.TensorInfo
	ShortConvInput          *gguf.TensorInfo
	ShortConvOutput         *gguf.TensorInfo
	AttentionKVAMQA         *gguf.TensorInfo
	AttentionKVANorm        *gguf.TensorInfo
	AttentionKVB            *gguf.TensorInfo

	AttentionQKV     *gguf.TensorInfo
	AttentionQKVBias *gguf.TensorInfo
	AttentionGate    *gguf.TensorInfo
	SSMConv1D        *gguf.TensorInfo
	SSMTimeStep      *gguf.TensorInfo
	SSMA             *gguf.TensorInfo
	SSMBeta          *gguf.TensorInfo
	SSMAlpha         *gguf.TensorInfo
	SSMNorm          *gguf.TensorInfo
	SSMOutput        *gguf.TensorInfo
}

// Weights: validated initial Llama/Qwen3 tensor catalog
type Weights struct {
	TokenEmbedding         gguf.TensorInfo
	PositionEmbedding      *gguf.TensorInfo
	TokenEmbeddingNorm     *gguf.TensorInfo
	TokenEmbeddingNormBias *gguf.TensorInfo
	OutputNorm             gguf.TensorInfo
	OutputNormBias         *gguf.TensorInfo
	Output                 *gguf.TensorInfo
	OutputBias             *gguf.TensorInfo
	Layers                 []LayerWeights
}

// ReadWeights: validates names and shapes without loading tensor bytes
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
	tokenEmbeddingName := "token_embd.weight"
	if spec.Architecture == "codeshell" {
		if _, ok := tensors[tokenEmbeddingName]; !ok {
			tokenEmbeddingName = "output.weight"
		}
	}
	if result.TokenEmbedding, err = required(
		tokenEmbeddingName,
		uint64(spec.EmbeddingLength),
		uint64(spec.VocabularySize),
	); err != nil {
		return Weights{}, err
	}
	if spec.Architecture == "gpt2" || spec.Architecture == "starcoder" {
		positionEmbedding, positionErr := required(
			"position_embd.weight",
			uint64(spec.EmbeddingLength),
			uint64(spec.ContextLength),
		)
		if positionErr != nil {
			return Weights{}, positionErr
		}
		result.PositionEmbedding = &positionEmbedding
	}
	if spec.Architecture == "mpt" {
		if _, ok := tensors["position_embd.weight"]; ok {
			positionEmbedding, positionErr := required(
				"position_embd.weight",
				uint64(spec.EmbeddingLength),
				uint64(spec.ContextLength),
			)
			if positionErr != nil {
				return Weights{}, positionErr
			}
			result.PositionEmbedding = &positionEmbedding
		}
	}
	if spec.Architecture == "bloom" {
		tokenNorm, normErr := required("token_embd_norm.weight", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		tokenNormBias, normErr := required("token_embd_norm.bias", uint64(spec.EmbeddingLength))
		if normErr != nil {
			return Weights{}, normErr
		}
		result.TokenEmbeddingNorm = &tokenNorm
		result.TokenEmbeddingNormBias = &tokenNormBias
	}
	outputNormName := "output_norm.weight"
	if spec.Architecture == "t5encoder" {
		outputNormName = "enc.output_norm.weight"
	} else if spec.Architecture == "lfm2" || spec.Architecture == "lfm2moe" {
		outputNormName = "token_embd_norm.weight"
	}
	if !spec.UsesUnweightedLayerNorm() {
		if result.OutputNorm, err = required(outputNormName, uint64(spec.EmbeddingLength)); err != nil {
			return Weights{}, err
		}
	}
	if spec.RequiresLayerNormBias() {
		outputNormBias, biasErr := required("output_norm.bias", uint64(spec.EmbeddingLength))
		if biasErr != nil {
			return Weights{}, biasErr
		}
		result.OutputNormBias = &outputNormBias
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
	if output, ok := tensors["output.weight"]; ok &&
		spec.Architecture != "cohere2" && spec.Architecture != "command-r" {
		if output.Dimensions != 2 ||
			output.Shape[0] != uint64(spec.EmbeddingLength) ||
			output.Shape[1] != uint64(spec.VocabularySize) {
			return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", output.Name, output.Shape)
		}
		result.Output = &output
	}
	if (spec.Architecture == "internlm2" ||
		spec.Architecture == "apertus" ||
		spec.Architecture == "baichuan" ||
		spec.Architecture == "bailingmoe" ||
		spec.Architecture == "bailingmoe2" ||
		spec.Architecture == "jais" ||
		spec.Architecture == "llada-moe" ||
		spec.Architecture == "xverse" ||
		spec.Architecture == "olmo2" ||
		spec.Architecture == "nemotron" ||
		spec.Architecture == "orion" ||
		spec.Architecture == "gptneox" ||
		spec.Architecture == "phi2" ||
		spec.Architecture == "phimoe" ||
		spec.Architecture == "plamo" ||
		spec.Architecture == "stablelm" ||
		spec.Architecture == "codeshell") &&
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
	if (spec.Architecture == "phi2" || spec.Architecture == "phimoe") && result.OutputBias == nil {
		return Weights{}, errors.New(`required tensor "output.bias" is missing`)
	}

	result.Layers = make([]LayerWeights, spec.BlockCount)
	for block := uint32(0); block < spec.BlockCount; block++ {
		prefix := fmt.Sprintf("blk.%d.", block)
		queryLength := uint64(spec.LayerHeadCount(block)) * uint64(spec.KeyLength)
		keyLength := uint64(spec.LayerKVHeadCount(block)) * uint64(spec.KeyLength)
		valueLength := uint64(spec.LayerKVHeadCount(block)) * uint64(spec.ValueLength)
		attentionOutputLength := uint64(spec.LayerHeadCount(block)) * uint64(spec.ValueLength)
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
				if supportsLongRoPE(spec.Architecture) {
					continue
				}
				return Weights{}, fmt.Errorf(
					"tensor %q requires an unsupported dense-decoder feature",
					prefix+unsupported,
				)
			}
		}
		layer := &result.Layers[block]
		if spec.Architecture == "mpt" {
			for _, unsupported := range []string{
				"attn_q_norm.weight", "attn_q_norm.bias",
				"attn_k_norm.weight", "attn_k_norm.bias", "ffn_act.scales",
			} {
				if _, ok := tensors[prefix+unsupported]; ok {
					return Weights{}, fmt.Errorf(
						"tensor %q requires an unsupported MPT variant", prefix+unsupported,
					)
				}
			}
		}
		if ropeFactors, ok := tensors[prefix+"rope_freqs.weight"]; ok {
			if spec.Architecture == "qwen35" {
				return Weights{}, fmt.Errorf(
					"tensor %q requires unsupported multi-axis RoPE factors",
					ropeFactors.Name,
				)
			}
			rotaryDimensions := spec.KeyLength
			if spec.RopeDimensionCount > 0 {
				rotaryDimensions = spec.RopeDimensionCount
			}
			if rotaryDimensions%2 != 0 ||
				ropeFactors.Type != dtype.F32 ||
				ropeFactors.Dimensions != 1 ||
				ropeFactors.Shape[0] != uint64(rotaryDimensions/2) {
				return Weights{}, fmt.Errorf(
					"tensor %q has incompatible shape %v",
					ropeFactors.Name,
					ropeFactors.Shape,
				)
			}
			layer.RopeFactors = &ropeFactors
		}
		if supportsLongRoPE(spec.Architecture) {
			longName := prefix + "rope_factors_long.weight"
			shortName := prefix + "rope_factors_short.weight"
			longFactors, hasLong := tensors[longName]
			shortFactors, hasShort := tensors[shortName]
			if block > 0 {
				if !hasLong {
					longFactors, hasLong = tensors["blk.0.rope_factors_long.weight"]
				}
				if !hasShort {
					shortFactors, hasShort = tensors["blk.0.rope_factors_short.weight"]
				}
			}
			if hasLong != hasShort {
				return Weights{}, fmt.Errorf("%s LongRoPE factor tensors must both be present or absent", spec.Architecture)
			}
			if spec.RopeScalingType == "longrope" && !hasLong {
				return Weights{}, fmt.Errorf("%s LongRoPE factor tensors are missing", spec.Architecture)
			}
			if hasLong {
				for _, item := range []gguf.TensorInfo{longFactors, shortFactors} {
					if item.Type != dtype.F32 || item.Dimensions != 1 ||
						item.Shape[0] != uint64(spec.RopeDimensionCount/2) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
				}
				if spec.ContextLength > spec.OriginalContextLength {
					layer.RopeFactors = &longFactors
				} else {
					layer.RopeFactors = &shortFactors
				}
			}
		}
		if spec.Architecture != "olmo2" && !usesPostOnlyNorm(spec.Architecture) &&
			!spec.UsesUnweightedLayerNorm() {
			if layer.AttentionNorm, err = required(prefix+"attn_norm.weight", uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if spec.RequiresLayerNormBias() {
				attentionNormBias, biasErr := required(
					prefix+"attn_norm.bias",
					uint64(spec.EmbeddingLength),
				)
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionNormBias = &attentionNormBias
			}
		}
		if spec.Architecture == "falcon" {
			if item, ok := tensors[prefix+"attn_norm_2.weight"]; ok {
				if item.Dimensions != 1 || item.Shape[0] != uint64(spec.EmbeddingLength) {
					return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
				}
				layer.AttentionNorm2 = &item
				if bias, hasBias := tensors[prefix+"attn_norm_2.bias"]; hasBias {
					if bias.Dimensions != 1 || bias.Shape[0] != uint64(spec.EmbeddingLength) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.AttentionNorm2Bias = &bias
				}
			} else if _, ok := tensors[prefix+"attn_norm_2.bias"]; ok {
				return Weights{}, errors.New("Falcon secondary attention norm bias has no weight")
			}
		}
		if spec.Architecture == "bitnet" {
			attentionSubNorm, subNormErr := required(
				prefix+"attn_sub_norm.weight", uint64(spec.EmbeddingLength),
			)
			if subNormErr != nil {
				return Weights{}, subNormErr
			}
			feedForwardSubNorm, subNormErr := required(
				prefix+"ffn_sub_norm.weight", uint64(spec.FeedForwardLength),
			)
			if subNormErr != nil {
				return Weights{}, subNormErr
			}
			layer.AttentionSubNorm = &attentionSubNorm
			layer.FeedForwardSubNorm = &feedForwardSubNorm
			for name, destination := range map[string]**gguf.TensorInfo{
				"attn_q.scale":      &layer.AttentionQScale,
				"attn_k.scale":      &layer.AttentionKScale,
				"attn_v.scale":      &layer.AttentionVScale,
				"attn_output.scale": &layer.AttentionOutputScale,
				"ffn_gate.scale":    &layer.FeedForwardGateScale,
				"ffn_up.scale":      &layer.FeedForwardUpScale,
				"ffn_down.scale":    &layer.FeedForwardDownScale,
			} {
				if item, ok := tensors[prefix+name]; ok {
					if item.Type != dtype.F32 || item.Dimensions != 1 || item.Shape[0] != 1 {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
					*destination = &item
				}
			}
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
		} else if (spec.Architecture == "lfm2" || spec.Architecture == "lfm2moe") && spec.IsRecurrentLayer(block) {
			layer.Recurrent = true
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"shortconv.conv.weight": {
					[]uint64{uint64(spec.ShortConvCacheLength), uint64(spec.EmbeddingLength)},
					&layer.ShortConvKernel,
				},
				"shortconv.in_proj.weight": {
					[]uint64{uint64(spec.EmbeddingLength), 3 * uint64(spec.EmbeddingLength)},
					&layer.ShortConvInput,
				},
				"shortconv.out_proj.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)},
					&layer.ShortConvOutput,
				},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
		} else if spec.Architecture == "plm" {
			nope := uint64(spec.KeyLength - spec.RopeDimensionCount)
			if layer.AttentionQ, err = required(
				prefix+"attn_q.weight", uint64(spec.EmbeddingLength), queryLength,
			); err != nil {
				return Weights{}, err
			}
			for name, shapeAndDestination := range map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"attn_kv_a_mqa.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.KVLoRARank + spec.RopeDimensionCount)},
					&layer.AttentionKVAMQA,
				},
				"attn_kv_a_norm.weight": {
					[]uint64{uint64(spec.KVLoRARank)}, &layer.AttentionKVANorm,
				},
				"attn_kv_b.weight": {
					[]uint64{uint64(spec.KVLoRARank), uint64(spec.HeadCount) * (nope + uint64(spec.ValueLength))},
					&layer.AttentionKVB,
				},
			} {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight", uint64(spec.HeadCount)*uint64(spec.ValueLength), uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		} else {
			if spec.Architecture == "apertus" || spec.Architecture == "bailingmoe2" || spec.Architecture == "bloom" || spec.Architecture == "exaone4" || spec.Architecture == "glm4" || spec.Architecture == "openelm" || spec.Architecture == "phi2" || spec.Architecture == "phi3" || spec.Architecture == "phimoe" || spec.Architecture == "gpt2" || spec.Architecture == "gptneox" || spec.Architecture == "jais" || spec.Architecture == "mpt" || spec.Architecture == "refact" || spec.Architecture == "starcoder" ||
				spec.Architecture == "falcon" {
				_, hasQKV := tensors[prefix+"attn_qkv.weight"]
				if hasQKV || spec.Architecture == "bailingmoe2" || spec.Architecture == "bloom" || spec.Architecture == "gpt2" || spec.Architecture == "gptneox" || spec.Architecture == "jais" || spec.Architecture == "mpt" || spec.Architecture == "starcoder" || spec.Architecture == "falcon" {
					qkv, qkvErr := required(
						prefix+"attn_qkv.weight",
						uint64(spec.EmbeddingLength),
						queryLength+keyLength+valueLength,
					)
					if qkvErr != nil {
						return Weights{}, qkvErr
					}
					layer.AttentionQKV = &qkv
					if _, ok := tensors[prefix+"attn_qkv.bias"]; ok {
						qkvBias, biasErr := required(
							prefix+"attn_qkv.bias",
							queryLength+keyLength+valueLength,
						)
						if biasErr != nil {
							return Weights{}, biasErr
						}
						if qkvBias.Type != dtype.F32 {
							return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", qkvBias.Name)
						}
						layer.AttentionQKVBias = &qkvBias
					}
					if (spec.Architecture == "bloom" || spec.Architecture == "gpt2" || spec.Architecture == "gptneox" || spec.Architecture == "jais" || spec.Architecture == "starcoder") && layer.AttentionQKVBias == nil {
						return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_qkv.bias")
					}
				}
			}
			if layer.AttentionQKV == nil {
				if spec.Architecture == "apertus" || spec.Architecture == "exaone4" || spec.Architecture == "glm4" || spec.Architecture == "phi2" || spec.Architecture == "phi3" || spec.Architecture == "phimoe" {
					if _, ok := tensors[prefix+"attn_qkv.bias"]; ok {
						return Weights{}, fmt.Errorf("%s fused QKV bias has no fused weight", spec.Architecture)
					}
				}
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
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight",
				attentionOutputLength,
				uint64(spec.EmbeddingLength),
			); err != nil {
				return Weights{}, err
			}
		}
		if spec.Architecture == "apertus" || spec.Architecture == "afmoe" || spec.Architecture == "bailingmoe2" || spec.Architecture == "exaone4" || spec.Architecture == "exaone-moe" || spec.Architecture == "llada-moe" || spec.Architecture == "openelm" || spec.Architecture == "qwen3" || spec.Architecture == "qwen3moe" || spec.Architecture == "rnd1" || spec.Architecture == "laguna" || spec.Architecture == "gemma3" ||
			spec.Architecture == "maincoder" ||
			(spec.Architecture == "qwen35" && !layer.Recurrent) ||
			((spec.Architecture == "lfm2" || spec.Architecture == "lfm2moe") && !layer.Recurrent) {
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
		if spec.Architecture == "laguna" || spec.Architecture == "afmoe" {
			gate, ok := tensors[prefix+"attn_gate.weight"]
			if !ok {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_gate.weight")
			}
			if gate.Dimensions != 2 || gate.Shape[0] != uint64(spec.EmbeddingLength) {
				return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", gate.Name, gate.Shape)
			}
			heads := uint64(spec.LayerHeadCount(block))
			if spec.Architecture == "afmoe" && gate.Shape[1] != heads*uint64(spec.ValueLength) {
				return Weights{}, fmt.Errorf("tensor %q has gate width %d, need %d", gate.Name, gate.Shape[1], heads*uint64(spec.ValueLength))
			}
			if spec.Architecture == "laguna" && gate.Shape[1] != heads && gate.Shape[1] != heads*uint64(spec.ValueLength) {
				return Weights{}, fmt.Errorf("tensor %q has gate width %d, need %d or %d", gate.Name, gate.Shape[1], heads, heads*uint64(spec.ValueLength))
			}
			layer.AttentionOutputGate = &gate
		}
		if spec.Architecture == "chameleon" {
			qNorm, normErr := required(
				prefix+"attn_q_norm.weight", uint64(spec.KeyLength), uint64(spec.HeadCount),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(
				prefix+"attn_k_norm.weight", uint64(spec.KeyLength), uint64(spec.HeadCountKV),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
			if _, ok := tensors[prefix+"attn_q_norm.bias"]; ok {
				bias, biasErr := required(prefix+"attn_q_norm.bias", uint64(spec.KeyLength), uint64(spec.HeadCount))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionQNormBias = &bias
			}
			if _, ok := tensors[prefix+"attn_k_norm.bias"]; ok {
				bias, biasErr := required(prefix+"attn_k_norm.bias", uint64(spec.KeyLength), uint64(spec.HeadCountKV))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.AttentionKNormBias = &bias
			}
		}
		if spec.Architecture == "olmo2" {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", queryLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(prefix+"attn_k_norm.weight", keyLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "olmoe" {
			qNorm, normErr := required(prefix+"attn_q_norm.weight", queryLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(prefix+"attn_k_norm.weight", keyLength)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "command-r" && spec.BlockCount >= 64 {
			qNorm, normErr := required(
				prefix+"attn_q_norm.weight",
				uint64(spec.KeyLength),
				uint64(spec.HeadCount),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			kNorm, normErr := required(
				prefix+"attn_k_norm.weight",
				uint64(spec.KeyLength),
				uint64(spec.HeadCountKV),
			)
			if normErr != nil {
				return Weights{}, normErr
			}
			layer.AttentionQNorm = &qNorm
			layer.AttentionKNorm = &kNorm
		}
		if spec.Architecture == "stablelm" {
			_, hasQNorm := tensors[prefix+"attn_q_norm.weight"]
			_, hasKNorm := tensors[prefix+"attn_k_norm.weight"]
			if hasQNorm != hasKNorm {
				return Weights{}, errors.New("StableLM Q/K norm tensors must both be present or absent")
			}
			if hasQNorm {
				qNorm, normErr := required(
					prefix+"attn_q_norm.weight",
					uint64(spec.KeyLength),
					uint64(spec.HeadCount),
				)
				if normErr != nil {
					return Weights{}, normErr
				}
				kNorm, normErr := required(
					prefix+"attn_k_norm.weight",
					uint64(spec.KeyLength),
					uint64(spec.HeadCountKV),
				)
				if normErr != nil {
					return Weights{}, normErr
				}
				layer.AttentionQNorm = &qNorm
				layer.AttentionKNorm = &kNorm
			}
		}
		if !layer.Recurrent {
			if item, ok := tensors[prefix+"attn_output.bias"]; ok {
				if item.Type != dtype.F32 ||
					item.Dimensions != 1 ||
					item.Shape[0] != uint64(spec.EmbeddingLength) {
					return Weights{}, fmt.Errorf(
						"tensor %q has incompatible shape %v",
						item.Name,
						item.Shape,
					)
				}
				layer.AttentionOutputBias = &item
			}
		}
		if !layer.Recurrent && layer.AttentionQKV == nil {
			for name, shapeAndDestination := range map[string]struct {
				shape       uint64
				destination **gguf.TensorInfo
			}{
				"attn_q.bias": {layer.AttentionQ.Shape[1], &layer.AttentionQBias},
				"attn_k.bias": {layer.AttentionK.Shape[1], &layer.AttentionKBias},
				"attn_v.bias": {layer.AttentionV.Shape[1], &layer.AttentionVBias},
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
		if hasPostNorm(spec.Architecture) || spec.Architecture == "olmo2" {
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
		if spec.Architecture == "qwen35" || spec.Architecture == "seed_oss" {
			feedForwardNormName = "post_attention_norm.weight"
		}
		if spec.Architecture == "stablelm" {
			if _, ok := tensors[prefix+feedForwardNormName]; ok {
				if layer.FeedForwardNorm, err = required(prefix+feedForwardNormName, uint64(spec.EmbeddingLength)); err != nil {
					return Weights{}, err
				}
				if _, ok := tensors[prefix+"ffn_norm.bias"]; ok {
					feedForwardNormBias, biasErr := required(
						prefix+"ffn_norm.bias",
						uint64(spec.EmbeddingLength),
					)
					if biasErr != nil {
						return Weights{}, biasErr
					}
					layer.FeedForwardNormBias = &feedForwardNormBias
				}
			} else if _, ok := tensors[prefix+"ffn_norm.bias"]; ok {
				return Weights{}, errors.New("StableLM FFN norm bias has no weight")
			}
		} else if spec.Architecture != "olmo2" && !usesPostOnlyNorm(spec.Architecture) &&
			!usesParallelResidual(spec.Architecture) &&
			!spec.UsesUnweightedLayerNorm() {
			if layer.FeedForwardNorm, err = required(prefix+feedForwardNormName, uint64(spec.EmbeddingLength)); err != nil {
				return Weights{}, err
			}
			if spec.RequiresLayerNormBias() {
				feedForwardNormBias, biasErr := required(
					prefix+"ffn_norm.bias",
					uint64(spec.EmbeddingLength),
				)
				if biasErr != nil {
					return Weights{}, biasErr
				}
				layer.FeedForwardNormBias = &feedForwardNormBias
			}
		}
		if (spec.Architecture == "llama" && spec.ExpertCount > 0) || spec.Architecture == "arctic" || spec.Architecture == "bailingmoe" || spec.Architecture == "llada-moe" || spec.Architecture == "qwen3moe" || spec.Architecture == "qwen2moe" || spec.Architecture == "olmoe" || spec.Architecture == "phimoe" || spec.Architecture == "rnd1" ||
			spec.Architecture == "granitemoe" ||
			(spec.Architecture == "deepseek" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "bailingmoe2" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "lfm2moe" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "exaone-moe" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "afmoe" && block >= spec.LeadingDenseBlocks) ||
			(spec.Architecture == "laguna" && block >= spec.LeadingDenseBlocks) {
			expertTensors := map[string]struct {
				shape       []uint64
				destination **gguf.TensorInfo
			}{
				"ffn_gate_inp.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)},
					&layer.FeedForwardRouter,
				},
				"ffn_up_exps.weight": {
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)},
					&layer.FeedForwardUpExperts,
				},
				"ffn_down_exps.weight": {
					[]uint64{uint64(spec.ExpertFeedForward), uint64(spec.EmbeddingLength), uint64(spec.ExpertCount)},
					&layer.FeedForwardDownExperts,
				},
			}
			if spec.Architecture != "granitemoe" {
				expertTensors["ffn_gate_exps.weight"] = struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					[]uint64{uint64(spec.EmbeddingLength), uint64(spec.ExpertFeedForward), uint64(spec.ExpertCount)},
					&layer.FeedForwardGateExperts,
				}
			}
			for name, shapeAndDestination := range expertTensors {
				item, itemErr := required(prefix+name, shapeAndDestination.shape...)
				if itemErr != nil {
					return Weights{}, itemErr
				}
				*shapeAndDestination.destination = &item
			}
			if spec.Architecture == "granitemoe" {
				if item, ok := tensors[prefix+"ffn_gate_exps.weight"]; ok {
					if item.Dimensions != 3 || item.Shape[0] != uint64(spec.EmbeddingLength) ||
						item.Shape[1] != uint64(spec.ExpertFeedForward) || item.Shape[2] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", item.Name, item.Shape)
					}
					layer.FeedForwardGateExperts = &item
				}
				if spec.SharedExpertFF > 0 {
					for name, shapeAndDestination := range map[string]struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{
						"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
						"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
						"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
					} {
						item, itemErr := required(prefix+name, shapeAndDestination.shape...)
						if itemErr != nil {
							return Weights{}, itemErr
						}
						*shapeAndDestination.destination = &item
					}
				}
			}
			if spec.Architecture == "laguna" || spec.Architecture == "afmoe" {
				shared := map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"exp_probs_b.bias": {
						[]uint64{uint64(spec.ExpertCount)}, &layer.FeedForwardExpertBias,
					},
				}
				if spec.SharedExpertFF > 0 {
					shared["ffn_gate_shexp.weight"] = struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate}
					shared["ffn_up_shexp.weight"] = struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp}
					shared["ffn_down_shexp.weight"] = struct {
						shape       []uint64
						destination **gguf.TensorInfo
					}{[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown}
				}
				for name, shapeAndDestination := range shared {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "exaone-moe" || spec.Architecture == "bailingmoe2" {
				if bias, ok := tensors[prefix+"exp_probs_b.bias"]; ok {
					if bias.Type != dtype.F32 || bias.Dimensions != 1 || bias.Shape[0] != uint64(spec.ExpertCount) {
						return Weights{}, fmt.Errorf("tensor %q has incompatible shape %v", bias.Name, bias.Shape)
					}
					layer.FeedForwardExpertBias = &bias
				}
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "bailingmoe" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "deepseek" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":   {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight": {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "qwen2moe" {
				for name, shapeAndDestination := range map[string]struct {
					shape       []uint64
					destination **gguf.TensorInfo
				}{
					"ffn_gate_inp_shexp.weight": {[]uint64{uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedRouter},
					"ffn_gate_shexp.weight":     {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedGate},
					"ffn_up_shexp.weight":       {[]uint64{uint64(spec.EmbeddingLength), uint64(spec.SharedExpertFF)}, &layer.FeedForwardSharedUp},
					"ffn_down_shexp.weight":     {[]uint64{uint64(spec.SharedExpertFF), uint64(spec.EmbeddingLength)}, &layer.FeedForwardSharedDown},
				} {
					item, itemErr := required(prefix+name, shapeAndDestination.shape...)
					if itemErr != nil {
						return Weights{}, itemErr
					}
					*shapeAndDestination.destination = &item
				}
			}
			if spec.Architecture == "lfm2moe" {
				bias, biasErr := required(prefix+"exp_probs_b.bias", uint64(spec.ExpertCount))
				if biasErr != nil {
					return Weights{}, biasErr
				}
				if bias.Type != dtype.F32 {
					return Weights{}, fmt.Errorf("tensor %q must use F32 bias storage", bias.Name)
				}
				layer.FeedForwardExpertBias = &bias
			}
			if spec.Architecture == "phimoe" && layer.AttentionOutputBias == nil {
				return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+"attn_output.bias")
			}
			if spec.Architecture == "arctic" {
				expertNorm, normErr := required(prefix+"ffn_norm_exps.weight", uint64(spec.EmbeddingLength))
				if normErr != nil {
					return Weights{}, normErr
				}
				layer.FeedForwardExpertNorm = &expertNorm
			} else {
				continue
			}
		}
		feedForwardLength := spec.LayerFeedForwardLength(block)
		if spec.Architecture == "arctic" {
			feedForwardLength = spec.EmbeddingLength
		}
		if !usesGateFreeFFN(spec.Architecture) {
			if layer.FeedForwardGate, err = required(
				prefix+"ffn_gate.weight",
				uint64(spec.EmbeddingLength),
				uint64(feedForwardLength),
			); err != nil {
				return Weights{}, err
			}
		}
		feedForwardUpLength := uint64(feedForwardLength)
		if usesFusedGateUp(spec.Architecture) {
			feedForwardUpLength *= 2
		}
		if layer.FeedForwardUp, err = required(
			prefix+"ffn_up.weight",
			uint64(spec.EmbeddingLength),
			feedForwardUpLength,
		); err != nil {
			return Weights{}, err
		}
		if layer.FeedForwardDown, err = required(
			prefix+"ffn_down.weight",
			uint64(feedForwardLength),
			uint64(spec.EmbeddingLength),
		); err != nil {
			return Weights{}, err
		}
		for name, shapeAndDestination := range map[string]struct {
			shape       uint64
			destination **gguf.TensorInfo
		}{
			"ffn_gate.bias": {uint64(feedForwardLength), &layer.FeedForwardGateBias},
			"ffn_up.bias":   {uint64(feedForwardLength), &layer.FeedForwardUpBias},
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
		if usesSequentialGELU(spec.Architecture) {
			for name, item := range map[string]*gguf.TensorInfo{
				"attn_output.bias": layer.AttentionOutputBias,
				"ffn_up.bias":      layer.FeedForwardUpBias,
				"ffn_down.bias":    layer.FeedForwardDownBias,
			} {
				if item == nil {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+name)
				}
			}
		}
		if spec.Architecture == "jais2" {
			for name, item := range map[string]*gguf.TensorInfo{
				"attn_q.bias":      layer.AttentionQBias,
				"attn_k.bias":      layer.AttentionKBias,
				"attn_v.bias":      layer.AttentionVBias,
				"attn_output.bias": layer.AttentionOutputBias,
				"ffn_up.bias":      layer.FeedForwardUpBias,
				"ffn_down.bias":    layer.FeedForwardDownBias,
			} {
				if item == nil {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+name)
				}
			}
		}
		if spec.Architecture == "jais" {
			for name, item := range map[string]*gguf.TensorInfo{
				"attn_output.bias": layer.AttentionOutputBias,
				"ffn_gate.bias":    layer.FeedForwardGateBias,
				"ffn_up.bias":      layer.FeedForwardUpBias,
				"ffn_down.bias":    layer.FeedForwardDownBias,
			} {
				if item == nil {
					return Weights{}, fmt.Errorf("required tensor %q is missing", prefix+name)
				}
			}
		}
	}
	return result, nil
}
