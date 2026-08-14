package model

import "overgo/internal/gguf"

func loadRecurrentMixerLayer(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	block uint32,
	mixer RecurrentMixerPolicy,
	deltaProjection gatedDeltaPolicy,
	recurrent bool,
	queryLength, keyLength, valueLength, attentionOutputLength uint64,
) (bool, error) {
	var err error
	if mixer == recurrentMixerWeightedSelectiveScan {
		layer.Recurrent = true
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("ssm_in.weight", &layer.SSMInput, uint64(spec.EmbeddingLength), 2*uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_x.weight", &layer.SSMX, uint64(spec.SSMInnerSize), uint64(spec.SSMTimeStepRank+2*spec.SSMStateSize)),
			requiredTensorPointer("ssm_dt_norm.weight", &layer.SSMTimeStepNorm, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_dt.weight", &layer.SSMTimeStepWeight, uint64(spec.SSMTimeStepRank), uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_b_norm.weight", &layer.SSMBNorm, uint64(spec.SSMStateSize)),
			requiredTensorPointer("ssm_c_norm.weight", &layer.SSMCNorm, uint64(spec.SSMStateSize)),
			requiredTensorPointer("ssm_a", &layer.SSMA, uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_d", &layer.SSMD, uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)),
		}); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerNormalizedSelectiveScan {
		layer.Recurrent = true
		dtDimension := reducedTimeStepWidth(spec.EmbeddingLength)
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("ssm_in.weight", &layer.SSMInput, uint64(spec.EmbeddingLength), 2*uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_x.weight", &layer.SSMX, uint64(spec.SSMInnerSize), dtDimension+2*uint64(spec.SSMStateSize)),
			requiredTensorPointer("ssm_dt.weight", &layer.SSMTimeStepWeight, dtDimension, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_a", &layer.SSMA, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_d", &layer.SSMD, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)),
			requiredTensorPointer("ssm_dt_norm.weight", &layer.SSMTimeStepNorm, dtDimension),
			requiredTensorPointer("ssm_b_norm.weight", &layer.SSMBNorm, uint64(spec.SSMStateSize)),
			requiredTensorPointer("ssm_c_norm.weight", &layer.SSMCNorm, uint64(spec.SSMStateSize)),
		}); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerSelectiveScan {
		layer.Recurrent = true
		if itemErr := loadTensorRequirements(required, tensors, prefix, selectiveScanTensorRequirements(spec, layer)); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerAttentionGroupedSelectiveScan {
		convDimension := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		if itemErr := loadTensorRequirements(
			required, tensors, prefix, groupedSelectiveScanTensorRequirements(spec, layer, false),
		); itemErr != nil {
			return true, itemErr
		}
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			optionalF32TensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, convDimension),
			optionalTensorPointer("ssm_norm.weight", &layer.SSMNorm,
				uint64(spec.SSMInnerSize/spec.SSMGroupCount), uint64(spec.SSMGroupCount)),
		}); itemErr != nil {
			return true, itemErr
		}
		if _, ok := tensors[prefix+"attn_qkv.weight"]; ok {
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("attn_qkv.weight", &layer.AttentionQKV,
					uint64(spec.EmbeddingLength), queryLength+keyLength+valueLength),
				optionalF32TensorPointer("attn_qkv.bias", &layer.AttentionQKVBias,
					queryLength+keyLength+valueLength),
			}); itemErr != nil {
				return true, itemErr
			}
		} else {
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensor("attn_q.weight", &layer.AttentionQ, uint64(spec.EmbeddingLength), queryLength),
				requiredTensor("attn_k.weight", &layer.AttentionK, uint64(spec.EmbeddingLength), keyLength),
				requiredTensor("attn_v.weight", &layer.AttentionV, uint64(spec.EmbeddingLength), valueLength),
			}); itemErr != nil {
				return true, itemErr
			}
		}
		if layer.AttentionOutput, err = required(
			prefix+"attn_output.weight", attentionOutputLength, uint64(spec.EmbeddingLength),
		); err != nil {
			return true, err
		}
	} else if mixer == recurrentMixerGroupedSelectiveScan || mixer == recurrentMixerScaledGroupedSelectiveScan {
		layer.Recurrent = true
		convDimension := uint64(spec.SSMInnerSize) +
			2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
		if itemErr := loadTensorRequirements(
			required, tensors, prefix, groupedSelectiveScanTensorRequirements(spec, layer, true),
		); itemErr != nil {
			return true, itemErr
		}
		bias := optionalTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, convDimension)
		if mixer == recurrentMixerGroupedSelectiveScan {
			bias.optional = false
		}
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{bias}); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerGatedDelta {
		layer.Recurrent = recurrent
		if layer.Recurrent {
			keyDimension := uint64(spec.SSMStateSize) * uint64(spec.SSMGroupCount)
			valueDimension := uint64(spec.SSMInnerSize)
			if deltaProjection == gatedDeltaInterleavedProjections {
				if _, ok := tensors[prefix+"attn_qkv.weight"]; ok {
					qkv, qkvErr := required(
						prefix+"attn_qkv.weight", uint64(spec.EmbeddingLength),
						keyDimension*2+valueDimension,
					)
					if qkvErr != nil {
						return true, qkvErr
					}
					layer.AttentionQKV = &qkv
					attentionGate, gateErr := required(
						prefix+"attn_gate.weight", uint64(spec.EmbeddingLength), valueDimension,
					)
					if gateErr != nil {
						return true, gateErr
					}
					layer.AttentionGate = &attentionGate
				} else {
					qkvz, qkvzErr := required(
						prefix+"ssm_in.weight", uint64(spec.EmbeddingLength),
						keyDimension*2+valueDimension*2,
					)
					if qkvzErr != nil {
						return true, qkvzErr
					}
					layer.AttentionQKV = &qkvz
				}
				betaAlpha, betaAlphaErr := required(
					prefix+"ssm_ba.weight", uint64(spec.EmbeddingLength),
					2*uint64(spec.SSMTimeStepRank),
				)
				if betaAlphaErr != nil {
					return true, betaAlphaErr
				}
				layer.SSMBetaAlpha = &betaAlpha
			} else {
				qkv, qkvErr := required(
					prefix+"attn_qkv.weight", uint64(spec.EmbeddingLength),
					keyDimension*2+valueDimension,
				)
				if qkvErr != nil {
					return true, qkvErr
				}
				layer.AttentionQKV = &qkv
				attentionGate, gateErr := required(
					prefix+"attn_gate.weight", uint64(spec.EmbeddingLength), valueDimension,
				)
				if gateErr != nil {
					return true, gateErr
				}
				layer.AttentionGate = &attentionGate
			}
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), keyDimension*2+valueDimension),
				requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMTimeStepRank)),
				requiredTensorPointer("ssm_a", &layer.SSMA, uint64(spec.SSMTimeStepRank)),
				requiredTensorPointer("ssm_norm.weight", &layer.SSMNorm, uint64(spec.SSMStateSize)),
				requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, valueDimension, uint64(spec.EmbeddingLength)),
			}); itemErr != nil {
				return true, itemErr
			}
			if deltaProjection != gatedDeltaInterleavedProjections {
				if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
					requiredTensorPointer("ssm_beta.weight", &layer.SSMBeta, uint64(spec.EmbeddingLength), uint64(spec.SSMTimeStepRank)),
					requiredTensorPointer("ssm_alpha.weight", &layer.SSMAlpha, uint64(spec.EmbeddingLength), uint64(spec.SSMTimeStepRank)),
				}); itemErr != nil {
					return true, itemErr
				}
			}
		} else {
			if layer.AttentionQ, err = required(
				prefix+"attn_q.weight",
				uint64(spec.EmbeddingLength),
				queryLength*2,
			); err != nil {
				return true, err
			}
			if layer.AttentionK, err = required(
				prefix+"attn_k.weight",
				uint64(spec.EmbeddingLength),
				keyLength,
			); err != nil {
				return true, err
			}
			if layer.AttentionV, err = required(
				prefix+"attn_v.weight",
				uint64(spec.EmbeddingLength),
				valueLength,
			); err != nil {
				return true, err
			}
			if layer.AttentionOutput, err = required(
				prefix+"attn_output.weight",
				attentionOutputLength,
				uint64(spec.EmbeddingLength),
			); err != nil {
				return true, err
			}
		}
	} else if mixer == recurrentMixerShortConvolution {
		layer.Recurrent = true
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("shortconv.conv.weight", &layer.ShortConvKernel, uint64(spec.ShortConvCacheLength), uint64(spec.EmbeddingLength)),
			requiredTensorPointer("shortconv.in_proj.weight", &layer.ShortConvInput, uint64(spec.EmbeddingLength), 3*uint64(spec.EmbeddingLength)),
			requiredTensorPointer("shortconv.out_proj.weight", &layer.ShortConvOutput, uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)),
		}); itemErr != nil {
			return true, itemErr
		}
	} else {
		return false, nil
	}
	return true, nil
}

func selectiveScanTensorRequirements(spec Spec, layer *LayerWeights) []tensorRequirement {
	return []tensorRequirement{
		requiredTensorPointer("ssm_in.weight", &layer.SSMInput, uint64(spec.EmbeddingLength), 2*uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_x.weight", &layer.SSMX, uint64(spec.SSMInnerSize), uint64(spec.SSMTimeStepRank+2*spec.SSMStateSize)),
		requiredTensorPointer("ssm_dt.weight", &layer.SSMTimeStepWeight, uint64(spec.SSMTimeStepRank), uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_a", &layer.SSMA, uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_d", &layer.SSMD, uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)),
	}
}

func groupedSelectiveScanTensorRequirements(spec Spec, layer *LayerWeights, includeNorm bool) []tensorRequirement {
	convDimension := uint64(spec.SSMInnerSize) +
		2*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
	inputDimension := uint64(spec.SSMInnerSize) + convDimension + uint64(spec.SSMTimeStepRank)
	requirements := []tensorRequirement{
		requiredTensorPointer("ssm_in.weight", &layer.SSMInput, uint64(spec.EmbeddingLength), inputDimension),
		requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), convDimension),
		requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMTimeStepRank)),
		requiredTensorPointer("ssm_a", &layer.SSMA, 1, uint64(spec.SSMTimeStepRank)),
		requiredTensorPointer("ssm_d", &layer.SSMD, 1, uint64(spec.SSMTimeStepRank)),
		requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)),
	}
	if includeNorm {
		requirements = append(requirements, requiredTensorPointer(
			"ssm_norm.weight", &layer.SSMNorm,
			uint64(spec.SSMInnerSize/spec.SSMGroupCount), uint64(spec.SSMGroupCount),
		))
	}
	return requirements
}
