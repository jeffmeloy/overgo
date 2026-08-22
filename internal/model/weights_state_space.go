package model

import "overgo/internal/tensor"

func loadRecurrentMixerLayer(
	catalog weightCatalog,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	plan LayerPlan,
	queryLength, keyLength, valueLength, attentionOutputLength uint64,
) (bool, error) {
	mixer, recurrent := plan.Mixer, plan.Recurrent
	if mixer == recurrentMixerWeightedSelectiveScan {
		layer.Recurrent = true
		if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer("ssm_in.weight", &layer.SSMInput,
				uint64(spec.EmbeddingLength), tensor.PairedExtent*uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_x.weight", &layer.SSMX, uint64(spec.SSMInnerSize),
				uint64(spec.SSMTimeStepRank+tensor.PairedExtent*spec.SSMStateSize)),
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
		if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredRelationalTensorPointer(
				"ssm_dt_norm.weight", &layer.SSMTimeStepNorm, tensor.SingletonExtent, true,
			),
		}); itemErr != nil {
			return true, itemErr
		}
		dtDimension := layer.SSMTimeStepNorm.Shape[tensor.FirstOffset]
		if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer("ssm_in.weight", &layer.SSMInput,
				uint64(spec.EmbeddingLength), tensor.PairedExtent*uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)),
			requiredTensorPointer("ssm_x.weight", &layer.SSMX, uint64(spec.SSMInnerSize),
				dtDimension+tensor.PairedExtent*uint64(spec.SSMStateSize)),
			requiredTensorPointer("ssm_dt.weight", &layer.SSMTimeStepWeight, dtDimension, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_a", &layer.SSMA, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_d", &layer.SSMD, uint64(spec.SSMTimeStepRank)),
			requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)),
			requiredTensorPointer("ssm_b_norm.weight", &layer.SSMBNorm, uint64(spec.SSMStateSize)),
			requiredTensorPointer("ssm_c_norm.weight", &layer.SSMCNorm, uint64(spec.SSMStateSize)),
		}); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerSelectiveScan {
		layer.Recurrent = true
		if itemErr := bindTensorProgram(catalog, prefix, selectiveScanTensorRequirements(spec, layer)); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerAttentionGroupedSelectiveScan {
		if itemErr := bindTensorProgram(
			catalog, prefix, groupedSelectiveScanTensorRequirements(spec, layer, false, mixer),
		); itemErr != nil {
			return true, itemErr
		}
		if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
			optionalTensorPointer("ssm_norm.weight", &layer.SSMNorm,
				uint64(spec.SSMInnerSize/spec.SSMGroupCount), uint64(spec.SSMGroupCount)),
		}); itemErr != nil {
			return true, itemErr
		}
		if itemErr := loadStandardAttentionCatalog(
			catalog, prefix, spec, layer,
			queryLength, keyLength, valueLength, attentionOutputLength,
		); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerSparseGroupedSelectiveScan {
		if !recurrent {
			return false, nil
		}
		layer.Recurrent = true
		if itemErr := bindTensorProgram(
			catalog, prefix, groupedSelectiveScanTensorRequirements(spec, layer, true, mixer),
		); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerGroupedSelectiveScan || mixer == recurrentMixerScaledGroupedSelectiveScan {
		layer.Recurrent = true
		if itemErr := bindTensorProgram(
			catalog, prefix, groupedSelectiveScanTensorRequirements(spec, layer, true, mixer),
		); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerGatedDelta {
		layer.Recurrent = recurrent
		if layer.Recurrent {
			keyDimension := uint64(spec.SSMStateSize) * uint64(spec.SSMGroupCount)
			valueDimension := uint64(spec.SSMInnerSize)
			if plan.AttentionGraph.deltaProjection == gatedDeltaInterleavedProjections {
				program := []tensorBinding{
					requiredTensorPointer("ssm_ba.weight", &layer.SSMBetaAlpha,
						uint64(spec.EmbeddingLength), tensor.PairedExtent*uint64(spec.SSMTimeStepRank)),
				}
				if _, ok := catalog.tensors[prefix+"attn_qkv.weight"]; ok {
					program = append(program,
						requiredTensorPointer("attn_qkv.weight", &layer.AttentionQKV,
							uint64(spec.EmbeddingLength), keyDimension*tensor.PairedExtent+valueDimension),
						requiredTensorPointer("attn_gate.weight", &layer.AttentionGate,
							uint64(spec.EmbeddingLength), valueDimension),
					)
				} else {
					program = append(program, requiredTensorPointer(
						"ssm_in.weight", &layer.AttentionQKV, uint64(spec.EmbeddingLength),
						keyDimension*tensor.PairedExtent+valueDimension*tensor.PairedExtent,
					))
				}
				if itemErr := bindTensorProgram(catalog, prefix, program); itemErr != nil {
					return true, itemErr
				}
			} else {
				if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					requiredTensorPointer("attn_qkv.weight", &layer.AttentionQKV,
						uint64(spec.EmbeddingLength), keyDimension*tensor.PairedExtent+valueDimension),
					requiredTensorPointer("attn_gate.weight", &layer.AttentionGate,
						uint64(spec.EmbeddingLength), valueDimension),
				}); itemErr != nil {
					return true, itemErr
				}
			}
			if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D,
					uint64(spec.SSMConvKernel), keyDimension*tensor.PairedExtent+valueDimension),
				requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMTimeStepRank)),
				requiredTensorPointer("ssm_a", &layer.SSMA, uint64(spec.SSMTimeStepRank)),
				requiredTensorPointer("ssm_norm.weight", &layer.SSMNorm, uint64(spec.SSMStateSize)),
				requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, valueDimension, uint64(spec.EmbeddingLength)),
			}); itemErr != nil {
				return true, itemErr
			}
			if plan.AttentionGraph.deltaProjection != gatedDeltaInterleavedProjections {
				if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
					requiredTensorPointer("ssm_beta.weight", &layer.SSMBeta, uint64(spec.EmbeddingLength), uint64(spec.SSMTimeStepRank)),
					requiredTensorPointer("ssm_alpha.weight", &layer.SSMAlpha, uint64(spec.EmbeddingLength), uint64(spec.SSMTimeStepRank)),
				}); itemErr != nil {
					return true, itemErr
				}
			}
		} else {
			if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
				requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ,
					uint64(spec.EmbeddingLength), queryLength*tensor.PairedExtent),
				requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK,
					uint64(spec.EmbeddingLength), keyLength),
				requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV,
					uint64(spec.EmbeddingLength), valueLength),
				requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput,
					attentionOutputLength, uint64(spec.EmbeddingLength)),
			}); itemErr != nil {
				return true, itemErr
			}
		}
	} else if mixer == recurrentMixerKeyedDelta {
		layer.Recurrent = recurrent
		if !recurrent {
			return true, loadLatentAttentionCatalog(
				catalog, prefix, spec, layer, plan, queryLength, attentionOutputLength,
			)
		}
		inner, width := uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)
		convShape := []uint64{uint64(spec.SSMConvKernel), tensor.SingletonExtent, inner}
		convShape4D := append(append([]uint64(nil), convShape...), tensor.SingletonExtent)
		if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer(attentionQueryWeightTensor, &layer.AttentionQ, width, inner),
			requiredTensorPointer(attentionKeyWeightTensor, &layer.AttentionK, width, inner),
			requiredTensorPointer(attentionValueWeightTensor, &layer.AttentionV, width, inner),
			requiredTensorPointer(attentionOutputWeightTensor, &layer.AttentionOutput, inner, width),
			requiredTensorPointerShapes("ssm_conv1d_q.weight", &layer.SSMQueryConv, convShape, convShape4D),
			requiredTensorPointerShapes("ssm_conv1d_k.weight", &layer.SSMKeyConv, convShape, convShape4D),
			requiredTensorPointerShapes("ssm_conv1d_v.weight", &layer.SSMValueConv, convShape, convShape4D),
			requiredTensorPointer("ssm_f_a.weight", &layer.SSMForgetA, width, uint64(spec.KDAHeadDim)),
			requiredTensorPointer("ssm_f_b.weight", &layer.SSMForgetB, uint64(spec.KDAHeadDim), inner),
			requiredTensorPointer("ssm_beta.weight", &layer.SSMBeta, width, uint64(spec.HeadCount)),
			requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, inner),
			requiredTensorPointer("ssm_g_a.weight", &layer.SSMOutputGateA, width, uint64(spec.KDAHeadDim)),
			requiredTensorPointer("ssm_g_b.weight", &layer.SSMOutputGateB, uint64(spec.KDAHeadDim), inner),
			requiredTensorPointer("ssm_norm.weight", &layer.SSMNorm, uint64(spec.KDAHeadDim)),
			requiredTensorPointerShapes("ssm_a", &layer.SSMA,
				[]uint64{tensor.SingletonExtent, uint64(spec.HeadCount)},
				[]uint64{tensor.SingletonExtent, uint64(spec.HeadCount), tensor.SingletonExtent, tensor.SingletonExtent}),
		}); itemErr != nil {
			return true, itemErr
		}
	} else if mixer == recurrentMixerShortConvolution {
		layer.Recurrent = true
		if itemErr := bindTensorProgram(catalog, prefix, []tensorBinding{
			requiredTensorPointer("shortconv.conv.weight", &layer.ShortConvKernel, uint64(spec.ShortConvCacheLength), uint64(spec.EmbeddingLength)),
			requiredTensorPointer("shortconv.in_proj.weight", &layer.ShortConvInput,
				uint64(spec.EmbeddingLength), tensor.TripleExtent*uint64(spec.EmbeddingLength)),
			requiredTensorPointer("shortconv.out_proj.weight", &layer.ShortConvOutput, uint64(spec.EmbeddingLength), uint64(spec.EmbeddingLength)),
		}); itemErr != nil {
			return true, itemErr
		}
	} else {
		return false, nil
	}
	return true, nil
}

func selectiveScanTensorRequirements(spec Spec, layer *LayerWeights) []tensorBinding {
	return []tensorBinding{
		requiredTensorPointer("ssm_in.weight", &layer.SSMInput,
			uint64(spec.EmbeddingLength), tensor.PairedExtent*uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_x.weight", &layer.SSMX, uint64(spec.SSMInnerSize),
			uint64(spec.SSMTimeStepRank+tensor.PairedExtent*spec.SSMStateSize)),
		requiredTensorPointer("ssm_dt.weight", &layer.SSMTimeStepWeight, uint64(spec.SSMTimeStepRank), uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_a", &layer.SSMA, uint64(spec.SSMStateSize), uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_d", &layer.SSMD, uint64(spec.SSMInnerSize)),
		requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)),
	}
}

func groupedSelectiveScanTensorRequirements(
	spec Spec,
	layer *LayerWeights,
	includeNorm bool,
	mixer RecurrentMixerPolicy,
) []tensorBinding {
	convDimension := groupedStateSpaceConvolutionWidth(spec)
	inputDimension := uint64(spec.SSMInnerSize) + convDimension + uint64(spec.SSMTimeStepRank)
	requirements := []tensorBinding{
		requiredTensorPointer("ssm_in.weight", &layer.SSMInput, uint64(spec.EmbeddingLength), inputDimension),
		requiredTensorPointer("ssm_conv1d.weight", &layer.SSMConv1D, uint64(spec.SSMConvKernel), convDimension),
		requiredTensorPointer("ssm_dt.bias", &layer.SSMTimeStep, uint64(spec.SSMTimeStepRank)),
		requiredTensorPointer("ssm_a", &layer.SSMA, tensor.SingletonExtent, uint64(spec.SSMTimeStepRank)),
		requiredTensorPointer("ssm_d", &layer.SSMD, tensor.SingletonExtent, uint64(spec.SSMTimeStepRank)),
		requiredTensorPointer("ssm_out.weight", &layer.SSMOutput, uint64(spec.SSMInnerSize), uint64(spec.EmbeddingLength)),
	}
	if includeNorm {
		requirements = append(requirements, requiredTensorPointer(
			"ssm_norm.weight", &layer.SSMNorm,
			uint64(spec.SSMInnerSize/spec.SSMGroupCount), uint64(spec.SSMGroupCount),
		))
	}
	bias := optionalTensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, convDimension)
	if mixer == recurrentMixerAttentionGroupedSelectiveScan {
		bias = optionalF32TensorPointer("ssm_conv1d.bias", &layer.SSMConv1DBias, convDimension)
	}
	bias.optional = mixer != recurrentMixerGroupedSelectiveScan
	requirements = append(requirements, bias)
	return requirements
}

func groupedStateSpaceConvolutionWidth(spec Spec) uint64 {
	return uint64(spec.SSMInnerSize) +
		tensor.PairedExtent*uint64(spec.SSMGroupCount)*uint64(spec.SSMStateSize)
}
