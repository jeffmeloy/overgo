package model

import "overgo/internal/gguf"

func loadTokenShiftRecurrentLayer(
	required weightRequirementLoader,
	tensors map[string]gguf.TensorInfo,
	prefix string,
	spec Spec,
	layer *LayerWeights,
	block uint32,
	mixer RecurrentMixerPolicy,
) error {
	var err error
	layer.Recurrent = true
	if mixer == recurrentMixerAffineWKV6 {
		embedding := uint64(spec.EmbeddingLength)
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, embedding),
			requiredTensorPointer("attn_norm_2.bias", &layer.AttentionNorm2Bias, embedding),
			requiredTensorPointer("time_mix_w1.weight", &layer.TimeMixW1, embedding, uint64(spec.TimeMixExtraDim)*5),
			requiredTensorPointer("time_mix_w2.weight", &layer.TimeMixW2, uint64(spec.TimeMixExtraDim), embedding, 5),
			requiredTensorPointer("time_mix_lerp_x.weight", &layer.TimeMixLerpX, embedding, 1, 1),
			requiredTensorPointer("time_mix_first.weight", &layer.TimeMixFirst, uint64(spec.WKVHeadSize), uint64(spec.HeadCount)),
			requiredTensorPointer("time_mix_decay.weight", &layer.TimeMixDecay, embedding),
			requiredTensorPointer("time_mix_decay_w1.weight", &layer.TimeMixDecayW1, embedding, uint64(spec.TimeDecayExtraDim)),
			requiredTensorPointer("time_mix_decay_w2.weight", &layer.TimeMixDecayW2, uint64(spec.TimeDecayExtraDim), embedding),
			requiredTensorPointer("time_mix_key.weight", &layer.TimeMixKey, embedding, embedding),
			requiredTensorPointer("time_mix_value.weight", &layer.TimeMixValue, embedding, embedding),
			requiredTensorPointer("time_mix_receptance.weight", &layer.TimeMixReceptance, embedding, embedding),
			requiredTensorPointer("time_mix_gate.weight", &layer.TimeMixGate, embedding, embedding),
			requiredTensorPointer("time_mix_ln.weight", &layer.TimeMixLN, embedding),
			requiredTensorPointer("time_mix_ln.bias", &layer.TimeMixLNBias, embedding),
			requiredTensorPointer("time_mix_output.weight", &layer.TimeMixOutput, embedding, embedding),
			requiredTensorPointer("channel_mix_lerp_k.weight", &layer.ChannelMixLerpK, embedding, 1, 1),
			requiredTensorPointer("channel_mix_lerp_r.weight", &layer.ChannelMixLerpR, embedding, 1, 1),
			requiredTensorPointer("channel_mix_key.weight", &layer.ChannelMixKey, embedding, uint64(spec.FeedForwardLength)),
			requiredTensorPointer("channel_mix_value.weight", &layer.ChannelMixValue, uint64(spec.FeedForwardLength), embedding),
			requiredTensorPointer("channel_mix_receptance.weight", &layer.ChannelMixReceptance, embedding, embedding),
		}); itemErr != nil {
			return itemErr
		}
		if item, ok := tensors[prefix+"time_mix_lerp_fused.weight"]; ok {
			validated, itemErr := required(item.Name, embedding, 1, 1, 5)
			if itemErr != nil {
				return itemErr
			}
			layer.TimeMixLerpFused = &validated
		} else {
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("time_mix_lerp_w.weight", &layer.TimeMixLerpW, embedding, 1, 1),
				requiredTensorPointer("time_mix_lerp_k.weight", &layer.TimeMixLerpK, embedding, 1, 1),
				requiredTensorPointer("time_mix_lerp_v.weight", &layer.TimeMixLerpV, embedding, 1, 1),
				requiredTensorPointer("time_mix_lerp_r.weight", &layer.TimeMixLerpR, embedding, 1, 1),
				requiredTensorPointer("time_mix_lerp_g.weight", &layer.TimeMixLerpG, embedding, 1, 1),
			}); itemErr != nil {
				return itemErr
			}
		}
		return nil
	}
	if mixer == recurrentMixerDynamicWKV7 {
		channelMix := spec.Profile().Normalization == NormalizationLayer
		embedding := uint64(spec.EmbeddingLength)
		valueRank := uint64(spec.ValueMixLoRARank)
		if block == 0 {
			valueRank = uint64(spec.ICLRLoRARank)
		}
		lerpCount := uint64(6)
		if !channelMix && spec.GateLoRARank == 0 {
			lerpCount = 5
		}
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("time_mix_w0.weight", &layer.TimeMixW0, embedding),
			requiredTensorPointer("time_mix_w1.weight", &layer.TimeMixW1, embedding, uint64(spec.DecayLoRARank)),
			requiredTensorPointer("time_mix_w2.weight", &layer.TimeMixW2, uint64(spec.DecayLoRARank), embedding),
			requiredTensorPointer("time_mix_a0.weight", &layer.TimeMixA0, embedding),
			requiredTensorPointer("time_mix_a1.weight", &layer.TimeMixA1, embedding, uint64(spec.ICLRLoRARank)),
			requiredTensorPointer("time_mix_a2.weight", &layer.TimeMixA2, uint64(spec.ICLRLoRARank), embedding),
			requiredTensorPointer("time_mix_v0.weight", &layer.TimeMixV0, embedding),
			requiredTensorPointer("time_mix_v1.weight", &layer.TimeMixV1, embedding, valueRank),
			requiredTensorPointer("time_mix_v2.weight", &layer.TimeMixV2, valueRank, embedding),
			requiredTensorPointer("time_mix_lerp_fused.weight", &layer.TimeMixLerpFused, embedding, 1, 1, lerpCount),
			requiredTensorPointer("time_mix_k_k.weight", &layer.TimeMixKK, embedding),
			requiredTensorPointer("time_mix_k_a.weight", &layer.TimeMixKA, embedding),
			requiredTensorPointer("time_mix_r_k.weight", &layer.TimeMixRK, embedding),
			requiredTensorPointer("time_mix_key.weight", &layer.TimeMixKey, embedding, embedding),
			requiredTensorPointer("time_mix_value.weight", &layer.TimeMixValue, embedding, embedding),
			requiredTensorPointer("time_mix_receptance.weight", &layer.TimeMixReceptance, embedding, embedding),
			requiredTensorPointer("time_mix_output.weight", &layer.TimeMixOutput, embedding, embedding),
		}); itemErr != nil {
			return itemErr
		}
		if spec.GateLoRARank > 0 {
			g1, itemErr := required(prefix+"time_mix_g1.weight", embedding, uint64(spec.GateLoRARank))
			if itemErr != nil {
				return itemErr
			}
			g2, itemErr := required(prefix+"time_mix_g2.weight", uint64(spec.GateLoRARank), embedding)
			if itemErr != nil {
				return itemErr
			}
			layer.TimeMixG1, layer.TimeMixG2 = &g1, &g2
		}
		if channelMix {
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensorPointer("attn_norm_2.weight", &layer.AttentionNorm2, embedding),
				requiredTensorPointer("attn_norm_2.bias", &layer.AttentionNorm2Bias, embedding),
				requiredTensorPointer("time_mix_ln.weight", &layer.TimeMixLN, embedding),
				requiredTensorPointer("time_mix_ln.bias", &layer.TimeMixLNBias, embedding),
				requiredTensorPointer("channel_mix_lerp_k.weight", &layer.ChannelMixLerpK, embedding, 1, 1),
				requiredTensorPointer("channel_mix_key.weight", &layer.ChannelMixKey, embedding, uint64(spec.FeedForwardLength)),
				requiredTensorPointer("channel_mix_value.weight", &layer.ChannelMixValue, uint64(spec.FeedForwardLength), embedding),
			}); itemErr != nil {
				return itemErr
			}
		} else {
			if item, ok := tensors[prefix+"time_mix_ln.weight"]; ok {
				norm, itemErr := required(item.Name, embedding)
				if itemErr != nil {
					return itemErr
				}
				bias, itemErr := required(prefix+"time_mix_ln.bias", embedding)
				if itemErr != nil {
					return itemErr
				}
				layer.TimeMixLN, layer.TimeMixLNBias = &norm, &bias
			}
			if layer.FeedForwardNorm, err = required(prefix+"ffn_norm.weight", embedding); err != nil {
				return err
			}
			if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
				requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, embedding, uint64(spec.FeedForwardLength)),
				requiredTensor("ffn_up.weight", &layer.FeedForwardUp, embedding, uint64(spec.FeedForwardLength)),
				requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), embedding),
			}); itemErr != nil {
				return itemErr
			}
		}
		return nil
	}
	if mixer == recurrentMixerDynamicWKV6 {
		embedding := uint64(spec.EmbeddingLength)
		keyValue := uint64(spec.HeadCountKV) * uint64(spec.WKVHeadSize)
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensorPointer("time_mix_w1.weight", &layer.TimeMixW1, embedding, uint64(spec.TimeMixExtraDim)*5),
			requiredTensorPointer("time_mix_w2.weight", &layer.TimeMixW2, uint64(spec.TimeMixExtraDim), embedding, 5),
			requiredTensorPointer("time_mix_lerp_x.weight", &layer.TimeMixLerpX, embedding, 1, 1),
			requiredTensorPointer("time_mix_lerp_fused.weight", &layer.TimeMixLerpFused, embedding, 1, 1, 5),
			requiredTensorPointer("time_mix_decay.weight", &layer.TimeMixDecay, embedding),
			requiredTensorPointer("time_mix_decay_w1.weight", &layer.TimeMixDecayW1, embedding, uint64(spec.TimeDecayExtraDim)),
			requiredTensorPointer("time_mix_decay_w2.weight", &layer.TimeMixDecayW2, uint64(spec.TimeDecayExtraDim), embedding),
			requiredTensorPointer("time_mix_key.weight", &layer.TimeMixKey, embedding, keyValue),
			requiredTensorPointer("time_mix_value.weight", &layer.TimeMixValue, embedding, keyValue),
			requiredTensorPointer("time_mix_receptance.weight", &layer.TimeMixReceptance, embedding, embedding),
			requiredTensorPointer("time_mix_gate.weight", &layer.TimeMixGate, embedding, embedding),
			requiredTensorPointer("time_mix_output.weight", &layer.TimeMixOutput, embedding, embedding),
			optionalTensorPointer("time_mix_key.bias", &layer.AttentionKBias, keyValue),
			optionalTensorPointer("time_mix_value.bias", &layer.AttentionVBias, keyValue),
			optionalTensorPointer("time_mix_receptance.bias", &layer.AttentionQBias, embedding),
		}); itemErr != nil {
			return itemErr
		}
		if layer.FeedForwardNorm, err = required(prefix+"ffn_norm.weight", embedding); err != nil {
			return err
		}
		if itemErr := loadTensorRequirements(required, tensors, prefix, []tensorRequirement{
			requiredTensor("ffn_gate.weight", &layer.FeedForwardGate, embedding, uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_up.weight", &layer.FeedForwardUp, embedding, uint64(spec.FeedForwardLength)),
			requiredTensor("ffn_down.weight", &layer.FeedForwardDown, uint64(spec.FeedForwardLength), embedding),
		}); itemErr != nil {
			return itemErr
		}
		return nil
	}
	return nil
}
