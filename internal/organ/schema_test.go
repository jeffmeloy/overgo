package organ

import "testing"

// TestComponentDecompositionContract pins the ported eight-axis organ
// classification against real GGUF and Hugging Face tensor names, and the
// contract validator against each cross-axis rule. The token rules mirror
// adaptive_new organ schema 2.0, so a name classifiable there classifies
// identically here.
func TestComponentDecompositionContract(t *testing.T) {
	classifications := []struct {
		name, dtype, family, modality, optimizerRole string
		want                                         Contract
	}{
		{"token_embd.weight", "f32", "llama", "", "", Contract{
			Modality: ModalityText, Role: RoleEmbedding, Space: SpaceTextHidden,
			DType: DTypeFP32, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"blk.0.attn_q.weight", "bf16", "qwen2", "text", "", Contract{
			Modality: ModalityText, Role: RoleQKV, Space: SpaceTextHidden,
			DType: DTypeBF16, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"blk.3.attn_output.weight", "f16", "llama", "text", "", Contract{
			Modality: ModalityText, Role: RoleAttentionOut, Space: SpaceTextHidden,
			DType: DTypeFP16, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"blk.1.ffn_gate.weight", "bf16", "llama", "text", "", Contract{
			Modality: ModalityText, Role: RoleMLPGate, Space: SpaceTextHidden,
			DType: DTypeBF16, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"blk.1.ffn_down.weight", "int8", "llama", "text", "", Contract{
			Modality: ModalityText, Role: RoleMLPDown, Space: SpaceTextHidden,
			DType: DTypeInt8, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"output_norm.weight", "f32", "llama", "text", "", Contract{
			Modality: ModalityText, Role: RoleNorm, Space: SpaceTextHidden,
			DType: DTypeFP32, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"model.layers.0.self_attn.qkv_proj.weight", "e4m3", "phi", "text", "", Contract{
			Modality: ModalityText, Role: RoleQKV, Space: SpaceTextHidden,
			DType: DTypeFP8, Layout: LayoutQKVPacked, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"vision_tower.patch_embedding.weight", "bf16", "gemma3", "", "", Contract{
			Modality: ModalityVision, Role: RoleEmbedding, Space: SpaceVisionHidden,
			DType: DTypeBF16, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"vae.decoder.conv_in.weight", "f32", "wan", "video", "", Contract{
			Modality: ModalityVideo, Role: RoleVAEDecoder, Space: SpaceVideoLatent,
			DType: DTypeFP32, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"lora_a.blk.0.attn_q", "f16", "llama", "text", "adapter", Contract{
			Modality: ModalityText, Role: RoleAdapter, Space: SpaceTextHidden,
			DType: DTypeFP16, Layout: LayoutUnknown, TrainingRole: TrainingRoleAdapter, Objective: ObjectiveUnknown,
		}},
		{"blk.2.ffn_up.weight", "bf16", "llama", "text", "momentum", Contract{
			Modality: ModalityText, Role: RoleMLPUp, Space: SpaceTextHidden,
			DType: DTypeBF16, Layout: LayoutUnknown, TrainingRole: TrainingRoleMomentum, Objective: ObjectiveUnknown,
		}},
		{"speech.projector.linear.weight", "f32", "pockettts", "", "", Contract{
			Modality: ModalityAudio, Role: RoleProjector, Space: SpaceAudioLatent,
			DType: DTypeFP32, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
		{"lm_head.weight", "bf16", "qwen2", "text", "", Contract{
			Modality: ModalityText, Role: RoleHead, Space: SpaceLogits,
			DType: DTypeBF16, Layout: LayoutUnknown, TrainingRole: TrainingRoleUnknown, Objective: ObjectiveUnknown,
		}},
	}
	for _, c := range classifications {
		got := Classify(c.name, c.dtype, c.family, c.modality, c.optimizerRole)
		if got != c.want {
			t.Errorf("Classify(%q) = %+v, want %+v", c.name, got, c.want)
		}
	}

	valid := Classify("blk.0.attn_q.weight", "bf16", "llama", "text", "")
	if issues := Validate(valid); len(issues) != 0 {
		t.Fatalf("valid contract reported issues: %v", issues)
	}
	violations := []struct {
		code     string
		contract Contract
	}{
		{"unknown-enum-value", Contract{Modality: "hologram"}},
		{"audio-latent-modality", Contract{Modality: ModalityText, Space: SpaceAudioLatent}},
		{"qkv-layout-role", Contract{Role: RoleMLPGate, Layout: LayoutQKVPacked}},
		{"logits-role", Contract{Role: RoleNorm, Space: SpaceLogits}},
		{"optimizer-state-role", Contract{Role: RoleQKV, TrainingRole: TrainingRoleOptimizerState}},
	}
	for _, violation := range violations {
		issues := Validate(violation.contract)
		found := false
		for _, issue := range issues {
			if issue.Code == violation.code {
				found = true
			}
		}
		if !found {
			t.Errorf("%s: not reported in %v", violation.code, issues)
		}
	}
}
