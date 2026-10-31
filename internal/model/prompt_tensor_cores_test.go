package model

import (
	"slices"
	"testing"
)

// TestPromptTensorCoresAdmittedPerArchitecture holds the tensor-core prompt attention to the architectures whose
// held-out quality held with it: qwen4exp (Flash-Next's MMLU-Pro), and with its products split into FP16 roundings and
// their remainders, qwen2, llama, gemma4 and deepseek2-ocr (Qwen2.5-0.5B, MiniCPM5-1B, gemma-4-E4B-it and
// Unlimited-OCR, 13 cases lost and 20 gained against the F32 paths over their BBH, IFEval and MMLU-Pro on 2026-10-31);
// every other reads its prompts on the F32 paths. The generic attention plan, which qwen2 and llama build their layers
// through, carries the admission as the dense blocks do.
func TestPromptTensorCoresAdmittedPerArchitecture(t *testing.T) {
	registry := mustLoadArchitectureRegistry()
	var admitted []string
	for name, profile := range registry {
		if profile.Runtime.PromptTensorCores {
			admitted = append(admitted, name)
		}
		if plan := (Spec{}).attentionGraphPlan(profile, 0); plan.TensorCores != profile.Runtime.PromptTensorCores {
			t.Errorf("%s's attention plan reads on tensor cores %v, its profile admits %v", name, plan.TensorCores, profile.Runtime.PromptTensorCores)
		}
	}
	slices.Sort(admitted)
	if want := []string{"deepseek2-ocr", "gemma4", "llama", "qwen2", "qwen4exp"}; !slices.Equal(admitted, want) {
		t.Fatalf("the catalog admits the tensor-core prompt attention for %v, want %v", admitted, want)
	}
}
