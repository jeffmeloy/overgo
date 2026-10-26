package model

import (
	"slices"
	"testing"
)

// TestPromptTensorCoresAdmittedPerArchitecture holds the tensor-core prompt attention to the architectures whose
// held-out quality held with it: qwen4exp, Flash-Next's, alone; every other reads its prompts on the F32 paths.
func TestPromptTensorCoresAdmittedPerArchitecture(t *testing.T) {
	var admitted []string
	for name, profile := range mustLoadArchitectureRegistry() {
		if profile.Runtime.PromptTensorCores {
			admitted = append(admitted, name)
		}
	}
	slices.Sort(admitted)
	if !slices.Equal(admitted, []string{"qwen4exp"}) {
		t.Fatalf("the catalog admits the tensor-core prompt attention for %v, want qwen4exp alone", admitted)
	}
}
