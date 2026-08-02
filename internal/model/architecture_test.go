package model

import "testing"

func TestArchitectureRegistryProfiles(t *testing.T) {
	tests := []struct {
		name       string
		family     ArchitectureFamily
		capability ArchitectureCapability
	}{
		{"llama", ArchitectureFamilyAttention, ArchitectureNormalRoPE},
		{"deepseek32", ArchitectureFamilyMoE, ArchitectureDSA | ArchitectureMLA},
		{"qwen35moe", ArchitectureFamilyHybrid, ArchitectureMoE | ArchitectureRecurrent},
		{"t5", ArchitectureFamilyEncoderDecoder, ArchitectureRoPEDisabled},
		{"llada", ArchitectureFamilyDiffusion, ArchitectureNonCausal | ArchitectureDiffusion},
		{"qwen3vl", ArchitectureFamilyAttention, ArchitectureMultimodal},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile, ok := LookupArchitecture(test.name)
			if !ok || profile.Name != test.name || profile.Family != test.family ||
				profile.Capabilities&test.capability != test.capability {
				t.Fatalf("profile = %#v", profile)
			}
		})
	}
	if _, ok := LookupArchitecture("unknown"); ok {
		t.Fatal("unknown architecture registered")
	}
	names := SupportedArchitectures()
	if len(names) != len(architectureRegistry) || len(names) < 100 {
		t.Fatalf("registered names = %d", len(names))
	}
	for index := 1; index < len(names); index++ {
		if names[index-1] >= names[index] {
			t.Fatalf("names are not sorted at %d", index)
		}
	}
}
