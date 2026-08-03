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
		{"gemma3n", ArchitectureFamilyAttention, ArchitectureSharedKV | ArchitectureAltUp | ArchitecturePerLayerEmbeddings},
		{"talkie", ArchitectureFamilyAttention, ArchitectureEmbeddingSkip},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile, ok := LookupArchitecture(test.name)
			if !ok || profile.Name != test.name || profile.Family != test.family ||
				profile.GraphFamily != test.family || profile.CatalogFamily != test.family ||
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

func TestArchitectureProfileDraftBlockPolicy(t *testing.T) {
	for architecture, want := range map[string]bool{
		"step35": true, "hy_v3": true, "glm4": true,
		"qwen35": false, "cohere2moe": false, "llama": false,
	} {
		profile, ok := LookupArchitecture(architecture)
		if !ok || profile.AppendsDraftBlocks() != want {
			t.Fatalf("%s appends draft blocks = %v, want %v", architecture, profile.AppendsDraftBlocks(), want)
		}
	}
}

func TestArchitectureProfileForwardPolicy(t *testing.T) {
	for architecture, want := range map[string]ForwardPolicy{
		"llama":            ForwardCached,
		"bert":             ForwardNonCausal,
		"dream":            ForwardNonCausal,
		"dflash":           ForwardDFlash,
		"eagle3":           ForwardEagle3,
		"gemma4-assistant": ForwardGemma4Assistant,
		"wavtokenizer-dec": ForwardWavTokenizer,
		"t5encoder":        ForwardT5Encoder,
		"t5":               ForwardT5,
	} {
		profile, ok := LookupArchitecture(architecture)
		if !ok || profile.Forward != want {
			t.Fatalf("%s forward policy = %v, want %v", architecture, profile.Forward, want)
		}
	}
}

func TestArchitectureProfileOutputNormPolicy(t *testing.T) {
	for architecture, want := range map[string]struct {
		policy OutputNormPolicy
		tensor string
	}{
		"llama":          {OutputNormModel, "output_norm.weight"},
		"bert":           {OutputNormAbsent, ""},
		"nomic-bert-moe": {OutputNormAbsent, ""},
		"neo-bert":       {OutputNormEncoder, "enc.output_norm.weight"},
		"t5encoder":      {OutputNormEncoder, "enc.output_norm.weight"},
		"t5":             {OutputNormDecoder, "dec.output_norm.weight"},
		"lfm2":           {OutputNormTokenEmbedding, "token_embd_norm.weight"},
	} {
		profile, ok := LookupArchitecture(architecture)
		if !ok || profile.OutputNorm != want.policy || profile.OutputNormTensor() != want.tensor {
			t.Fatalf("%s output norm = %v/%q, want %v/%q",
				architecture, profile.OutputNorm, profile.OutputNormTensor(), want.policy, want.tensor)
		}
		if want.policy == OutputNormAbsent && !profile.Has(ArchitectureBERTNormLayout) {
			t.Fatalf("%s has no BERT normalization layout", architecture)
		}
	}
}

func TestArchitectureProfileFusedQKVPolicy(t *testing.T) {
	for architecture, capabilities := range map[string]ArchitectureCapability{
		"bloom": ArchitectureFusedQKV | ArchitectureRequiresFusedQKV | ArchitectureRequiresFusedQKVBias,
		"mpt":   ArchitectureFusedQKV | ArchitectureRequiresFusedQKV,
		"phi3":  ArchitectureFusedQKV | ArchitectureRejectsOrphanFusedQKVBias,
		"llama": 0,
	} {
		profile, ok := LookupArchitecture(architecture)
		mask := ArchitectureFusedQKV | ArchitectureRequiresFusedQKV |
			ArchitectureRequiresFusedQKVBias | ArchitectureRejectsOrphanFusedQKVBias
		if !ok || profile.Capabilities&mask != capabilities {
			t.Fatalf("%s fused QKV capabilities = %064b", architecture, profile.Capabilities)
		}
	}
}
