package model

import (
	"encoding/json"
	"testing"
)

const (
	fixtureArchitectureName = "fixture"
	unsupportedPolicyValue  = ^uint8(0)
)

func TestArchitectureCatalogStrictParsing(t *testing.T) {
	tests := map[string]string{
		"empty":         `[]`,
		"unknown field": `[{"Name":"fixture","Unknown":1}]`,
		"invalid name":  `[{"Name":"Fixture"}]`,
		"unordered":     `[{"Name":"second"},{"Name":"first"}]`,
		"duplicate":     `[{"Name":"same"},{"Name":"same"}]`,
		"trailing":      `[{"Name":"fixture"}] {}`,
	}
	for name, content := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := parseArchitectureRegistry([]byte(content)); err == nil {
				t.Fatal("invalid catalog accepted")
			}
		})
	}
}

func TestArchitectureCatalogRejectsInvalidPolicy(t *testing.T) {
	tests := map[string]func(*ArchitectureProfile){
		"family": func(profile *ArchitectureProfile) {
			profile.Family = ArchitectureFamily(unsupportedPolicyValue)
		},
		"nested validation": func(profile *ArchitectureProfile) {
			profile.Validation.Hybrid = HybridValidationPolicy(unsupportedPolicyValue)
		},
		"capability bits": func(profile *ArchitectureProfile) {
			profile.Capabilities = allArchitectureCapabilities + 1
		},
		"expert supplement bits": func(profile *ArchitectureProfile) {
			profile.Experts.SupplementalCatalog = allExpertSupplements + 1
		},
		"metadata read bits": func(profile *ArchitectureProfile) {
			profile.MetadataRead = allMetadataReadPolicies + 1
		},
		"metadata default relationship": func(profile *ArchitectureProfile) {
			profile.MetadataDefaults.RopeDimension = RopeDimensionDefaultPolicy(unsupportedPolicyValue)
		},
		"metadata default scalar": func(profile *ArchitectureProfile) {
			profile.MetadataDefaults.AttentionSoftcap = -1
		},
		"metadata read relationship": func(profile *ArchitectureProfile) {
			profile.MetadataRead = MetadataReadQwen3VLDeepstack
		},
		"Qwen GDN graph": func(profile *ArchitectureProfile) {
			profile.AttentionGraph.QwenGDN = qwenGDNStandard
		},
		"multi-axis rotary": func(profile *ArchitectureProfile) {
			profile.Rotary.MultiAxis = multiAxisRotaryAlways
		},
		"fused QKV requirement": func(profile *ArchitectureProfile) {
			profile.Capabilities = ArchitectureRequiresFusedQKV
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			profile := ArchitectureProfile{Name: fixtureArchitectureName}
			mutate(&profile)
			content, err := json.Marshal([]ArchitectureProfile{profile})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := parseArchitectureRegistry(content); err == nil {
				t.Fatal("invalid profile accepted")
			}
		})
	}
}

func TestEmbeddedArchitectureCatalog(t *testing.T) {
	registry, err := parseArchitectureRegistry(architectureProfileCatalog)
	if err != nil {
		t.Fatal(err)
	}
	if len(registry) != len(architectureRegistry) {
		t.Fatalf("catalog profiles = %d, registry profiles = %d", len(registry), len(architectureRegistry))
	}
	for name, profile := range registry {
		if architectureRegistry[name] != profile {
			t.Fatalf("profile %q drifted during bootstrap", name)
		}
	}
}
