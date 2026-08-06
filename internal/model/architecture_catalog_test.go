package model

import "testing"

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
