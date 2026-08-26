package densecausal

import "testing"

func TestResolvedModelFactsRejectMissingConfig(t *testing.T) {
	for _, config := range []artifactConfig{
		{ModelType: "fixture"},
		{ModelType: "wrapper", LanguageConfig: &artifactConfig{}},
	} {
		if _, _, err := config.resolvedDecoder(); err == nil {
			t.Fatalf("missing decoder facts accepted: %+v", config)
		}
	}
}
