package main

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestGenerateValidatesEvidenceAndSortsOutput(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	source := writeEvidence(t, root, "internal/feature.go", "package feature\nfunc Feature() {}\n", "func Feature(", roleSource)
	proof := writeEvidence(t, root, "internal/feature_test.go", "package feature\nfunc TestFeature() {}\n", "func TestFeature(", roleArtifact)
	writeTestManifest(t, root, []claim{{
		ID: "feature", EvidenceTier: tierContract,
		Summary:  "Feature works.",
		Evidence: []evidence{source, proof},
	}}, map[string]modelClaim{
		"zeta":       {RealModelValidation: "pending-fixture"},
		"alpha":      {ValidatedFixture: "alpha.gguf"},
		"multimodal": {RealModelValidation: "pending-language-model-oracle", MultimodalValidatedFixture: "projector.gguf + image.png + golden.json"},
	})
	output := testcheck.Must(generate(root))(t)
	text := string(output)
	if strings.Contains(text, "experimental") || strings.Contains(text, "| Status |") {
		t.Fatalf("prototype rows must carry no status vocabulary:\n%s", text)
	}
	if strings.Index(text, "`alpha`") > strings.Index(text, "`zeta`") ||
		!strings.Contains(text, "validated: alpha.gguf") ||
		!strings.Contains(text, "multimodal: projector.gguf + image.png + golden.json; pending-language-model-oracle") ||
		!strings.Contains(text, "go test ./internal -run '^TestFeature$' -count=1 -v") {
		t.Fatalf("generated matrix:\n%s", text)
	}
}

// TestGenerateRejectsStaleClaimEvidence refuses a claim whose evidence no longer holds the text it names: the claim
// stands on a source or a test the checkout no longer has.
func TestGenerateRejectsStaleClaimEvidence(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	source := writeEvidence(t, root, "feature.go", "package feature\nfunc Feature() {}\n", "func Feature(", roleSource)
	proof := writeEvidence(t, root, "feature_test.go", "package feature\nfunc TestFeature() {}\n", "func TestFeature(", roleArtifact)
	testutil.WriteTextFile(t, root, "feature.go", "package feature\nfunc Renamed() {}\n")
	writeTestManifest(t, root, []claim{{
		ID: "stale", EvidenceTier: tierContract,
		Summary:  "Stale claim.",
		Evidence: []evidence{source, proof},
	}}, testModels())
	if _, err := generate(root); err == nil || !strings.Contains(err.Error(), `lacks "func Feature("`) {
		t.Fatalf("stale evidence error = %v", err)
	}
}

func TestClaimsRequireLiveEvidenceTier(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	source := writeEvidence(t, root, "feature.go", "package feature\nfunc Feature() {}\n", "func Feature(", roleSource)
	proof := writeEvidence(t, root, "feature_test.go", "package feature\nfunc TestFeature() {}\n", "func TestFeature(", roleArtifact)
	valid := claim{
		ID: "feature", EvidenceTier: tierContract,
		Summary:  "Feature works.",
		Evidence: []evidence{source, proof},
	}
	writeTestManifest(t, root, []claim{valid}, testModels())
	if _, err := generate(root); err != nil {
		t.Fatalf("live claim rejected: %v", err)
	}

	for name, mutate := range map[string]func(*claim){
		"tier":     func(item *claim) { item.EvidenceTier = "" },
		"source":   func(item *claim) { item.Evidence = item.Evidence[1:] },
		"artifact": func(item *claim) { item.Evidence = item.Evidence[:1] },
	} {
		t.Run(name, func(t *testing.T) {
			item := valid
			item.Evidence = append([]evidence(nil), valid.Evidence...)
			mutate(&item)
			writeTestManifest(t, root, []claim{item}, testModels())
			if _, err := generate(root); err == nil {
				t.Fatal("incomplete evidence accepted")
			}
		})
	}
}

func TestValidateModelCoverageRejectsMissingAndExtra(t *testing.T) {
	t.Parallel()
	models := map[string]modelClaim{
		"alpha": {},
		"extra": {},
	}
	err := validateModelCoverage(models, []string{"alpha", "beta"})
	if err == nil || !strings.Contains(err.Error(), "missing=[beta]") ||
		!strings.Contains(err.Error(), "extra=[extra]") {
		t.Fatalf("coverage error = %v", err)
	}
	if err := validateModelCoverage(models, []string{"alpha", "extra"}); err != nil {
		t.Fatalf("complete coverage error = %v", err)
	}
}

func writeEvidence(t *testing.T, root, path, data, contains string, role evidenceRole) evidence {
	t.Helper()
	testutil.WriteTextFile(t, root, path, data)
	return evidence{Path: path, Contains: contains, Role: role}
}

func writeTestManifest(t *testing.T, root string, claims []claim, models map[string]modelClaim) {
	t.Helper()
	document := manifest{
		Schema:   manifestSchema,
		Upstream: upstream{Repository: "ggml-org/llama.cpp", Commit: "42fc243060709331ff9b158a9ed2cbe37219ae83"},
		Host:     host{OS: "windows", Arch: "amd64"},
		Go:       goRuntime{Minimum: "1.26"},
		Claims:   claims,
		Models:   models,
	}
	data := testcheck.Must(json.MarshalIndent(document, "", "  "))(t)
	testutil.WriteTextFile(t, root, manifestPath, string(data))
}

func testModels() map[string]modelClaim {
	return map[string]modelClaim{"model": {}}
}
