package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenerateValidatesEvidenceAndSortsOutput(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "internal/feature.go", "package feature\nfunc Feature() {}\n")
	writeTestFile(t, root, "internal/feature_test.go", "package feature\nfunc TestFeature() {}\n")
	writeTestFile(t, root, manifestPath, `schema: 1
upstream:
  repository: ggml-org/llama.cpp
  commit: 42fc243060709331ff9b158a9ed2cbe37219ae83
host:
  os: windows
  arch: amd64
go:
  minimum: "1.26"
  cgo: false
claims:
  - id: feature
    status: implemented
    summary: Feature works.
    evidence:
      - path: internal/feature.go
        contains: "func Feature("
      - path: internal/feature_test.go
        contains: "func TestFeature("
models:
  zeta:
    status: experimental
    features: [z]
    real_model_validation: pending-fixture
  alpha:
    status: experimental
    features: [a]
    validated_fixture: alpha.gguf
`)
	output, err := generate(root)
	if err != nil {
		t.Fatal(err)
	}
	text := string(output)
	if strings.Index(text, "`alpha`") > strings.Index(text, "`zeta`") ||
		!strings.Contains(text, "validated: alpha.gguf") ||
		!strings.Contains(text, "[`func TestFeature(`](../internal/feature_test.go)") {
		t.Fatalf("generated matrix:\n%s", text)
	}
}

func TestGenerateRejectsStaleClaimEvidence(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "feature_test.go", "package feature\nfunc TestOther() {}\n")
	writeTestFile(t, root, manifestPath, `schema: 1
upstream: {repository: ggml-org/llama.cpp, commit: 42fc243060709331ff9b158a9ed2cbe37219ae83}
host: {os: windows, arch: amd64}
go: {minimum: "1.26", cgo: false}
claims:
  - id: stale
    status: implemented
    summary: Stale claim.
    evidence:
      - path: feature_test.go
        contains: "func TestMissing("
models:
  model:
    status: experimental
    features: [feature]
`)
	if _, err := generate(root); err == nil || !strings.Contains(err.Error(), "lacks") {
		t.Fatalf("stale evidence error = %v", err)
	}
}

func TestValidateModelCoverageRejectsMissingAndExtra(t *testing.T) {
	models := map[string]modelClaim{
		"alpha": {Status: "experimental", Features: []string{"a"}},
		"extra": {Status: "experimental", Features: []string{"x"}},
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

func writeTestFile(t *testing.T, root, path, data string) {
	t.Helper()
	absolute := filepath.Join(root, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(absolute), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(absolute, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}
