package main

import (
	"encoding/json"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestVerifyingCommitRequiresCleanWorktree(t *testing.T) {
	root := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		command := exec.Command("git", append([]string{"-C", root}, args...)...)
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	git("init")
	git("config", "user.name", "Overgo Test")
	git("config", "user.email", "overgo@example.invalid")
	testutil.WriteTextFile(t, root, "source.go", "package source\n")
	git("add", "source.go")
	git("commit", "-m", "fixture")
	want := git("rev-parse", "HEAD")
	if got, err := verifyingCommit(root); err != nil || got != want {
		t.Fatalf("clean verifying commit = %q, %v; want %q", got, err, want)
	}

	testutil.WriteTextFile(t, root, "source.go", "package changed\n")
	if _, err := verifyingCommit(root); err == nil || !strings.Contains(err.Error(), "dirty") {
		t.Fatalf("tracked change error = %v", err)
	}
	git("restore", "source.go")
	testutil.WriteTextFile(t, root, "untracked.go", "package source\n")
	if _, err := verifyingCommit(root); err == nil || !strings.Contains(err.Error(), "dirty") {
		t.Fatalf("untracked source error = %v", err)
	}
}

func TestGenerateValidatesEvidenceAndSortsOutput(t *testing.T) {
	root := t.TempDir()
	source := writeEvidence(t, root, "internal/feature.go", "package feature\nfunc Feature() {}\n", "func Feature(", roleSource)
	proof := writeEvidence(t, root, "internal/feature_test.go", "package feature\nfunc TestFeature() {}\n", "func TestFeature(", roleArtifact)
	writeTestManifest(t, root, []claim{{
		ID: "feature", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test ./internal -run '^TestFeature$' -count=1 -v", Summary: "Feature works.",
		Evidence: []evidence{source, proof},
	}}, map[string]modelClaim{
		"zeta":       {Status: "experimental", Features: []string{"z"}, RealModelValidation: "pending-fixture"},
		"alpha":      {Status: "experimental", Features: []string{"a"}, ValidatedFixture: "alpha.gguf"},
		"multimodal": {Status: "experimental", Features: []string{"image"}, RealModelValidation: "pending-language-model-oracle", MultimodalValidatedFixture: "projector.gguf + image.png + golden.json"},
	})
	output, err := generate(root)
	if err != nil {
		t.Fatal(err)
	}
	text := string(output)
	if strings.Index(text, "`alpha`") > strings.Index(text, "`zeta`") ||
		!strings.Contains(text, "validated: alpha.gguf") ||
		!strings.Contains(text, "multimodal: projector.gguf + image.png + golden.json; pending-language-model-oracle") ||
		!strings.Contains(text, "go test ./internal -run '^TestFeature$' -count=1 -v") {
		t.Fatalf("generated matrix:\n%s", text)
	}
}

func TestGenerateRejectsStaleClaimEvidence(t *testing.T) {
	root := t.TempDir()
	source := writeEvidence(t, root, "feature.go", "package feature\nfunc Feature() {}\n", "func Feature(", roleSource)
	proof := writeEvidence(t, root, "feature_test.go", "package feature\nfunc TestFeature() {}\n", "func TestFeature(", roleArtifact)
	testutil.WriteTextFile(t, root, "feature_test.go", "package feature\nfunc TestFeature() { panic(\"changed\") }\n")
	writeTestManifest(t, root, []claim{{
		ID: "stale", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test . -run '^TestFeature$' -count=1 -v", Summary: "Stale claim.",
		Evidence: []evidence{source, proof},
	}}, testModels())
	if _, err := generate(root); err == nil || !strings.Contains(err.Error(), "is stale") {
		t.Fatalf("stale evidence error = %v", err)
	}
}

func TestSymbolEvidenceIgnoresUnrelatedFileChanges(t *testing.T) {
	root := t.TempDir()
	data := "package feature\nfunc Feature() int { return 1 }\nfunc Unrelated() int { return 2 }\n"
	proof := writeSymbolEvidence(t, root, "feature.go", data, "func Feature(", "Feature", roleSource)
	testProof := writeSymbolEvidence(t, root, "feature_test.go", "package feature\nfunc TestFeature() {}\n", "func TestFeature(", "TestFeature", roleArtifact)
	writeTestManifest(t, root, []claim{{
		ID: "symbol", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test . -run '^TestFeature$' -count=1 -v", Summary: "Symbol evidence.",
		Evidence: []evidence{proof, testProof},
	}}, testModels())
	testutil.WriteTextFile(t, root, "feature.go", "package feature\nfunc Feature() int { return 1 }\nfunc Unrelated() int { return 3 }\n")
	if _, err := generate(root); err != nil {
		t.Fatalf("unrelated declaration invalidated symbol evidence: %v", err)
	}
	testutil.WriteTextFile(t, root, "feature.go", "package feature\nfunc Feature() int { return 4 }\nfunc Unrelated() int { return 3 }\n")
	if _, err := generate(root); err == nil || !strings.Contains(err.Error(), "is stale") {
		t.Fatalf("changed symbol evidence error = %v", err)
	}
}

func TestGenerateNormalizesEvidenceLineEndings(t *testing.T) {
	root := t.TempDir()
	sourceText := "package feature\nfunc Feature() {}\n"
	proofText := "package feature\nfunc TestFeature() {}\n"
	source := writeEvidence(t, root, "feature.go", sourceText, "func Feature(", roleSource)
	proof := writeEvidence(t, root, "feature_test.go", proofText, "func TestFeature(", roleArtifact)
	testutil.WriteTextFile(t, root, "feature.go", strings.ReplaceAll(sourceText, "\n", "\r\n"))
	testutil.WriteTextFile(t, root, "feature_test.go", strings.ReplaceAll(proofText, "\n", "\r\n"))
	writeTestManifest(t, root, []claim{{
		ID: "portable", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test . -run '^TestFeature$' -count=1 -v", Summary: "Portable evidence.",
		Evidence: []evidence{source, proof},
	}}, testModels())
	if _, err := generate(root); err != nil {
		t.Fatalf("line-ending-equivalent evidence rejected: %v", err)
	}
}

func TestRefreshEvidenceIdentitiesUsesCanonicalBytes(t *testing.T) {
	root := t.TempDir()
	source := writeEvidence(t, root, "feature.go", "package feature\r\nfunc Feature() {}\r\n", "func Feature(", roleSource)
	extra := writeEvidence(t, root, "extra.go", "package feature\r\nfunc Extra() {}\r\n", "func Extra(", roleSource)
	proof := writeEvidence(t, root, "feature_test.go", "package feature\r\nfunc TestFeature() {}\r\n", "func TestFeature(", roleArtifact)
	source.Identity = "file:sha256:" + strings.Repeat("0", 64)
	extra.Identity = source.Identity
	proof.Identity = "evidence:sha256:" + strings.Repeat("0", 64)
	writeTestManifest(t, root, []claim{{
		ID: "refresh", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test . -run '^TestFeature$' -count=1 -v", Summary: "Refresh evidence.",
		Evidence: []evidence{source, extra, proof},
	}}, testModels())
	if err := refreshEvidenceIdentities(root); err != nil {
		t.Fatal(err)
	}
	document, err := loadManifest(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range document.Claims[0].Evidence {
		if strings.Contains(item.Identity, strings.Repeat("0", 64)) {
			t.Fatalf("identity not refreshed: %s", item.Identity)
		}
	}
	if _, err := generate(root); err != nil {
		t.Fatalf("refreshed evidence rejected: %v", err)
	}
}

func TestClaimsRequireLiveEvidenceTier(t *testing.T) {
	root := t.TempDir()
	source := writeEvidence(t, root, "feature.go", "package feature\nfunc Feature() {}\n", "func Feature(", roleSource)
	proof := writeEvidence(t, root, "feature_test.go", "package feature\nfunc TestFeature() {}\n", "func TestFeature(", roleArtifact)
	valid := claim{
		ID: "feature", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test . -run '^TestFeature$' -count=1 -v", Summary: "Feature works.",
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
		"command":  func(item *claim) { item.Verify = "go test . -run '^TestOther$' -count=1 -v" },
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

func writeEvidence(t *testing.T, root, path, data, contains string, role evidenceRole) evidence {
	t.Helper()
	testutil.WriteTextFile(t, root, path, data)
	kind := artifact.KindFile
	if role == roleArtifact {
		kind = artifact.KindEvidence
	}
	id, err := artifact.IdentifyBytes(kind, []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	return evidence{Path: path, Contains: contains, Role: role, Identity: id.String()}
}

func writeSymbolEvidence(t *testing.T, root, path, data, contains, symbol string, role evidenceRole) evidence {
	t.Helper()
	testutil.WriteTextFile(t, root, path, data)
	proof := evidence{Path: path, Contains: contains, Symbol: symbol, Role: role}
	payload, err := evidencePayload(proof, []byte(data))
	if err != nil {
		t.Fatal(err)
	}
	kind := artifact.KindFile
	if role == roleArtifact {
		kind = artifact.KindEvidence
	}
	id, err := artifact.IdentifyBytes(kind, payload)
	if err != nil {
		t.Fatal(err)
	}
	proof.Identity = id.String()
	return proof
}

func writeTestManifest(t *testing.T, root string, claims []claim, models map[string]modelClaim) {
	t.Helper()
	document := manifest{
		Schema:   2,
		Upstream: upstream{Repository: "ggml-org/llama.cpp", Commit: "42fc243060709331ff9b158a9ed2cbe37219ae83"},
		Host:     host{OS: "windows", Arch: "amd64"},
		Go:       goRuntime{Minimum: "1.26"},
		Claims:   claims,
		Models:   models,
	}
	data, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	testutil.WriteTextFile(t, root, manifestPath, string(data))
}

func testModels() map[string]modelClaim {
	return map[string]modelClaim{"model": {Status: "experimental", Features: []string{"feature"}}}
}
