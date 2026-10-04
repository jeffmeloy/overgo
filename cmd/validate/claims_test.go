package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestGenerateValidatesEvidenceAndSortsOutput(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	source := writeEvidence(t, root, "internal/feature.go", "package feature\nfunc Feature() {}\n", "func Feature(", roleSource)
	proof := writeEvidence(t, root, "internal/feature_test.go", "package feature\nfunc TestFeature() {}\n", "func TestFeature(", roleArtifact)
	writeTestManifest(t, root, []claim{{
		ID: "feature", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test ./internal -run '^TestFeature$' -count=1 -v", Summary: "Feature works.",
		Evidence: []evidence{source, proof},
	}}, map[string]modelClaim{
		"zeta":       {Features: []string{"z"}, RealModelValidation: "pending-fixture"},
		"alpha":      {Features: []string{"a"}, ValidatedFixture: "alpha.gguf"},
		"multimodal": {Features: []string{"image"}, RealModelValidation: "pending-language-model-oracle", MultimodalValidatedFixture: "projector.gguf + image.png + golden.json"},
	})
	output, err := generate(root)
	if err != nil {
		t.Fatal(err)
	}
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

func TestGenerateRejectsStaleClaimEvidence(t *testing.T) {
	t.Parallel()
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
	t.Parallel()
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
	t.Parallel()
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

func TestRebindClaimsUsesCanonicalBytes(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	source := writeEvidence(t, root, "feature.go", "package feature\r\nfunc Feature() {}\r\n", "func Feature(", roleSource)
	extra := writeEvidence(t, root, "extra.go", "package feature\r\nfunc Extra() {}\r\n", "func Extra(", roleSource)
	proof := writeEvidence(t, root, "feature_test.go", "package feature\r\nfunc TestFeature() {}\r\n", "func TestFeature(", roleArtifact)
	source.Identity = "file:sha256:" + strings.Repeat("0", 64)
	extra.Identity = source.Identity
	proof.Identity = "evidence:sha256:" + strings.Repeat("0", 64)
	item := claim{
		ID: "rebind", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test . -run '^TestFeature$' -count=1 -v", Summary: "Rebind evidence.",
		Evidence: []evidence{source, extra, proof},
	}
	writeTestManifest(t, root, []claim{item}, testModels())
	var tested []string
	for _, proof := range item.Evidence {
		identity, err := evidenceIdentity(root, item.ID, proof)
		if err != nil {
			t.Fatal(err)
		}
		tested = append(tested, identity)
	}
	if err := rebindClaim(root, item, tested); err != nil {
		t.Fatal(err)
	}
	document, err := loadManifest(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range document.Claims[0].Evidence {
		if strings.Contains(item.Identity, strings.Repeat("0", 64)) {
			t.Fatalf("identity not rebound: %s", item.Identity)
		}
	}
	if _, err := generate(root); err != nil {
		t.Fatalf("rebound evidence rejected: %v", err)
	}
}

// TestReverifyRebindsOnlyPassingClaims holds a moved claim to its oracle: the
// claims whose verify test passes at the tree are rebound, and the claim whose
// test skips keeps its old binding, so the matrix stays refused. Two claims
// with the same verify share one run of it.
func TestReverifyRebindsOnlyPassingClaims(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testutil.WriteTextFile(t, root, "go.mod", "module fixture\n\ngo 1.21\n")
	passingSource := writeEvidence(t, root, "feature/feature.go", "package feature\n\nfunc Feature() int { return 1 }\n", "func Feature(", roleSource)
	sharingSource := writeEvidence(t, root, "feature/shared.go", "package feature\n\nfunc Shared() int { return 1 }\n", "func Shared(", roleSource)
	// The passing test records each run of it beside itself.
	passingTest := writeEvidence(t, root, "feature/feature_test.go", "package feature\n\nimport (\n\t\"os\"\n\t\"testing\"\n)\n\nfunc TestFeature(t *testing.T) {\n\truns, err := os.ReadFile(\"runs.txt\")\n\tif err == nil || os.IsNotExist(err) {\n\t\terr = os.WriteFile(\"runs.txt\", append(runs, \"run\\n\"...), 0o600)\n\t}\n\tif err != nil {\n\t\tt.Fatal(err)\n\t}\n}\n", "func TestFeature(", roleArtifact)
	skippedSource := writeEvidence(t, root, "feature/other.go", "package feature\n\nfunc Other() int { return 1 }\n", "func Other(", roleSource)
	skippedTest := writeEvidence(t, root, "feature/other_test.go", "package feature\n\nimport \"testing\"\n\nfunc TestOther(t *testing.T) { t.Skip(\"not applicable\") }\n", "func TestOther(", roleArtifact)
	writeTestManifest(t, root, []claim{{
		ID: "passing", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test ./feature -run '^TestFeature$' -count=1 -v", Summary: "Feature works.",
		Evidence: []evidence{passingSource, passingTest},
	}, {
		ID: "skipped", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test ./feature -run '^TestOther$' -count=1 -v", Summary: "Other works.",
		Evidence: []evidence{skippedSource, skippedTest},
	}, {
		ID: "sharing", Status: "implemented", EvidenceTier: tierContract,
		Verify: "go test ./feature -run '^TestFeature$' -count=1 -v", Summary: "Shared works.",
		Evidence: []evidence{sharingSource, passingTest},
	}}, testModels())
	testutil.WriteTextFile(t, root, "feature/feature.go", "package feature\n\nfunc Feature() int { return 2 }\n")
	testutil.WriteTextFile(t, root, "feature/other.go", "package feature\n\nfunc Other() int { return 2 }\n")
	testutil.WriteTextFile(t, root, "feature/shared.go", "package feature\n\nfunc Shared() int { return 2 }\n")
	moved, err := movedClaims(root)
	if err != nil || len(moved) != 3 {
		t.Fatalf("moved claims = %v, %v; want all three", moved, err)
	}
	if err := reverifyClaims(t.Context(), io.Discard, root); err == nil || !strings.Contains(err.Error(), "skipped") || strings.Contains(err.Error(), "passing") || strings.Contains(err.Error(), "sharing") {
		t.Fatalf("re-verification error = %v; want the skipped claim alone refused", err)
	}
	moved, err = movedClaims(root)
	if err != nil || len(moved) != 1 || moved[0].ID != "skipped" {
		t.Fatalf("moved claims after re-verification = %v, %v; want the skipped claim alone", moved, err)
	}
	if runs, err := os.ReadFile(filepath.Join(root, "feature", "runs.txt")); err != nil || strings.Count(string(runs), "run\n") != 1 {
		t.Fatalf("the shared verify ran %q (%v); want one run for both claims", runs, err)
	}
	if _, err := generate(root); err == nil || !strings.Contains(err.Error(), "is stale") {
		t.Fatalf("matrix over an unverified claim error = %v", err)
	}
}

// TestReverifyBindsTestedBytes holds a passing claim to the bytes its test
// ran over: a claim whose test edits its bound source during the run, or
// whose manifest entry is edited under it, is refused and keeps its old
// binding, never the new bytes; a claim whose inputs held still is rebound
// to the identities recorded before its run, and an unrelated manifest edit
// made during that run survives the rebinding.
func TestReverifyBindsTestedBytes(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testutil.WriteTextFile(t, root, "go.mod", "module fixture\n\ngo 1.21\n")
	// editing is a fixture test that, while it runs, replaces from with to
	// in path, relative to its package directory.
	editing := func(test, path, from, to string) string {
		return fmt.Sprintf("package feature\n\nimport (\n\t\"os\"\n\t\"strings\"\n\t\"testing\"\n)\n\nfunc %s(t *testing.T) {\n\tdata, err := os.ReadFile(%q)\n\tif err == nil {\n\t\terr = os.WriteFile(%[2]q, []byte(strings.Replace(string(data), %q, %q, 1)), 0o600)\n\t}\n\tif err != nil {\n\t\tt.Fatal(err)\n\t}\n}\n", test, path, from, to)
	}
	source := func(name, value string) string {
		return "package feature\n\nfunc " + name + "() int { return " + value + " }\n"
	}
	var claims []claim
	for _, fixture := range []struct{ name, test string }{
		{"Stable", editing("TestStable", "../compatibility.json", "Unrelated works.", "Unrelated still works.")},
		{"Edited", editing("TestEdited", "edited.go", "return 2", "return 3")},
		{"Authority", editing("TestAuthority", "../compatibility.json", "Authority works.", "Authority changed.")},
		{"Unrelated", "package feature\n\nimport \"testing\"\n\nfunc TestUnrelated(t *testing.T) {}\n"},
	} {
		path := "feature/" + strings.ToLower(fixture.name)
		claims = append(claims, claim{
			ID: strings.ToLower(fixture.name) + "-claim", Status: "implemented", EvidenceTier: tierContract,
			Verify: "go test ./feature -run '^Test" + fixture.name + "$' -count=1 -v", Summary: fixture.name + " works.",
			Evidence: []evidence{
				writeEvidence(t, root, path+".go", source(fixture.name, "1"), "func "+fixture.name+"(", roleSource),
				writeEvidence(t, root, path+"_test.go", fixture.test, "func Test"+fixture.name+"(", roleArtifact),
			},
		})
	}
	writeTestManifest(t, root, claims, testModels())
	for _, name := range []string{"Stable", "Edited", "Authority"} {
		testutil.WriteTextFile(t, root, "feature/"+strings.ToLower(name)+".go", source(name, "2"))
	}
	err := reverifyClaims(t.Context(), io.Discard, root)
	if err == nil || !strings.Contains(err.Error(), "edited-claim (feature/edited.go changed during its verify)") ||
		!strings.Contains(err.Error(), "authority-claim (its manifest entry changed during its verify)") || strings.Contains(err.Error(), "stable-claim") {
		t.Fatalf("re-verification error = %v; want the edited and authority claims alone refused", err)
	}
	moved, err := movedClaims(root)
	if err != nil || len(moved) != 2 || moved[0].ID != "edited-claim" || moved[1].ID != "authority-claim" {
		t.Fatalf("moved claims after re-verification = %v, %v; want the stable claim rebound to its tested bytes", moved, err)
	}
	document, err := loadManifest(root)
	if err != nil {
		t.Fatal(err)
	}
	if edited := document.Claims[1].Evidence[0]; edited.Identity != claims[1].Evidence[0].Identity {
		t.Fatalf("a source edited during its verify was rebound to %s", edited.Identity)
	}
	if authority := document.Claims[2]; authority.Summary != "Authority changed." || authority.Evidence[0].Identity != claims[2].Evidence[0].Identity {
		t.Fatalf("a claim whose entry changed during its verify = %+v", authority)
	}
	if unrelated := document.Claims[3]; unrelated.Summary != "Unrelated still works." {
		t.Fatalf("a manifest edit made during another claim's verify was lost: %+v", unrelated)
	}
}

func TestClaimsRequireLiveEvidenceTier(t *testing.T) {
	t.Parallel()
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
	t.Parallel()
	models := map[string]modelClaim{
		"alpha": {Features: []string{"a"}},
		"extra": {Features: []string{"x"}},
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
	return map[string]modelClaim{"model": {Features: []string{"feature"}}}
}
