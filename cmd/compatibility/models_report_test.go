package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
	"overgo/internal/testutil"
)

// TestCompatibilityReportsAreRenderedOnDemand pins that the model report
// left the tree: docs/model_compatibility.json is not tracked, and the report
// renders from its inputs -- the typed verification specifications joined
// against the store -- into the typed report a reader decodes.
func TestCompatibilityReportsAreRenderedOnDemand(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	tracked, err := gitauthority.Query(t.Context(), root, "ls-files", "--", "docs/model_compatibility.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(tracked) != 0 {
		t.Fatal("docs/model_compatibility.json is committed; the model report renders with go run ./cmd/compatibility -models")
	}
	if report := renderedModelsReport(t); len(report.Prototypes) == 0 {
		t.Fatal("the rendered model report is empty")
	}
}

// modelsReportRender renders the model report once per test process, from
// the checkout's specifications and the retained store, decoded strictly as
// a reader would. The report is a function of the checkout and the store,
// which a test process does not change, and a render parses every present
// model file's header.
var modelsReportRender = sync.OnceValues(func() (modelsReport, error) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		return modelsReport{}, err
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		return modelsReport{}, err
	}
	store, err := overgodb.OpenReadOnly(retainedReferenceStore(roots.Store))
	if err != nil {
		return modelsReport{}, err
	}
	defer store.Close()
	data, err := generateModelsReport(root, store)
	if err != nil {
		return modelsReport{}, err
	}
	var report modelsReport
	return report, strictjson.DecodeBytes(data, &report)
})

// renderedModelsReport is the report a reader renders now.
func renderedModelsReport(t *testing.T) modelsReport {
	t.Helper()
	report, err := modelsReportRender()
	if err != nil {
		t.Fatal(err)
	}
	return report
}

// TestModelsReportNestsSpecificModels pins the report contract:
// prototypes are identified without status, each specific model nests
// inside the prototype its declared architecture attributes it to, a
// specification's weaker duplicate claim reduces away, and a model
// with no resolvable architecture lands in the explicit unattributed
// bucket instead of a guessed one.
func TestModelsReportNestsSpecificModels(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testutil.WriteTextFile(t, root, "compatibility.json", `{"models": {"zeta": {"features": ["z"]}}}`)
	weights := filepath.Join(root, "weights", "model.safetensors")
	if err := os.MkdirAll(filepath.Dir(weights), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(weights, []byte("st"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "weights", "config.json"), []byte(`{"model_type":"zeta"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	commit := strings.Repeat("ab", 20)
	model := testutil.ArtifactID(t, artifact.KindModel, "alpha")
	proof := testutil.ArtifactID(t, artifact.KindEvidence, "proof")
	orphan := testutil.ArtifactID(t, artifact.KindModel, "orphan")
	writeSpecification(t, root, "alpha.json", map[string]any{
		"model": model.String(), "name": "Alpha-1B",
		"model_file": filepath.ToSlash(weights),
		"claims": []map[string]any{
			{"capability": "inference", "tier": "real-artifact-smoke", "commit": commit, "evidence": []string{proof.String()}},
			{"capability": "inference", "tier": "exact-golden", "commit": commit, "evidence": []string{proof.String()}},
		},
	})
	writeSpecification(t, root, "orphan.json", map[string]any{
		"model": orphan.String(), "name": "Orphan-1B",
		"claims": []map[string]any{
			{"capability": "inference", "tier": "real-artifact-smoke", "commit": commit, "evidence": []string{proof.String()}},
		},
	})
	writeSpecification(t, root, "external.json", map[string]any{
		"model": map[string]any{"tool": "external"}, "result": "ignored",
	})
	if err := os.WriteFile(filepath.Join(root, "docs", "verification", "external-array.json"), []byte(`[{"result":"external oracle"}]`), 0o600); err != nil {
		t.Fatal(err)
	}
	malformed := filepath.Join(root, "docs", "verification", "malformed.json")
	if err := os.WriteFile(malformed, []byte(`[{`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadVerificationInventory(root); err == nil {
		t.Fatal("malformed external oracle accepted")
	}
	if err := os.Remove(malformed); err != nil {
		t.Fatal(err)
	}
	data, err := generateModelsReport(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	var report modelsReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatal(err)
	}
	alpha, ok := report.Prototypes["zeta"].Models["Alpha-1B"]
	if !ok || alpha.Model != model.String() || len(alpha.Training) != 0 {
		t.Fatalf("zeta models = %+v", report.Prototypes["zeta"].Models)
	}
	if len(alpha.Inference) != 1 || alpha.Inference[0].Tier != "exact-golden" {
		t.Fatalf("strongest inference claim = %+v", alpha.Inference)
	}
	if _, ok := report.Prototypes["unattributed"].Models["Orphan-1B"]; !ok {
		t.Fatalf("unattributed bucket = %+v", report.Prototypes["unattributed"])
	}
	if strings.Contains(string(data), "experimental") {
		t.Fatal("report must carry no status vocabulary")
	}
	// Distinct artifacts can share a display name, as raw and registered Qwen4 do.
	sibling := testutil.ArtifactID(t, artifact.KindModel, "alpha-sibling")
	writeSpecification(t, root, "alpha-sibling.json", map[string]any{
		"model": sibling.String(), "name": "Alpha-1B", "model_file": filepath.ToSlash(weights),
		"claims": []map[string]any{
			{"capability": "inference", "tier": "real-artifact-smoke", "commit": commit, "evidence": []string{proof.String()}},
		},
	})
	data, err = generateModelsReport(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatal(err)
	}
	models := report.Prototypes["zeta"].Models
	if len(models) != 2 {
		t.Fatalf("same-name artifacts lost or duplicated: %+v", models)
	}
	seen := map[string]string{}
	for _, record := range models {
		seen[record.Model] = record.Inference[0].Tier
	}
	if seen[model.String()] != "exact-golden" || seen[sibling.String()] != "real-artifact-smoke" {
		t.Fatalf("same-name artifacts lost independent claims: %+v", seen)
	}
}

func writeSpecification(t *testing.T, root, name string, document map[string]any) {
	t.Helper()
	directory := filepath.Join(root, "docs", "verification")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, name), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
