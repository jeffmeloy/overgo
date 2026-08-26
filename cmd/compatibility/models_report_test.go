package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestModelsReportNestsSpecificModels pins the report contract:
// prototypes are identified without status, each specific model nests
// inside the prototype its declared architecture attributes it to, a
// specification's weaker duplicate claim reduces away, and a model
// with no resolvable architecture lands in the explicit unattributed
// bucket instead of a guessed one.
func TestModelsReportNestsSpecificModels(t *testing.T) {
	root := t.TempDir()
	writeTestManifest(t, root, []claim{}, map[string]modelClaim{
		"zeta": {Features: []string{"z"}},
	})
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
	data, err := generateModelsReport(root, filepath.Join(root, "absent-store"))
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
