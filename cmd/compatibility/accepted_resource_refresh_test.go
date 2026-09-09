package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/longform"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
)

// TestAcceptedE4BResourceRefresh checks current protocol and resource evidence
// while preserving the original quality and native-mask bundle.
func TestAcceptedE4BResourceRefresh(t *testing.T) {
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT for canonical resource refresh acceptance")
	}
	root := testutil.RepoRoot(t)
	var selected struct {
		Surface string                       `json:"surface"`
		Media   modelValidationSpecification `json:"media"`
		Text    artifact.ID                  `json:"text"`
	}
	if err := jsonfile.Decode(filepath.Join(root, "docs", "verification", "e4b-resource-refresh.json"), &selected); err != nil {
		t.Fatal(err)
	}
	surface, err := longform.Surface(t.Context(), root)
	if err != nil || surface != selected.Surface {
		t.Fatalf("resource evidence does not bind the current inference surface: %v", err)
	}
	var original modelValidationSpecification
	if err := jsonfile.Decode(filepath.Join(root, "docs", "verification", "e4b-validation.json"), &original); err != nil {
		t.Fatal(err)
	}
	if selected.Media.Model != original.Model || selected.Media.Projector != original.Projector {
		t.Fatal("resource refresh substituted the accepted model or projector")
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var required []string
	for _, name := range e4bValidationCells() {
		if strings.HasPrefix(name, "resources/") || strings.HasPrefix(name, "protocol/") {
			required = append(required, name)
		}
	}
	if err := checkModelValidation(t.Context(), store, selected.Media, required); err != nil {
		t.Fatal(err)
	}
	content, found, err := artifact.ReadContent(t.Context(), store, selected.Text)
	if err != nil || !found || selected.Text.Kind() != artifact.KindEvidence {
		t.Fatalf("text recovery capture absent: %v", err)
	}
	var capture struct {
		CodeCommit  string      `json:"code_commit"`
		Surface     string      `json:"surface"`
		Model       artifact.ID `json:"model"`
		Recipe      artifact.ID `json:"recipe"`
		Environment artifact.ID `json:"environment"`
		TestSHA256  string      `json:"test_sha256"`
		Events      string      `json:"events"`
	}
	if err := json.Unmarshal(content.Data, &capture); err != nil {
		t.Fatal(err)
	}
	active, found, err := modelrecipe.ActiveRecord(t.Context(), store, original.Model, recipe.TaskInference)
	if err != nil || !found || capture.Model != original.Model || capture.Recipe != active.Definition.ID ||
		capture.CodeCommit != selected.Media.CodeCommit || capture.Surface != surface || capture.Environment != selected.Media.Cells[0].Environment {
		t.Fatalf("text recovery execution identity differs: %v", err)
	}
	source, err := os.ReadFile(filepath.Join(root, "cmd", "longform", "e4b_recovery_windows_test.go"))
	if err != nil || capture.TestSHA256 != fmt.Sprintf("%x", sha256.Sum256(source)) {
		t.Fatalf("text recovery contract changed: %v", err)
	}
	report, err := testevidence.GoTestJSONReport(capture.Events)
	if err != nil || testevidence.RequireComplete(report) != nil || report.PassedTests != 1 ||
		report.PassedPackages != 1 || !report.PackagePassed("overgo/cmd/longform") {
		t.Fatalf("text recovery did not execute completely: %+v %v", report, err)
	}
	for line := range strings.SplitSeq(strings.TrimSpace(capture.Events), "\n") {
		var event struct{ Package, Test string }
		if err := json.Unmarshal([]byte(line), &event); err != nil || event.Package != "overgo/cmd/longform" ||
			event.Test != "" && event.Test != "TestE4BResourceRecovery" {
			t.Fatalf("foreign recovery event: %s %v", line, err)
		}
	}
	t.Log("18 protocol cases, 270 media resource requests across six modes and full 32768-token text recovery accepted at current surface; original quality and mask evidence retained")
}
