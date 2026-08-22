package automationcheck

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestManifestCheck(t *testing.T) {
	root := generatedRoot(t)
	if err := os.MkdirAll(filepath.Join(root, "kernels"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "kernels", "manifest.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	var calls []string
	checks := GeneratedChecks(root, recordingCommand(&calls))
	impact := OwnershipImpact(checks, Surface{Identity: "candidate", Packages: []string{"cmd/kernel-manifest"}})
	planned, err := Plan(checks, impact)
	if err != nil || len(planned) != 1 || planned[0].Check.Name != "manifest" {
		t.Fatalf("plan = %+v, %v", planned, err)
	}
	if _, err := Run(context.Background(), planned[0]); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 || !strings.Contains(calls[1], "TestGeneratedBindingsMatchManifest") {
		t.Fatalf("calls = %v", calls)
	}
}

func TestSBOMCheck(t *testing.T) {
	root := generatedRoot(t)
	var calls []string
	checks := GeneratedChecks(root, recordingCommand(&calls))
	impact := OwnershipImpact(checks, Surface{Identity: "candidate", Packages: []string{"cmd/sbom"}})
	planned, err := Plan(checks, impact)
	if err != nil || len(planned) != 1 || planned[0].Check.Name != "sbom" {
		t.Fatalf("plan = %+v, %v", planned, err)
	}
	if _, err := Run(context.Background(), planned[0]); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 1 || !strings.Contains(calls[0], "./cmd/sbom -check") {
		t.Fatalf("calls = %v", calls)
	}
}

func TestCompatibilityCheck(t *testing.T) {
	root := generatedRoot(t)
	if err := os.WriteFile(filepath.Join(root, "compatibility.json"), []byte(`{"evidence":"internal/runtime/changed.go"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var calls []string
	checks := GeneratedChecks(root, recordingCommand(&calls))
	impact := OwnershipImpact(checks, Surface{Identity: "candidate", Packages: []string{"internal/model"}})
	planned, err := Plan(checks, impact)
	if err != nil || len(planned) != 1 || planned[0].Check.Name != "claims" {
		t.Fatalf("plan = %+v, %v", planned, err)
	}
	if _, err := Run(context.Background(), planned[0]); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(calls, []string{"go run ./cmd/compatibility -check"}) {
		t.Fatalf("calls = %v", calls)
	}
}

func TestGeneratedImpactUsesDeclaredOwnership(t *testing.T) {
	checks := GeneratedChecks(t.TempDir(), recordingCommand(new([]string)))
	tests := []struct {
		packagePath, check string
	}{
		{"cmd/kernel-bindings", manifestCheckName},
		{"cmd/sbom", sbomCheckName},
		{"cmd/compatibility", compatibilityCheckName},
	}
	for _, test := range tests {
		impact := OwnershipImpact(checks, Surface{Identity: "candidate", Packages: []string{test.packagePath}})
		planned, err := Plan(checks, impact)
		if err != nil || len(planned) != 1 || planned[0].Check.Name != test.check {
			t.Fatalf("package %s planned = %+v impact=%+v err=%v", test.packagePath, planned, impact, err)
		}
	}
}

func generatedRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "compatibility.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func recordingCommand(calls *[]string) Command {
	return func(_ string, name string, arguments ...string) (string, error) {
		*calls = append(*calls, strings.Join(append([]string{name}, arguments...), " "))
		return "", nil
	}
}
