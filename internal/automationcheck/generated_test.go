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
	facts, err := GeneratedImpact(root, []string{"kernels/cuda/a.cu"})
	if err != nil {
		t.Fatal(err)
	}
	var calls []string
	checks := GeneratedChecks(root, recordingCommand(&calls))
	planned, err := Plan(checks, facts)
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
	facts, err := GeneratedImpact(root, []string{"go.sum"})
	if err != nil {
		t.Fatal(err)
	}
	var calls []string
	planned, err := Plan(GeneratedChecks(root, recordingCommand(&calls)), facts)
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
	facts, err := GeneratedImpact(root, []string{"internal/runtime/changed.go"})
	if err != nil {
		t.Fatal(err)
	}
	var calls []string
	planned, err := Plan(GeneratedChecks(root, recordingCommand(&calls)), facts)
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
