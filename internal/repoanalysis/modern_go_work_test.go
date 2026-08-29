package repoanalysis

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestModernGoWorkSelectionDeterministic(t *testing.T) {
	root := modernGoWorkTestRepository(t)
	request := ModernGoWorkRequest{Guidelines: []string{"maps_copy", "maps_copy"}, Limit: 1}
	first, err := BuildModernGoWorkSelection(root, ModernGoTargetVersion, request)
	if err != nil {
		t.Fatal(err)
	}
	second, err := BuildModernGoWorkSelection(root, ModernGoTargetVersion, request)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("identical source and request produced different work selections")
	}
	if !reflect.DeepEqual(first.Request.Guidelines, []string{"maps_copy"}) {
		t.Fatalf("normalized guidelines = %v", first.Request.Guidelines)
	}
}

func TestModernGoWorkSelectionDerivesVerification(t *testing.T) {
	selection, err := BuildModernGoWorkSelection(
		modernGoWorkTestRepository(t), ModernGoTargetVersion,
		ModernGoWorkRequest{Guidelines: []string{"maps_copy"}, Limit: 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(selection.Verification.DirectPackages, []string{"example/internal/sample"}) {
		t.Fatalf("direct packages = %v", selection.Verification.DirectPackages)
	}
	if !reflect.DeepEqual(selection.Verification.DependentPackages, []string{"example/cmd/app"}) {
		t.Fatalf("dependent packages = %v", selection.Verification.DependentPackages)
	}
	want := []string{"test", "example/cmd/app", "example/internal/sample"}
	if !reflect.DeepEqual(selection.Verification.ClosureArgs, want) {
		t.Fatalf("closure args = %v, want %v", selection.Verification.ClosureArgs, want)
	}
	if len(selection.Findings) != 1 || selection.Findings[0].Risk != ModernGoRiskMechanical {
		t.Fatalf("risk-classified findings = %+v", selection.Findings)
	}
}

func modernGoWorkTestRepository(t *testing.T) string {
	t.Helper()
	root := modernGoTestRepository(t, `package sample

func Copy(dst, src map[string]int) {
	for key, value := range src {
		dst[key] = value
	}
}
`)
	directory := filepath.Join(root, "cmd", "app")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "main.go"), []byte(`package main

import "example/internal/sample"

func main() { sample.Copy(map[string]int{}, map[string]int{}) }
`), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}
