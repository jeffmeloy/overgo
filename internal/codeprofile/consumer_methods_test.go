package codeprofile

import (
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

// TestCensusReportsUnusedConcreteMethods pins the audit's census finding:
// methods are no longer blanket "method-dispatch". An unexported method
// matching no declared interface is direct-call only -- unused ones report as
// zero-consumer. Exported methods and interface-matching methods keep the
// dispatch boundary (they may satisfy interfaces outside the snapshot), and
// test-support packages are identified through imports, not names.
func TestCensusReportsUnusedConcreteMethods(t *testing.T) {
	snapshot, err := (repoanalysis.SourceSnapshot{}).Overlay(map[string][]byte{
		"internal/api/api.go": []byte(`package api
type sink interface{ drain() }
type Worker struct{}
func (Worker) drain() {}
func (Worker) deadHelper() {}
func (Worker) usedHelper() int { return 1 }
func (Worker) Exported() {}
func Run(w Worker) int { return w.usedHelper() }
`),
		"internal/helpers/helpers.go": []byte("package helpers\nfunc Support() {}\n"),
		"internal/api/api_test.go":    []byte("package api\nimport h \"example/internal/helpers\"\nfunc TestOnly() { h.Support() }\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{
		Context: "linux/amd64",
		Files: map[string]bool{
			"internal/api/api.go": true, "internal/api/api_test.go": true, "internal/helpers/helpers.go": true,
		},
		Packages: map[string]string{
			"internal/api/api.go":         "example/internal/api",
			"internal/api/api_test.go":    "example/internal/api",
			"internal/helpers/helpers.go": "example/internal/helpers",
		},
	}
	declarations, _, err := ProductionConsumerCensus(snapshot, selection, nil)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]ConsumerDeclaration{}
	for _, declaration := range declarations {
		byName[declaration.Name] = declaration
	}
	if d := byName["deadHelper"]; d.Boundary != "" || d.ProductionReferences != 0 {
		t.Errorf("deadHelper = boundary %q refs %d, want reportable zero-consumer", d.Boundary, d.ProductionReferences)
	}
	if d := byName["usedHelper"]; d.Boundary != "" || d.ProductionReferences == 0 {
		t.Errorf("usedHelper = boundary %q refs %d, want counted direct calls", d.Boundary, d.ProductionReferences)
	}
	if d := byName["drain"]; d.Boundary != "method-dispatch" {
		t.Errorf("drain boundary = %q, want method-dispatch (declared interface)", d.Boundary)
	}
	if d := byName["Exported"]; d.Boundary != "method-dispatch" {
		t.Errorf("Exported boundary = %q, want method-dispatch (external interfaces unknowable)", d.Boundary)
	}
	if d := byName["Support"]; d.Boundary != "test-helper" {
		t.Errorf("Support boundary = %q, want test-helper via imports", d.Boundary)
	}
}
