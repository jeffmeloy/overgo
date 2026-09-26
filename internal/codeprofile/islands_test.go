package codeprofile

import (
	"slices"
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

// TestPrivateCodeReachedOnlyByTestsIsFound holds the island detector to
// following liveness from the command entry: a private helper reached only
// through an export that only a test calls is an island, and a private helper
// the live command also reaches is not, however a test reaches it too.
func TestPrivateCodeReachedOnlyByTestsIsFound(t *testing.T) {
	t.Parallel()
	files := map[string][]byte{
		"cmd/tool/main.go":            []byte("package main\nimport \"example/internal/lib\"\nfunc main() { lib.Run() }\n"),
		"internal/lib/lib.go":         []byte("package lib\nfunc Run() { shared() }\nfunc shared() {}\nfunc TrainOnlyInTests() { stranded(); shared() }\nfunc stranded() { deeper() }\nfunc deeper() {}\n"),
		"internal/lib/lib_test.go":    []byte("package lib\nimport \"testing\"\nfunc TestTrain(t *testing.T) { TrainOnlyInTests() }\n"),
		"internal/lib/initialized.go": []byte("package lib\nfunc init() { registered() }\nfunc registered() {}\n"),
	}
	snapshot, err := (repoanalysis.SourceSnapshot{}).Overlay(files)
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{Context: "linux/amd64", Files: map[string]bool{}, Packages: map[string]string{}}
	for path := range files {
		selection.Files[path] = true
		selection.Packages[path] = map[bool]string{true: "example/cmd/tool", false: "example/internal/lib"}[path == "cmd/tool/main.go"]
	}
	declarations, references, _, err := ProductionConsumerGraph(snapshot, selection)
	if err != nil {
		t.Fatal(err)
	}
	var islands []string
	for _, island := range PrivateIslands(declarations, references) {
		islands = append(islands, island.Name)
	}
	slices.Sort(islands)
	if !slices.Equal(islands, []string{"deeper", "stranded"}) {
		t.Fatalf("private islands = %v, want the helpers reached only through the test-only export", islands)
	}
}
