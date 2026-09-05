package testscope

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestNonGoOwnership(t *testing.T) {
	repo := filepath.Join("root", "repo")
	packages := []Package{
		{
			ImportPath:      "overgo/internal/model",
			Dir:             filepath.Join(repo, "internal", "model"),
			ProductionFiles: []string{"architecture_profiles.json"},
		},
		{
			ImportPath:     "overgo/internal/server",
			Dir:            filepath.Join(repo, "internal", "server"),
			TestEmbedFiles: []string{"testdata/request.json"},
		},
	}
	want := []string{"overgo/internal/model", "overgo/internal/server"}
	got, production := DirectPackages(repo, []string{
		"internal/model/architecture_profiles.json",
		"internal/server/testdata/request.json",
		"docs/plan.json",
	}, packages)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DirectPackages() = %v, want %v", got, want)
	}
	if !reflect.DeepEqual(production, want[:1]) {
		t.Fatalf("production = %v, want %v", production, want[:1])
	}
}
