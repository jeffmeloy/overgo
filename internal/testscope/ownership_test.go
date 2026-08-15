package testscope

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestNonGoOwnership(t *testing.T) {
	repo := filepath.Join("root", "repo")
	packages := []Package{
		{
			ImportPath: "overgo/internal/model",
			Dir:        filepath.Join(repo, "internal", "model"),
			EmbedFiles: []string{"architecture_profiles.json"},
		},
		{
			ImportPath:     "overgo/internal/server",
			Dir:            filepath.Join(repo, "internal", "server"),
			TestEmbedFiles: []string{"testdata/request.json"},
		},
	}
	want := []string{"overgo/internal/model", "overgo/internal/server"}
	got := DirectPackages(repo, []string{
		"internal/model/architecture_profiles.json",
		"internal/server/testdata/request.json",
		"docs/plan.json",
	}, packages)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DirectPackages() = %v, want %v", got, want)
	}
}

func TestDecodePackages(t *testing.T) {
	raw := strings.NewReader(`{"ImportPath":"overgo/a","Dir":"/repo/a","EmbedFiles":["x.json"]}
{"ImportPath":"overgo/b","Dir":"/repo/b"}`)
	packages, err := DecodePackages(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(packages) != 2 || packages[0].EmbedFiles[0] != "x.json" {
		t.Fatalf("unexpected packages: %#v", packages)
	}
}
