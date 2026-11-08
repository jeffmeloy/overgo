package main

import (
	"bytes"
	"encoding/json"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

func TestGenerateIsDeterministicAndValidJSON(t *testing.T) {
	t.Parallel()
	first := testcheck.Must(generate("../..", moduleLicenses))(t)
	second := testcheck.Must(generate("../..", moduleLicenses))(t)
	if !bytes.Equal(first, second) {
		t.Fatal("SBOM output is not deterministic")
	}
	var document map[string]any
	testcheck.NoError(t, json.Unmarshal(first, &document))
	if document["bomFormat"] != "CycloneDX" || document["specVersion"] != "1.6" {
		t.Fatalf("document header = %#v", document)
	}
}

func TestSBOMOmitUndeclaredLicenses(t *testing.T) {
	t.Parallel()
	generated := testcheck.Must(generate("../..", moduleLicenses))(t)
	if bytes.Contains(generated, []byte("NOASSERTION")) {
		t.Fatal("SBOM retains undeclared-license policy")
	}
	var document struct {
		Metadata struct {
			Component map[string]any `json:"component"`
		} `json:"metadata"`
	}
	testcheck.NoError(t, json.Unmarshal(generated, &document))
	if _, ok := document.Metadata.Component["licenses"]; ok {
		t.Fatal("project component declares a license")
	}
}

// TestTheSBOMListsWhatTheBuildCompilesIn holds the SBOM to the owner's rule of 2026-11-08: no model is ever
// distributed, and the SBOM lists only the third-party code the build compiles in. It listed 61 entries: the module
// graph go list -m all gives, six modules of it no package compiles in, and overgo's own kernel and PTX files, with
// licence tags on the two llama.cpp tables, though code ported or generated into overgo is overgo's own. Its libraries
// are exactly the modules the packages compile in, each with its licence; beside them stand the standard library and
// the NVIDIA libraries the binaries load, each an external requirement the machine provides; no file is a component;
// and a compiled-in module whose licence is not declared is refused.
func TestTheSBOMListsWhatTheBuildCompilesIn(t *testing.T) {
	t.Parallel()
	var document struct {
		Components []struct {
			Type, Name, Scope string
			Licenses          []any
			Properties        []struct{ Name, Value string }
		}
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(generate("../..", moduleLicenses))(t), &document))
	compiled := testcheck.Must(compiledModules("../.."))(t)
	var libraries, external []string
	for _, entry := range document.Components {
		switch {
		case entry.Type == "file":
			t.Fatalf("overgo's own file %s is listed as a component", entry.Name)
		case len(entry.Licenses) == 0:
			t.Fatalf("%s is listed with no licence", entry.Name)
		case entry.Type == "library":
			libraries = append(libraries, entry.Name)
		case entry.Scope != "":
			external = append(external, entry.Name)
		}
	}
	want := make([]string, len(compiled))
	for index, item := range compiled {
		want[index] = item.path
	}
	if !slices.Equal(libraries, want) {
		t.Fatalf("the SBOM lists the modules %v, and the packages compile in %v", libraries, want)
	}
	graph := testcheck.Must(exec.Command("go", "list", "-m", "all").Output())(t)
	for _, line := range strings.Split(strings.TrimSpace(string(graph)), "\n")[1:] {
		path, _, _ := strings.Cut(line, " ")
		if !slices.Contains(want, path) && bytes.Contains(testcheck.Must(json.Marshal(document))(t), []byte(path)) {
			t.Fatalf("the SBOM lists %s, which no package compiles in", path)
		}
	}
	if len(external) != len(requirements) {
		t.Fatalf("the external requirements listed are %v, want the %d NVIDIA libraries the binaries load", external, len(requirements))
	}
	undeclared := maps.Clone(moduleLicenses)
	delete(undeclared, "github.com/icza/bitio")
	if _, err := generate("../..", undeclared); err == nil {
		t.Fatal("a compiled-in module with no declared licence was listed")
	}
}

func TestReleaseIntegrityContract(t *testing.T) {
	t.Parallel()
	generated := testcheck.Must(generate("../..", moduleLicenses))(t)
	committed := testcheck.Must(os.ReadFile(filepath.Join("..", "..", sbomPath)))(t)
	if !bytes.Equal(generated, committed) {
		t.Fatal("committed SBOM does not match the dependency graph")
	}
}

// TestEveryNoticeIsOfWhatOvergoCopies holds licenses/ to the owner's rule of 2026-11-08: it holds notices only for
// what overgo copies, the third-party modules compiled into the binaries, data copied verbatim and units ported
// literally, while a port that rebuilds a behaviour in overgo's own code needs none. regexp2 and bitio, compiled in,
// had no notice there, and four notices covered ports that rebuild behaviour or nothing at all. Every compiled-in
// module has its notice, and every notice is a compiled-in module's or is cited by the code or document of what it
// covers.
func TestEveryNoticeIsOfWhatOvergoCopies(t *testing.T) {
	t.Parallel()
	root := filepath.Join("..", "..")
	notices := testcheck.Must(os.ReadDir(filepath.Join(root, "licenses")))(t)
	held := map[string]bool{}
	for _, notice := range notices {
		held[notice.Name()] = false
	}
	for _, item := range testcheck.Must(compiledModules(root))(t) {
		notice, noted := moduleNotices[item.path]
		if _, present := held[notice]; !noted || !present {
			t.Fatalf("the build compiles in %s, and licenses/ holds no notice of it", item.path)
		}
		held[notice] = true
	}
	for _, directory := range []string{"internal", "cmd", "kernels", "docs"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(root, directory), func(path string, entry os.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !slices.Contains([]string{".go", ".cu", ".cuh", ".md"}, filepath.Ext(path)) {
				return err
			}
			data, err := os.ReadFile(path)
			for notice := range held {
				held[notice] = held[notice] || bytes.Contains(data, []byte("licenses/"+notice))
			}
			return err
		}))
	}
	for notice, covered := range held {
		if !covered {
			t.Fatalf("licenses/%s is the notice of no compiled-in module, and no code or document of what overgo copies cites it", notice)
		}
	}
}
