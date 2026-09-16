package codemanifest

import (
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

func TestChangedCompilerSelectionInvalidatesCache(t *testing.T) {
	root := t.TempDir()
	name := "internal/example/example.go"
	writeGeneratorFixture(t, root, name, "package example\nfunc Value() int { return 1 }\n")
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	selection := gosource.BuildSelection{Context: "linux/amd64", Root: root, Files: map[string]bool{name: true}, Packages: map[string]string{name: "overgo/internal/example"}}
	cache, err := NewCache(2)
	if err != nil {
		t.Fatal(err)
	}
	before, reused, err := cache.Generate(snapshot, []gosource.BuildSelection{selection}, nil)
	if err != nil || reused || len(before.Symbols) != 1 {
		t.Fatalf("baseline: %v reused=%t symbols=%d", err, reused, len(before.Symbols))
	}
	selection.Files[name] = false
	want, err := Generate(snapshot, []gosource.BuildSelection{selection}, nil)
	if err != nil {
		t.Fatal(err)
	}
	after, reused, err := cache.Generate(snapshot, []gosource.BuildSelection{selection}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if reused || after.ID != want.ID {
		t.Fatalf("changed compiler file selection reused=%t cached_symbols=%d fresh_symbols=%d same_context=%s", reused, len(after.Symbols), len(want.Symbols), selection.Context)
	}
	selection.Files[name] = true
	inputs := []ExternalInput{{Path: "README.md", ContentID: fixtureDigest, Kind: "repository-file", Owner: "."}}
	first, _, err := cache.Generate(snapshot, []gosource.BuildSelection{selection}, inputs)
	if err != nil {
		t.Fatal(err)
	}
	inputs[0].ContentID = "1123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	rebound, reused, err := cache.Generate(snapshot, []gosource.BuildSelection{selection}, inputs)
	if err != nil || reused || first.ID == rebound.ID {
		t.Fatalf("external rebind: %v reused=%t identity=%s", err, reused, rebound.ID)
	}
	fresh, err := Generate(snapshot, []gosource.BuildSelection{selection}, inputs)
	if err != nil || fresh.ID != rebound.ID {
		t.Fatalf("rebound manifest differs from fresh analysis: %v", err)
	}
	if first.ExternalInputs[0].ContentID != fixtureDigest {
		t.Fatal("rebind mutated the retained manifest")
	}
	again, reused, err := cache.Generate(snapshot, []gosource.BuildSelection{selection}, inputs)
	if err != nil || !reused || again.ID != rebound.ID {
		t.Fatalf("exact reuse: %v reused=%t", err, reused)
	}
	t.Run("resolved dependency package name", func(t *testing.T) {
		writeGeneratorFixture(t, root, "go.mod", "module fixture\n\ngo 1.26\nrequire example.com/dependency v0.0.0\nreplace example.com/dependency => ./dependency\n")
		writeGeneratorFixture(t, root, "dependency/go.mod", "module example.com/dependency\n\ngo 1.26\n")
		writeGeneratorFixture(t, root, "dependency/value.go", "package dependency\nfunc Value() int { return 1 }\n")
		writeGeneratorFixture(t, root, name, "package example\nimport \"example.com/dependency\"\nfunc Value() int { return dependency.Value() }\n")
		snapshot, err := repoanalysis.DiscoverGo(root, "internal")
		if err != nil {
			t.Fatal(err)
		}
		cache, err := NewCache(2)
		if err != nil {
			t.Fatal(err)
		}
		if _, reused, err := cache.Generate(snapshot, []gosource.BuildSelection{selection}, nil); err != nil || reused {
			t.Fatalf("first dependency: %v reused=%t", err, reused)
		}
		writeGeneratorFixture(t, root, "dependency/value.go", "package renamed\nfunc Value() int { return 1 }\n")
		if _, reused, err := cache.Generate(snapshot, []gosource.BuildSelection{selection}, nil); err != nil || reused {
			t.Fatalf("changed dependency name: %v reused=%t", err, reused)
		}
	})
}
