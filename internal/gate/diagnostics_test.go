package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/overgodb"
)

func TestPermanentMagicGate(t *testing.T) {
	t.Parallel()
	t.Run("accepts exact closed authority", func(t *testing.T) {
		root, _ := magicGateFixture(t, true)
		gate := gateContext{repo: root, paths: []string{"internal/p/p.go"}, storePath: "store"}
		if _, err := gate.stepMagics(); err != nil {
			t.Fatalf("closed authority rejected: %v", err)
		}
	})
	t.Run("rejects uncatalogued policy", func(t *testing.T) {
		root, path := magicGateFixture(t, false)
		writeMagicSource(t, path, "package p\nconst ExistingLimit = 8\n")
		gate := gateContext{repo: root, paths: []string{"internal/p/p.go"}, storePath: "store"}
		if _, err := gate.stepMagics(); err == nil || !strings.Contains(err.Error(), "uncatalogued production policy") {
			t.Fatalf("uncatalogued error = %v", err)
		}
	})
	t.Run("rejects stale authority", func(t *testing.T) {
		root, path := magicGateFixture(t, true)
		writeMagicSource(t, path, "package p\nconst ExistingLimit = 9\n")
		gate := gateContext{repo: root, paths: []string{"internal/p/p.go"}, storePath: "store"}
		if _, err := gate.stepMagics(); err == nil || !strings.Contains(err.Error(), "stale active binding") {
			t.Fatalf("stale error = %v", err)
		}
	})
	t.Run("rejects copied test policy", func(t *testing.T) {
		root, _ := magicGateFixture(t, true)
		relative := "internal/p/p_test.go"
		writeMagicSource(t, filepath.Join(root, filepath.FromSlash(relative)),
			"package p\nconst expectedExistingLimit = 8\nfunc verify() { if ExistingLimit != expectedExistingLimit { panic(\"policy\") } }\n")
		gate := gateContext{repo: root, paths: []string{relative}, storePath: "store"}
		if _, err := gate.stepMagics(); err == nil || !strings.Contains(err.Error(), "test policy copy") {
			t.Fatalf("policy-copy error = %v", err)
		}
	})
	t.Run("fails closed without ledger", func(t *testing.T) {
		root, _ := magicGateFixture(t, false)
		if err := os.RemoveAll(filepath.Join(root, "store")); err != nil {
			t.Fatal(err)
		}
		gate := gateContext{repo: root, paths: []string{"internal/p/p.go"}, storePath: "store"}
		if _, err := gate.stepMagics(); err == nil {
			t.Fatal("missing ledger accepted")
		}
	})
}

func magicGateFixture(t *testing.T, publish bool) (string, string) {
	t.Helper()
	root := t.TempDir()
	path := filepath.Join(root, "internal", "p", "p.go")
	writeMagicSource(t, path, "package p\nconst ExistingLimit = 8\n")
	writeMagicSource(t, filepath.Join(root, "internal", "p", "use.go"), "package p\nvar configured = ExistingLimit\n")
	for _, args := range [][]string{
		{"init"}, {"add", "."},
		{"-c", "user.name=fixture", "-c", "user.email=fixture@example.com", "commit", "-m", "base"},
	} {
		if _, err := command(root, "git", args...); err != nil {
			t.Fatal(err)
		}
	}
	store, err := overgodb.Open(filepath.Join(root, "store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if !publish {
		return root, path
	}
	candidates, err := closurescan.ScanRoot(root, closurescan.CandidateConstants)
	if err != nil || len(candidates) != 1 {
		t.Fatalf("candidates = (%+v, %v)", candidates, err)
	}
	binding, err := candidates[0].Binding()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "magic/dependency", Artifacts: []artifact.Descriptor{{ID: binding.Owner}},
	}); err != nil {
		t.Fatal(err)
	}
	document, err := closureledger.New(
		candidates[0].Name, candidates[0].ValueJSON(), closureledger.TierImplementation,
		closureledger.StatusClosed, "Fixed gate fixture.", []closureledger.SourceBinding{binding},
		"Replace with the fixture contract.", "Fixture contract change.", binding.Owner,
	)
	if err != nil {
		t.Fatal(err)
	}
	alias, err := closureledger.ActiveAlias(binding)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := document.Batch("magic/document", &artifact.AliasBinding{Name: alias, Target: document.ID})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), batch); err != nil {
		t.Fatal(err)
	}
	return root, path
}

func writeMagicSource(t *testing.T, path, source string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
}
