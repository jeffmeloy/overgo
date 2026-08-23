package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/closurescan"
	"overgo/internal/repodb"
)

func TestMagicGateRejectsUncataloguedChangedAndStaleBindings(t *testing.T) {
	t.Run("unrelated source edit", func(t *testing.T) {
		baseline := closurescan.Candidate{
			Name: "ExistingLimit", Package: "internal/p", File: "internal/p/p.go",
			Scope: "package", Expression: "8", Value: "8", SourceID: "before",
		}
		current := baseline
		current.SourceID = "after"
		if _, err := admitMagicDelta([]closurescan.Candidate{current}, []closurescan.Candidate{baseline}, nil); err != nil {
			t.Fatalf("unrelated source edit rejected: %v", err)
		}
	})
	t.Run("uncatalogued", func(t *testing.T) {
		root, path := magicGateFixture(t, false)
		writeMagicSource(t, path, "package p\nconst ExistingLimit = 8\nconst AddedLimit = 9\n")
		gate := gateContext{repo: root, paths: []string{"internal/p/p.go"}, storePath: "store"}
		if _, err := gate.stepMagics(); err == nil || !strings.Contains(err.Error(), "new or changed uncatalogued") {
			t.Fatalf("uncatalogued error = %v", err)
		}
	})
	t.Run("stale", func(t *testing.T) {
		root, path := magicGateFixture(t, true)
		writeMagicSource(t, path, "package p\nconst ExistingLimit = 9\n")
		gate := gateContext{repo: root, paths: []string{"internal/p/p.go"}, storePath: "store"}
		if _, err := gate.stepMagics(); err == nil || !strings.Contains(err.Error(), "stale active binding") {
			t.Fatalf("stale error = %v", err)
		}
	})
	t.Run("stale callsite", func(t *testing.T) {
		root, _ := magicGateFixture(t, true)
		relative := "internal/p/use.go"
		writeMagicSource(t, filepath.Join(root, filepath.FromSlash(relative)), "package p\nvar configured = 8\n")
		gate := gateContext{repo: root, paths: []string{relative}, storePath: "store"}
		if _, err := gate.stepMagics(); err == nil || !strings.Contains(err.Error(), "callsite") {
			t.Fatalf("stale callsite error = %v", err)
		}
	})
}

func TestMagicGateFailsClosedWhenLedgerUnavailable(t *testing.T) {
	root, path := magicGateFixture(t, false)
	if err := os.RemoveAll(filepath.Join(root, "store")); err != nil {
		t.Fatal(err)
	}
	writeMagicSource(t, path, "package p\nconst ExistingLimit = 9\n")
	gate := gateContext{repo: root, paths: []string{"internal/p/p.go"}, storePath: "store"}
	if _, err := gate.stepMagics(); err == nil {
		t.Fatal("missing ledger accepted")
	}
}

func TestMagicGateRejectsLiteralAndAssumptionDebtIncrease(t *testing.T) {
	for _, fixture := range []struct {
		name, relative, source string
	}{
		{"inline", "internal/p/p.go", "package p\nconst ExistingLimit = 8\nfunc use() int { return 7 }\n"},
		{"assumption", "internal/p/p.go", "package p\nconst ExistingLimit = 8\nfunc fits(tensorRows int) bool { return tensorRows > ExistingLimit }\n"},
		{"test_policy", "internal/p/p_test.go", "package p\nconst expectedExistingLimit = 8\nfunc verify() { if ExistingLimit != expectedExistingLimit { panic(\"policy\") } }\n"},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			root, path := magicGateFixture(t, false)
			if fixture.relative != "internal/p/p.go" {
				path = filepath.Join(root, filepath.FromSlash(fixture.relative))
			}
			writeMagicSource(t, path, fixture.source)
			gate := gateContext{repo: root, paths: []string{fixture.relative}, storePath: "store"}
			if _, err := gate.stepMagics(); err == nil || !strings.Contains(err.Error(), fixture.name) {
				t.Fatalf("%s increase error = %v", fixture.name, err)
			}
		})
	}
}

func TestMagicGateAllowsMeasuredDebtReduction(t *testing.T) {
	baseline := magicDebt{Named: 1, Inline: 1, TestPolicy: 1, Assumption: 1}
	if err := rejectMagicDebtIncrease(magicDebt{}, baseline); err != nil {
		t.Fatalf("measured reduction rejected: %v", err)
	}
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
	store, err := repodb.Open(filepath.Join(root, "store"))
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
	if _, err := store.Commit(context.Background(), artifact.Batch{
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
	if _, err := store.Commit(context.Background(), batch); err != nil {
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
