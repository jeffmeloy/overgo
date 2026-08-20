package closurescan

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestClosureScanSnapshotConsumer(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/live.go":      "package p\nconst Limit = 3\n",
		"internal/other.go":     "package p\nconst Other = 4\n",
		"internal/live_test.go": "package p\nconst TestLimit = 5\n",
	})
	candidates, err := ScanSnapshot(snapshot, []string{"internal/live.go"})
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 1 || candidates[0].Name != "Limit" {
		t.Fatalf("candidates = %+v", candidates)
	}
}

func TestScanIncludesFunctionLocalConstants(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{"internal/policy.go": `package policy
const base = 3
func derive() {
	const base = 5
	const local = base * 4
}
`})
	candidates, err := ScanSnapshot(snapshot, nil)
	if err != nil {
		t.Fatal(err)
	}
	byScope := map[string]Candidate{}
	for _, candidate := range candidates {
		byScope[candidate.Scope+"/"+candidate.Name] = candidate
	}
	if got := byScope["package/base"]; got.Value != "3" || got.Expression != "3" {
		t.Fatalf("package base = %+v", got)
	}
	if got := byScope["derive/base"]; got.Value != "5" || got.Line == 0 {
		t.Fatalf("local base = %+v", got)
	}
	if got := byScope["derive/local"]; got.Value != "20" || got.Expression != "base * 4" {
		t.Fatalf("derived local = %+v", got)
	}
}

func TestCandidateIdentityIncludesEvaluatedSourceBinding(t *testing.T) {
	snapshot := scanTestSnapshot(t, map[string]string{
		"internal/policy.go": "package policy\nconst allocation = 4 * 8\n",
	})
	before, err := ScanSnapshot(snapshot, nil)
	if err != nil || len(before) != 1 {
		t.Fatalf("before = %+v, %v", before, err)
	}
	rewritten, err := snapshot.Overlay(map[string][]byte{
		"internal/policy.go": []byte("package policy\nconst allocation = 2 * 16\n"),
	})
	if err != nil {
		t.Fatal(err)
	}
	after, err := ScanSnapshot(rewritten, nil)
	if err != nil || len(after) != 1 {
		t.Fatalf("after = %+v, %v", after, err)
	}
	left, right := before[0], after[0]
	if left.Package != "internal" || left.Scope != "package" ||
		left.Line == 0 || left.SourceID == "" || left.Expression != "4 * 8" || left.Value != "32" {
		t.Fatalf("source binding = %+v", left)
	}
	if left.DeclarationKey() != right.DeclarationKey() || left.ExactKey() == right.ExactKey() || right.Value != left.Value {
		t.Fatalf("before = %+v, after = %+v", left, right)
	}
}

func TestRankRawPolicyLiterals(t *testing.T) {
	source := `package policy
func first(n int, values []int) bool {
	_ = 3 * 3
	return n > 4096 && values[3] > 0 && "strict" != ""
}
func second(n int) bool {
	_ = 6 / 3
	return n <= 4096 && "strict" != ""
}
`
	snapshot := scanTestSnapshot(t, map[string]string{"internal/policy.go": source})
	ranked, err := RankRawPolicyLiterals(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range ranked {
		if row.Value == "3" {
			t.Fatalf("structural math literal ranked: %+v", row)
		}
	}
	if len(ranked) == 0 || ranked[0].Value != "4096" || len(ranked[0].Functions) != 2 {
		t.Fatalf("ranked = %+v", ranked)
	}
}

func scanTestSnapshot(t *testing.T, files map[string]string) repoanalysis.SourceSnapshot {
	t.Helper()
	root := t.TempDir()
	for relative, content := range files {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}
