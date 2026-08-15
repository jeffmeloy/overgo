package closurescan

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestClosureScanSnapshotConsumer(t *testing.T) {
	root := t.TempDir()
	for relative, content := range map[string]string{
		"internal/live.go":      "package p\nconst Limit = 3\n",
		"internal/other.go":     "package p\nconst Other = 4\n",
		"internal/live_test.go": "package p\nconst TestLimit = 5\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
	candidates, err := ScanSnapshot(snapshot, []string{"internal/live.go"})
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 1 || candidates[0].Name != "Limit" {
		t.Fatalf("candidates = %+v", candidates)
	}
}

func TestRankRawPolicyLiterals(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "policy.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
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
	if err := os.WriteFile(path, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	snapshot, err := repoanalysis.DiscoverGo(root, "internal")
	if err != nil {
		t.Fatal(err)
	}
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
