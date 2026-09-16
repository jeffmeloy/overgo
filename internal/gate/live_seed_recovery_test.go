package gate

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestLiveSeedRestoresAfterFailure(t *testing.T) {
	const environment = "OVERGO_GATE_SEED_RECOVERY_TEST"
	if root := os.Getenv(environment); root != "" {
		live := &liveRepository{worktree: root}
		t.Run("partial seed", func(t *testing.T) {
			live.plan(t, []string{"first.txt", "first.txt", "missing.txt"}, func(*gateContext) { t.Fatal("partial seed reached planning") })
		})
		if !live.mutex.TryLock() {
			t.Fatal("failed seed retained the fixture lock")
		}
		live.mutex.Unlock()
		t.Log("fixture lock released after seed failure")
		return
	}
	t.Parallel()
	root := t.TempDir()
	path := filepath.Join(root, "first.txt")
	const original = "original fixture"
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	child := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestLiveSeedRestoresAfterFailure$", "-test.v")
	child.Env = append(os.Environ(), environment+"="+root)
	output, err := child.CombinedOutput()
	if err == nil || !strings.Contains(string(output), "missing.txt") || !strings.Contains(string(output), "fixture lock released after seed failure") {
		t.Fatalf("child did not reach the intended failure and release: %v\n%s", err, output)
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != original {
		t.Fatalf("failed seed contaminated the next test: %q %v", data, err)
	}
}
