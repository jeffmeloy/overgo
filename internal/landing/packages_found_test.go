package landing

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestALandingRunsANamedTestWhereItIs holds a step's check to running where its test is declared, found by the landing
// (fail-early/packages-found): a check naming ./a and ./b whose test only ./c declares runs in ./c, the rest of it as
// written, and the landing says so; a check already naming where its test is runs as written; and a check whose test no
// test file declares is refused naming the test. On 2026-11-10 card-budgets/declared-budget's check named three packages
// written before its work was surveyed and was refused twice, after the full stages, as matching no passing test.
func TestALandingRunsANamedTestWhereItIs(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testcheck.NoError(t, exec.Command("git", "-C", root, "init", "-q").Run())
	for name, content := range map[string]string{
		"a/a_test.go": "package a\n\nimport \"testing\"\n\nfunc TestOther(t *testing.T) {}\n",
		"c/c_test.go": "package c\n\nimport \"testing\"\n\nfunc TestBudget(t *testing.T) {}\n",
	} {
		path := filepath.Join(root, filepath.FromSlash(name))
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(path), 0o755))
		testcheck.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	}
	resolved, said, err := whereDeclared(t.Context(), root, "go test ./a ./b -count=1 -run '^TestBudget$'")
	testcheck.NoError(t, err)
	if want := "go test ./c -count=1 -run '^TestBudget$'"; resolved != want {
		t.Errorf("a check naming ./a and ./b for a test ./c declares runs %q; want %q", resolved, want)
	}
	if !strings.Contains(said, "./c") || !strings.Contains(said, "./a ./b") {
		t.Errorf("the landing says %q; want it to name ./c and the packages the check named", said)
	}
	right := "go test ./c -count=1 -run '^TestBudget$'"
	if resolved, said, err := whereDeclared(t.Context(), root, right); err != nil || resolved != right || said != "" {
		t.Errorf("a check naming where its test is runs %q saying %q (%v); want it as written, saying nothing", resolved, said, err)
	}
	if _, _, err := whereDeclared(t.Context(), root, "go test ./a -count=1 -run '^TestMissing$'"); err == nil || !strings.Contains(err.Error(), "TestMissing") {
		t.Errorf("a check whose test no file declares answered %v; want a refusal naming TestMissing", err)
	}
}
