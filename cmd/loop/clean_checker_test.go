package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestALandingsCheckerIsBuiltFromCommittedCode holds the checker a landing runs to the trunk's commit: in a fixture
// repository whose gate prints "committed" and whose checkout holds an uncommitted edit printing "uncommitted", the
// checker built prints "committed", and the edit stays in the checkout as it was. The gate had been built in the
// checkout, so 3e7b49d0c was checked with uncommitted counts code (clean-checker/do).
func TestALandingsCheckerIsBuiltFromCommittedCode(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	gate := func(says string) string {
		return "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Print(\"" + says + "\") }\n"
	}
	for path, content := range map[string]string{
		"go.mod": "module fixture\n\ngo 1.26\n", "cmd/gate/main.go": gate("committed"),
	} {
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(filepath.Join(root, path)), 0o755))
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, path), []byte(content), 0o644))
	}
	for _, arguments := range [][]string{{"init", "-q"}, {"add", "."}, {"-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "fixture"}} {
		testcheck.NoError(t, gitWrite(root, arguments...))
	}
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, "cmd/gate/main.go"), []byte(gate("uncommitted")), 0o644))
	checker := filepath.Join(t.TempDir(), "gate.exe")
	build := func(arguments ...string) (string, error) {
		output, err := exec.Command(arguments[0], arguments[1:]...).CombinedOutput()
		return string(output), err
	}
	if refusal, err := buildChecker(root, checker, "./cmd/gate", build); err != nil {
		t.Fatalf("the checker was not built: %v: %s", err, refusal)
	}
	if said := testcheck.Must(exec.Command(checker).Output())(t); string(said) != "committed" {
		t.Fatalf("the checker says %q, want the committed gate's %q", said, "committed")
	}
	if held := testcheck.Must(os.ReadFile(filepath.Join(root, "cmd/gate/main.go")))(t); !strings.Contains(string(held), "uncommitted") {
		t.Fatal("building the checker changed the checkout's uncommitted edit")
	}
}
