package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestACheckerThatCannotBeBuiltRefusesWithWhatStoppedIt holds a landing's checker to SQA-397: in a fixture repository
// whose checkout holds no data-root file, which building cmd/gate never reads, the checker builds from the trunk's
// commit; and in a directory that is no repository, where git cannot read the trunk's commit, building it refuses with
// git's error. It had copied the replay's data roots in first, so a checkout without them built no checker, and a
// setup error came back as an empty refusal, which Gate took for a checker that ran.
func TestACheckerThatCannotBeBuiltRefusesWithWhatStoppedIt(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	for path, content := range map[string]string{"go.mod": "module fixture\n\ngo 1.26\n", "cmd/gate/main.go": "package main\n\nfunc main() {}\n"} {
		testcheck.NoError(t, os.MkdirAll(filepath.Dir(filepath.Join(root, path)), 0o755))
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, path), []byte(content), 0o644))
	}
	for _, arguments := range [][]string{{"init", "-q"}, {"add", "."}, {"-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "fixture"}} {
		testcheck.NoError(t, gitWrite(root, arguments...))
	}
	build := func(arguments ...string) (string, error) {
		output, err := exec.Command(arguments[0], arguments[1:]...).CombinedOutput()
		return string(output), err
	}
	checker := filepath.Join(t.TempDir(), "gate.exe")
	if refusal, err := buildChecker(root, checker, build); err != nil {
		t.Fatalf("in a checkout with no data-root file the checker was not built: %v: %s", err, refusal)
	}
	refusal, err := buildChecker(t.TempDir(), checker, build)
	if err == nil || refusal == "" || !strings.Contains(refusal, err.Error()) {
		t.Fatalf("a checker git could not set up refused with %q (%v), want the error that stopped it", refusal, err)
	}
}
