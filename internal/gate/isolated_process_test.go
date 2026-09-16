package gate

import (
	"os"
	"os/exec"
	"regexp"
	"strings"
	"testing"
)

// isolatedProcessEnvironment marks a test binary that runs one test's body
// on behalf of a parallel parent.
const isolatedProcessEnvironment = "OVERGO_GATE_TEST_PROCESS"

// isolatedProcess runs the calling test's body in a child process of its
// own when the body sets the process environment or installs a package
// hook, so the parent joins the parallel phase and holds the child's
// verdict; in the child it returns false and the body runs. A skipped child
// skips the parent with the same reason, so the gate sees the classified
// reason, never a vacuous pass.
func isolatedProcess(t *testing.T) bool {
	t.Helper()
	if os.Getenv(isolatedProcessEnvironment) != "" {
		return false
	}
	t.Parallel()
	arguments := []string{"-test.run=^" + regexp.QuoteMeta(t.Name()) + "$", "-test.count=1", "-test.v"}
	if testing.Short() {
		arguments = append(arguments, "-test.short")
	}
	command := exec.CommandContext(t.Context(), os.Args[0], arguments...)
	command.Env = append(os.Environ(), isolatedProcessEnvironment+"=1")
	output, err := command.CombinedOutput()
	text := string(output)
	if err != nil {
		t.Fatalf("%s in its own process: %v\n%s", t.Name(), err, text)
	}
	if !strings.Contains(text, "--- PASS: "+t.Name()) {
		marker := "--- SKIP: " + t.Name()
		index := strings.Index(text, marker)
		if index < 0 {
			t.Fatalf("%s in its own process left no verdict:\n%s", t.Name(), text)
		}
		rest := text[index+len(marker):]
		reason := ""
		for line := range strings.SplitSeq(rest, "\n") {
			if trimmed := strings.TrimSpace(line); trimmed != "" && !strings.HasPrefix(trimmed, "(") && !strings.HasPrefix(trimmed, "PASS") && !strings.HasPrefix(trimmed, "ok") {
				reason = trimmed
				break
			}
		}
		if colon := strings.Index(reason, ": "); colon >= 0 && strings.Contains(reason[:colon], "_test.go:") {
			reason = reason[colon+2:]
		}
		t.Skip(reason)
	}
	for line := range strings.SplitSeq(text, "\n") {
		if strings.Contains(line, "_test.go:") && !strings.HasPrefix(strings.TrimSpace(line), "---") {
			t.Log(strings.TrimSpace(line))
		}
	}
	return true
}
