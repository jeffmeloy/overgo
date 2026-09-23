package gate

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestGateSuiteSerialFloorBudget holds every top-level test that uses the
// shared live-checkout fixture to running in parallel. Go runs the parallel
// tests only after every serial one has finished, and the fixture's one-time
// build -- a git worktree and the package input graph, about 50 s -- is paid
// by whichever test asks for it first. One serial user put that build in the
// serial phase, where nothing overlapped it: measured 2026-09-23 at 50.8 s of
// the suite's 54.1 s serial floor in a 177.6 s run. Text is scanned rather
// than parsed, since gate tests do not import the Go source packages.
func TestGateSuiteSerialFloorBudget(t *testing.T) {
	t.Parallel()
	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for _, function := range strings.Split(string(data), "\nfunc ")[1:] {
			name, body, found := strings.Cut(function, "\n")
			if !found || !strings.HasPrefix(name, "Test") || !strings.Contains(name, "(t *testing.T)") {
				continue
			}
			body, _, _ = strings.Cut(body, "\n}\n")
			if strings.Contains(body, "liveRepositoryFixture(") && !strings.HasPrefix(strings.TrimSpace(body), "t.Parallel()") {
				t.Errorf("%s: %s builds or reads the live-checkout fixture from the serial phase; call t.Parallel() first", file, strings.TrimSuffix(name, " {"))
			}
		}
	}
}
