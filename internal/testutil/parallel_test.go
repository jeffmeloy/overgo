package testutil

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestSuitesRunInParallel holds the suites every control-plane landing pays
// for to running their tests in parallel: Go runs parallel tests only after
// every serial one, so each serial test adds its whole time to its suite's
// wall. Measured 2026-09-23, serial to parallel: internal/plan 74.6 to about
// 41 s, cmd/plan 56 to about 26 s, internal/evaluation 50.3 to 11.7 s,
// internal/overgodb 66.9 to about 15 s, internal/repoanalysis 64.3 to about
// 26 s, internal/server 50.9 to about 17 s. In the gate suite only the users
// of the shared live-checkout fixture are held, since its one-time build
// belongs in the parallel phase.
func TestSuitesRunInParallel(t *testing.T) {
	t.Parallel()
	for _, suite := range []struct {
		directory string
		markers   []string
	}{
		{directory: "../gate", markers: []string{"liveRepositoryFixture("}},
		{directory: "../plan"},
		{directory: "../../cmd/plan"},
		{directory: "../evaluation"},
		{directory: "../overgodb"},
		{directory: "../repoanalysis"},
		{directory: "../server"},
	} {
		t.Run(suite.directory, func(t *testing.T) {
			t.Parallel()
			requireParallel(t, suite.directory, suite.markers...)
		})
	}
}

// processGlobal marks a test body that changes state the whole test process
// shares, which Go allows only in a serial test.
var processGlobal = []string{"Setenv", "Chdir", "os.Stdout =", "os.Stderr ="}

// serialMark opens a test that stays serial for a reason the body does not
// show, such as a helper that sets the process environment.
const serialMark = "// Serial:"

// requireParallel fails the test for every top-level test in directory's test
// files that does not begin by running in parallel, among those whose body
// contains every marker given (all tests when none is). A test that changes
// process-wide state, or opens with the serial mark and its reason, may stay
// serial. The files are scanned as text, so no suite takes on a Go parser to
// be checked.
func requireParallel(t *testing.T, directory string, markers ...string) {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(directory, "*_test.go"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no test files in %s: %v", directory, err)
	}
	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		for _, function := range strings.Split(string(data), "\nfunc ")[1:] {
			header, body, found := strings.Cut(function, "\n")
			if !found || !strings.HasPrefix(header, "Test") || !strings.HasSuffix(header, "(t *testing.T) {") {
				continue
			}
			body, _, _ = strings.Cut(body, "\n}\n")
			first := strings.TrimSpace(body)
			absent := func(marker string) bool { return !strings.Contains(body, marker) }
			present := func(marker string) bool { return strings.Contains(body, marker) }
			if slices.ContainsFunc(markers, absent) || slices.ContainsFunc(processGlobal, present) ||
				strings.HasPrefix(first, "t.Parallel()") || strings.HasPrefix(first, serialMark) {
				continue
			}
			name, _, _ := strings.Cut(header, "(")
			t.Errorf("%s: %s runs serially; call t.Parallel() first, or open with %q and the reason", filepath.ToSlash(file), name, serialMark)
		}
	}
}
