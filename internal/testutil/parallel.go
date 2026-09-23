package testutil

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// processGlobal marks a test body that changes state the whole test process
// shares, which Go allows only in a serial test.
var processGlobal = []string{"Setenv", "Chdir", "os.Stdout =", "os.Stderr ="}

// RequireParallel runs the test calling it in parallel and fails it for every
// top-level test in the calling package's test files that does not begin by
// running in parallel, among those whose body contains every marker given
// (all tests when none is), leaving out tests that change process-wide
// state. Go runs parallel tests only after every serial one, so a serial test
// adds its whole time to its suite's wall. The files are scanned as text,
// which a package that must not parse Go source can use.
func RequireParallel(t *testing.T, markers ...string) {
	t.Helper()
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
			header, body, found := strings.Cut(function, "\n")
			if !found || !strings.HasPrefix(header, "Test") || !strings.HasSuffix(header, "(t *testing.T) {") {
				continue
			}
			body, _, _ = strings.Cut(body, "\n}\n")
			absent := func(marker string) bool { return !strings.Contains(body, marker) }
			present := func(marker string) bool { return strings.Contains(body, marker) }
			if slices.ContainsFunc(markers, absent) || slices.ContainsFunc(processGlobal, present) || parallelFirst(body) {
				continue
			}
			name, _, _ := strings.Cut(header, "(")
			t.Errorf("%s: %s runs serially; call t.Parallel() first", file, name)
		}
	}
}

// parallelFirst reports a test body that begins by running in parallel,
// itself or through RequireParallel, which does so for the test calling it.
func parallelFirst(body string) bool {
	first := strings.TrimSpace(body)
	return strings.HasPrefix(first, "t.Parallel()") || strings.HasPrefix(first, "testutil.RequireParallel(t")
}
