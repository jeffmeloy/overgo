package plan

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestPlanSuiteWallBudget holds every top-level test in this package to
// running in parallel unless it sets the environment or the working
// directory, which Go allows only serially. The suite is selected by every
// control-plane landing; with 86 of its 88 tests serial it measured 74.6 s,
// and with them parallel 41 to 51 s over four runs (2026-09-23). A new serial
// test would quietly bring the wall back.
func TestPlanSuiteWallBudget(t *testing.T) {
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
			if strings.Contains(body, "Setenv") || strings.Contains(body, "Chdir") {
				continue
			}
			if !strings.HasPrefix(strings.TrimSpace(body), "t.Parallel()") {
				t.Errorf("%s: %s runs serially; call t.Parallel() first unless it sets the environment or the working directory", file, strings.TrimSuffix(header, " {"))
			}
		}
	}
}
