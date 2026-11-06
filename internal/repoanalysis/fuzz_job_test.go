package repoanalysis

import (
	"errors"
	"go/scanner"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// fuzzWorkflow is the scheduled job that fuzzes the tree's targets, fuzzScript the script it runs and stallCheck the
// script nothing ran.
const (
	fuzzWorkflow = ".github/workflows/fuzz.yml"
	fuzzScript   = "scripts/fuzz-smoke.sh"
	stallCheck   = "scripts/stall_check.sh"
)

// TestEveryFuzzTargetRunsInAScheduledJob holds the tree's fuzz targets to a job that fuzzes them: fuzz.yml runs on a
// schedule and runs scripts/fuzz-smoke.sh, which lists the targets from the tree with go test -list and names none by
// hand, so a target added is fuzzed and one removed leaves no name behind; and scripts/stall_check.sh, which nothing
// ran, is gone.
func TestEveryFuzzTargetRunsInAScheduledJob(t *testing.T) {
	t.Parallel()
	var targets []string
	for _, tree := range []string{"cmd", "internal"} {
		testcheck.NoError(t, filepath.WalkDir(filepath.Join(checkoutRoot, tree), func(file string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.HasSuffix(file, "_test.go") {
				return err
			}
			source := testcheck.Must(os.ReadFile(file))(t)
			var words scanner.Scanner
			words.Init(token.NewFileSet().AddFile(file, -1, len(source)), source, nil, 0)
			function := false
			for _, kind, literal := words.Scan(); kind != token.EOF; _, kind, literal = words.Scan() {
				if function && kind == token.IDENT && strings.HasPrefix(literal, "Fuzz") {
					targets = append(targets, literal)
				}
				function = kind == token.FUNC
			}
			return nil
		}))
	}
	if len(targets) == 0 {
		t.Fatal("the tree holds no fuzz target")
	}
	workflow := string(testcheck.Must(os.ReadFile(filepath.Join(checkoutRoot, filepath.FromSlash(fuzzWorkflow))))(t))
	if !strings.Contains(workflow, "schedule:") || !strings.Contains(workflow, "cron:") || !strings.Contains(workflow, fuzzScript) {
		t.Errorf("%s runs %s on no schedule", fuzzWorkflow, fuzzScript)
	}
	script := string(testcheck.Must(os.ReadFile(filepath.Join(checkoutRoot, filepath.FromSlash(fuzzScript))))(t))
	if !strings.Contains(script, "go test -list") {
		t.Errorf("%s lists no fuzz target from the tree", fuzzScript)
	}
	for _, target := range targets {
		if strings.Contains(script, target) {
			t.Errorf("%s names %s by hand; list the targets from the tree", fuzzScript, target)
		}
	}
	if _, err := os.Stat(filepath.Join(checkoutRoot, filepath.FromSlash(stallCheck))); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("%s stands, which nothing runs", stallCheck)
	}
}
