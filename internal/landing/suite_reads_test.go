//go:build wholetree

// The check of the suites' reads runs them whole, so it is built apart and is no whole-tree test the invariants stage
// runs: its file is not named with wholeTreeSuffix.

package landing

import (
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// suiteDependencies are the files of the module outside their own package that a test of the commit tool's or the
// driver's suite reads because its result depends on them, each beside the tests that read it.
var suiteDependencies = []string{
	// cmd/loop TestChecksRunTheLongSequences: the long sequences' test, which the cycle runs.
	"internal/overgodb/long_sequences_test.go",
	// internal/landing TestLanesMeasurementLine: the device lane's admission, whose line it reads.
	"cmd/device-lane/admission.go",
	// internal/landing TestInvariantPackagesExist: the invariant packages.
	"internal/commanddoc", "internal/plan", "internal/repoanalysis",
	// internal/landing TestATestOnlyChangeRunsNoScorecardSlice: the benchmark command.
	"cmd/benchmark",
	// internal/landing's hook tests: the module and the hook a trunk carries.
	"go.mod", "internal/go.mod", "scripts/githooks/reference-transaction",
}

// TestTheLoopsSuitesReadOnlyTheirOwnPackages holds the commit tool's and the driver's suites, as the test stage builds
// them, to reading no file of the module outside their own package's folder beyond those one of their tests depends on
// (suiteDependencies). go test counts what a suite reads among its inputs: on 2026-11-11 the driver's identity carry
// copied the tree and the commit tool's reach invariants ran the lane discovery over it, so any change ran both suites
// whole again, cmd/loop for 226 s at e9b82cdf6 on a change it does not import (test-reuse). It builds each suite's
// tests, runs them whole with a test log, and names each path read outside.
func TestTheLoopsSuitesReadOnlyTheirOwnPackages(t *testing.T) {
	repo := testcheck.Must(filepath.Abs(filepath.Join("..", "..")))(t)
	for _, suite := range []string{"./internal/landing", "./cmd/loop"} {
		t.Run(suite, func(t *testing.T) {
			t.Parallel()
			directory := filepath.Join(repo, filepath.FromSlash(suite))
			binary := filepath.Join(t.TempDir(), "suite.test.exe")
			build := exec.CommandContext(t.Context(), "go", "test", "-c", "-o", binary, suite)
			build.Dir = repo
			if output, err := build.CombinedOutput(); err != nil {
				t.Fatalf("building %s's tests: %v\n%s", suite, err, output)
			}
			log := filepath.Join(t.TempDir(), "inputs.log")
			run := exec.CommandContext(t.Context(), binary, "-test.count=1", "-test.testlogfile="+log)
			run.Dir = directory
			if output, err := run.CombinedOutput(); err != nil {
				t.Fatalf("%s's tests failed: %v\n%s", suite, err, output)
			}
			for _, path := range outsideReads(t, repo, directory, log) {
				t.Errorf("%s's tests read %s, outside the package: go test runs the suite again whenever it changes", suite, path)
			}
		})
	}
}

// outsideReads are the files of the module at repo outside directory that the test log at log names as opened or
// stated, relative to repo with forward slashes, less suiteDependencies. A relative name is read from the folder the
// tests last changed to, as go test reads it.
func outsideReads(t *testing.T, repo, directory, log string) []string {
	t.Helper()
	data := testcheck.Must(os.ReadFile(log))(t)
	folder, outside := directory, map[string]string{}
	for line := range strings.Lines(string(data)) {
		operation, name, named := strings.Cut(strings.TrimRight(line, "\r\n"), " ")
		if !named {
			continue
		}
		if !filepath.IsAbs(name) {
			name = filepath.Join(folder, name)
		}
		switch operation {
		case "chdir":
			folder = name
			continue
		case "open", "stat":
		default:
			continue
		}
		relative, err := filepath.Rel(repo, filepath.Clean(name))
		if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
			continue
		}
		if own, err := filepath.Rel(directory, filepath.Clean(name)); err == nil && own != ".." && !strings.HasPrefix(own, ".."+string(filepath.Separator)) {
			continue
		}
		if slashed := filepath.ToSlash(relative); !slices.Contains(suiteDependencies, slashed) && outside[slashed] == "" {
			outside[slashed] = strings.TrimSpace(line)
		}
	}
	read := make([]string, 0, len(outside))
	for _, path := range slices.Sorted(maps.Keys(outside)) {
		read = append(read, path+" (\""+outside[path]+"\")")
	}
	return read
}
