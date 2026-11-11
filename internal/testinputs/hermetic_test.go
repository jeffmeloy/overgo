package testinputs

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// liveState are the checkout's files and folders that landings and the driver change: the plan, the doctrine and the
// README, the root and docs folders a listing reads every entry of, and the process state.
var liveState = []string{"docs/plan.json", "skill.md", "README.md", ".", "docs", ".overgo-runtime"}

// TestTheCommitAndDriverTestsReadNoLiveState holds internal/landing's and cmd/loop's tests to reading nothing a landing
// changes (fail-early/hermetic-tests): go test keeps a package's result until its code, or a file inside the module its
// tests read, changes, and these two, 93 s and 149 s, ran again in 36 and 30 of 37 landings since 2026-11-09 though
// their code changed in 2 and 11, because their tests read the live plan, skill.md, the README and the process state.
// Each package's tests run here with Go's input log, and no read of those, resolved against the folder the test had
// changed into, may stand among their inputs.
func TestTheCommitAndDriverTestsReadNoLiveState(t *testing.T) {
	root := testcheck.Must(filepath.Abs(filepath.Join("..", "..")))(t)
	for _, pkg := range []string{"internal/landing", "cmd/loop"} {
		// Listing the package's folder makes its files this test's inputs, so go test runs it again exactly when the
		// code of either package changes, and keeps its result otherwise: 74 s on every landing would undo what it holds.
		testcheck.Must(os.ReadDir(filepath.Join(root, filepath.FromSlash(pkg))))(t)
		binary := filepath.Join(t.TempDir(), "package.test.exe")
		build := exec.CommandContext(t.Context(), "go", "test", "-c", "-o", binary, "./"+pkg)
		build.Dir = root
		if output, err := build.CombinedOutput(); err != nil {
			t.Fatalf("%s's tests do not build: %v\n%s", pkg, err, output)
		}
		log := filepath.Join(t.TempDir(), "inputs.log")
		run := exec.CommandContext(t.Context(), binary, "-test.count=1", "-test.testlogfile="+log)
		run.Dir = filepath.Join(root, filepath.FromSlash(pkg))
		if output, err := run.CombinedOutput(); err != nil {
			t.Fatalf("%s's tests fail: %v\n%s", pkg, err, tail(output))
		}
		if read := liveReads(t, root, run.Dir, log); len(read) != 0 {
			t.Errorf("%s's tests read what landings change, so they run again at every landing: %s", pkg, strings.Join(read, ", "))
		}
	}
}

// liveReads are the reads of liveState in a test input log, each resolved against the folder its test had changed into.
func liveReads(t *testing.T, root, start, log string) []string {
	t.Helper()
	file := testcheck.Must(os.Open(log))(t)
	defer file.Close()
	cwd, found := start, map[string]bool{}
	lines := bufio.NewScanner(file)
	lines.Buffer(nil, 1<<20)
	for lines.Scan() {
		kind, path, ok := strings.Cut(lines.Text(), " ")
		if !ok {
			continue
		}
		if kind == "chdir" {
			cwd = path
			continue
		}
		if kind != "open" && kind != "stat" {
			continue
		}
		if !filepath.IsAbs(path) {
			path = filepath.Join(cwd, path)
		}
		relative, err := filepath.Rel(root, filepath.Clean(path))
		if err != nil || strings.HasPrefix(relative, "..") {
			continue
		}
		relative = filepath.ToSlash(relative)
		directory := relative == "." || relative == "docs"
		if directory && kind != "open" {
			continue
		}
		if slices.Contains(liveState, relative) || strings.HasPrefix(relative, ".overgo-runtime/") {
			found[kind+" "+relative] = true
		}
	}
	testcheck.NoError(t, lines.Err())
	return slices.Sorted(func(yield func(string) bool) {
		for read := range found {
			if !yield(read) {
				return
			}
		}
	})
}

// tail is the end of a command's output, where a failing test says why.
func tail(output []byte) string {
	const shown = 2000
	if len(output) > shown {
		return string(output[len(output)-shown:])
	}
	return string(output)
}
