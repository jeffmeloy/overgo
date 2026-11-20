package latentimage

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// workflowSectionTime is one section's figures in a run of the workflow: its wall from request to picture, its
// sampling, its passes, the bytes it placed on the card, and its recording's wall.
type workflowSectionTime struct {
	Section      string `json:"section"`
	Milliseconds int64  `json:"ms"`
	SamplingMs   int64  `json:"sampling_ms"`
	Passes       int    `json:"passes"`
	PlacedBytes  uint64 `json:"placed_bytes"`
	ComfyMs      int64  `json:"comfy_ms"`
}

// workflowRunTime is one run of the six sections, the first beside ComfyUI's pass-1 and the second beside its pass-2.
type workflowRunTime struct {
	Recorded     string                `json:"recorded"`
	Milliseconds int64                 `json:"ms"`
	ComfyMs      int64                 `json:"comfy_ms"`
	Sections     []workflowSectionTime `json:"sections"`
}

// workflowTiming is one timing of the workflow as its record keeps it: the commit it timed, the paths the tree held
// changed from it, when, and its two runs.
type workflowTiming struct {
	Commit  string            `json:"commit"`
	Changed []string          `json:"changed"`
	At      string            `json:"at"`
	Runs    []workflowRunTime `json:"runs"`
}

// workflowSource is the code a timing of the workflow runs, read as the timing starts, beside the test binary built
// from the tree just before: the commit the repository's checkout holds, and the paths its tree holds changed from it
// but the record's own (SQA-441).
type workflowSource struct {
	commit  string
	changed []string
}

// readWorkflowSource reads the source of the checkout at repository, the record's path, relative to it, left out of
// its changed paths.
func readWorkflowSource(repository, record string) (workflowSource, error) {
	git := func(arguments ...string) (string, error) {
		command := exec.Command("git", arguments...)
		command.Dir = repository
		output, err := command.Output()
		return string(output), err
	}
	commit, err := git("rev-parse", "HEAD")
	if err != nil {
		return workflowSource{}, err
	}
	status, err := git("status", "--porcelain", "--untracked-files=no")
	if err != nil {
		return workflowSource{}, err
	}
	source := workflowSource{commit: strings.TrimSpace(commit), changed: []string{}}
	for line := range strings.Lines(status) {
		if changed := strings.TrimSpace(line[min(3, len(line)):]); changed != "" && changed != filepath.ToSlash(record) {
			source.changed = append(source.changed, changed)
		}
	}
	return source, nil
}

// appendWorkflowTiming appends timing to the record at path in repository, one JSON line, under source, the commit and
// changed paths read as the timing started; a checkout whose commit has moved since appends nothing and is refused,
// naming both: another landing advanced it while the workflow ran, and its timing had been recorded under the later
// commit, whose code it did not run, where that commit's clock check accepted it (SQA-441).
func appendWorkflowTiming(repository, path string, source workflowSource, timing workflowTiming) error {
	now, err := readWorkflowSource(repository, path)
	if err != nil {
		return err
	}
	if now.commit != source.commit {
		return fmt.Errorf("the checkout moved from %s to %s while the workflow ran; its timing is of %s's code, and none is appended",
			source.commit, now.commit, source.commit)
	}
	timing.Commit, timing.Changed = source.commit, source.changed
	line, err := json.Marshal(timing)
	if err != nil {
		return err
	}
	record, err := os.OpenFile(filepath.Join(repository, filepath.FromSlash(path)), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	_, err = record.Write(append(line, '\n'))
	return errors.Join(err, record.Close())
}

// TestAWorkflowTimingRefusesASourceThatMovedWhileItRan holds a timing of the workflow to the code it ran (clock-identity,
// SQA-441): over a scratch repository, a timing appends the commit and the changed paths read as it started, though a
// path changes before it appends, and a timing whose checkout took another commit while it ran appends nothing and is
// refused, naming both commits. The record had read the commit and the changed paths after both runs, so a landing
// during a run credited its timing to the later commit, whose clock check accepted it.
func TestAWorkflowTimingRefusesASourceThatMovedWhileItRan(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	git := func(arguments ...string) string {
		t.Helper()
		command := exec.CommandContext(t.Context(), "git", append([]string{"-C", root, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "-c", "core.hooksPath="}, arguments...)...)
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v %s", arguments, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	const record = "docs/record.jsonl"
	git("init", "-q")
	testutil.WriteTextFile(t, root, "code.go", "package code\n", 0o600)
	testutil.WriteTextFile(t, root, "notes.md", "notes\n", 0o600)
	testutil.WriteTextFile(t, root, record, "", 0o600)
	git("add", "-A")
	git("commit", "-q", "-m", "first")
	first := git("rev-parse", "HEAD")
	testutil.WriteTextFile(t, root, "code.go", "package code // as the binary was built\n", 0o600)
	source := testcheck.Must(readWorkflowSource(root, record))(t)
	testutil.WriteTextFile(t, root, "notes.md", "notes changed while the workflow ran\n", 0o600)
	testcheck.NoError(t, appendWorkflowTiming(root, record, source, workflowTiming{At: "first"}))
	var appended workflowTiming
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(root, record)))(t), &appended))
	if appended.Commit != first || !slices.Equal(appended.Changed, []string{"code.go"}) {
		t.Errorf("a timing appended commit %s with %v changed; want %s with code.go, as it started", appended.Commit, appended.Changed, first)
	}
	git("commit", "-q", "-a", "-m", "second")
	second := git("rev-parse", "HEAD")
	written := testcheck.Must(os.ReadFile(filepath.Join(root, record)))(t)
	err := appendWorkflowTiming(root, record, source, workflowTiming{At: "moved"})
	if err == nil || !strings.Contains(err.Error(), first) || !strings.Contains(err.Error(), second) {
		t.Errorf("a timing whose checkout moved from %s to %s while it ran was appended with %v; want it refused, naming both", first, second, err)
	}
	if after := testcheck.Must(os.ReadFile(filepath.Join(root, record)))(t); !slices.Equal(after, written) {
		t.Error("a refused timing changed the record")
	}
}
