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
// sampling, its passes, the bytes it placed on the card, its recording's wall, the process's processor time and the
// bytes it read from files over the section, and its parts (moss-first-run/quiet).
type workflowSectionTime struct {
	Section      string         `json:"section"`
	Milliseconds int64          `json:"ms"`
	SamplingMs   int64          `json:"sampling_ms"`
	Passes       int            `json:"passes"`
	PlacedBytes  uint64         `json:"placed_bytes"`
	ComfyMs      int64          `json:"comfy_ms"`
	CPUMs        int64          `json:"cpu_ms,omitzero"`
	ReadBytes    uint64         `json:"read_bytes,omitzero"`
	Parts        []workflowPart `json:"parts,omitempty"`
}

// workflowPart is one part of a section's wall, in the order the parts first ended: the session's load or reset, the
// prompt's preparation, the sampling, the decode, and the parts the code names inside them (the decoder compiled and its
// static inputs bound, the picture decoded on the card, the picture finished). Its wall, processor time and bytes read
// from files are its own, what a part inside it spent excluded, all read on the section's one ledger
// (processmeasure.Stages, served-parts), so a first run's time is split where it goes, each file read and each program
// built in the part that did it. Before served-parts a timing's parts were the four outer parts, each with what its inner
// parts spent, and the decode's own parts named "decode: " and a part, their walls alone read on another clock (SQA-471).
type workflowPart struct {
	Part         string `json:"part"`
	Milliseconds int64  `json:"ms"`
	CPUMs        int64  `json:"cpu_ms"`
	ReadBytes    uint64 `json:"read_bytes"`
}

// workflowRunTime is one run of the six sections, the first beside ComfyUI's pass-1 and the second beside its pass-2.
type workflowRunTime struct {
	Recorded     string                `json:"recorded"`
	Milliseconds int64                 `json:"ms"`
	ComfyMs      int64                 `json:"comfy_ms"`
	Sections     []workflowSectionTime `json:"sections"`
}

// workflowTiming is one timing of the workflow as its record keeps it: the commit it timed, the paths the tree held
// changed from it, whether it ran the commit's code alone (committed: every changed path under docs/, which builds
// nothing), when, and its two runs. A clock step requires committed, so a timing beside changed code stays in the
// record and satisfies none (clock-committed, SQA-441).
type workflowTiming struct {
	Commit    string            `json:"commit"`
	Changed   []string          `json:"changed"`
	Committed bool              `json:"committed"`
	At        string            `json:"at"`
	Runs      []workflowRunTime `json:"runs"`
}

// workflowSource is the code a timing of the workflow runs, read as the timing starts, beside the test binary built
// from the tree just before: the commit the repository's checkout holds, and the paths its tree holds changed from it
// but the record's own, untracked paths among them, since an untracked source file builds into the binary as a tracked
// one does (SQA-441).
type workflowSource struct {
	commit  string
	changed []string
}

// committed says the source is its commit's code alone: each changed path lies under docs/ (the plan, the findings
// register, the records), which a session edits beside a run and which builds nothing.
func (s workflowSource) committed() bool {
	return !slices.ContainsFunc(s.changed, func(path string) bool { return !strings.HasPrefix(path, "docs/") })
}

// readWorkflowSource reads the source of the checkout at repository, the record's path, relative to it, left out of
// its changed paths. A record path that is not relative to the repository and inside it is refused here, as the timing
// starts, before it holds the machine or draws a section (SQA-460): an absolute path had been taken and joined under
// the repository only as the timing was appended, after both runs, so twelve sections drawn saved nothing.
func readWorkflowSource(repository, record string) (workflowSource, error) {
	if !filepath.IsLocal(filepath.FromSlash(record)) {
		return workflowSource{}, fmt.Errorf("the workflow record %s is not a path relative to the repository and inside it (as docs/verification/moss-workflow-time.jsonl is)", record)
	}
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
	status, err := git("status", "--porcelain", "--untracked-files=all")
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
// changed paths read as the timing started, and whether they were its commit's code alone; a checkout whose commit has
// moved since appends nothing and is refused, naming both: another landing advanced it while the workflow ran, and its
// timing had been recorded under the later commit, whose code it did not run, where that commit's clock check
// accepted it (SQA-441).
func appendWorkflowTiming(repository, path string, source workflowSource, timing workflowTiming) error {
	now, err := readWorkflowSource(repository, path)
	if err != nil {
		return err
	}
	if now.commit != source.commit {
		return fmt.Errorf("the checkout moved from %s to %s while the workflow ran; its timing is of %s's code, and none is appended",
			source.commit, now.commit, source.commit)
	}
	timing.Commit, timing.Changed, timing.Committed = source.commit, source.changed, source.committed()
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

// scratchRepository is a git repository in a test's directory holding files, its paths and contents, committed once,
// and a function that runs git in it and gives its output.
func scratchRepository(t *testing.T, files map[string]string) (string, func(arguments ...string) string) {
	t.Helper()
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
	git("init", "-q")
	for path, content := range files {
		testutil.WriteTextFile(t, root, path, content, 0o600)
	}
	git("add", "-A")
	git("commit", "-q", "-m", "first")
	return root, git
}

// lastWorkflowTiming is the last timing the record at path in root holds.
func lastWorkflowTiming(t *testing.T, root, path string) workflowTiming {
	t.Helper()
	lines := strings.Split(strings.TrimSpace(string(testcheck.Must(os.ReadFile(filepath.Join(root, path)))(t))), "\n")
	var timing workflowTiming
	testcheck.NoError(t, json.Unmarshal([]byte(lines[len(lines)-1]), &timing))
	return timing
}

// TestAWorkflowTimingRefusesASourceThatMovedWhileItRan holds a timing of the workflow to the code it ran (clock-identity,
// SQA-441): over a scratch repository, a timing appends the commit and the changed paths read as it started, though a
// path changes before it appends, and a timing whose checkout took another commit while it ran appends nothing and is
// refused, naming both commits. The record had read the commit and the changed paths after both runs, so a landing
// during a run credited its timing to the later commit, whose clock check accepted it.
func TestAWorkflowTimingRefusesASourceThatMovedWhileItRan(t *testing.T) {
	t.Parallel()
	const record = "docs/record.jsonl"
	root, git := scratchRepository(t, map[string]string{"code.go": "package code\n", "notes.md": "notes\n", record: ""})
	first := git("rev-parse", "HEAD")
	testutil.WriteTextFile(t, root, "code.go", "package code // as the binary was built\n", 0o600)
	source := testcheck.Must(readWorkflowSource(root, record))(t)
	testutil.WriteTextFile(t, root, "notes.md", "notes changed while the workflow ran\n", 0o600)
	testcheck.NoError(t, appendWorkflowTiming(root, record, source, workflowTiming{At: "first"}))
	if appended := lastWorkflowTiming(t, root, record); appended.Commit != first || !slices.Equal(appended.Changed, []string{"code.go"}) {
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

// TestTheWorkflowClockRefusesARecordItCannotWrite holds the workflow clock to refusing, as its timing starts, a record
// it cannot write (SQA-460): over a scratch repository, the source check refuses an absolute record path, with either
// separator, and one leaving the repository, each named, and reads a path inside it. The timing reads its source
// before it holds the machine or draws a section (TestTheUltraWorkflowIsTimedInItsOrder). An absolute path had been
// taken there and joined under the repository only when the timing was appended, after both runs, so a timing drew
// twelve sections (81 s measured) and saved nothing.
func TestTheWorkflowClockRefusesARecordItCannotWrite(t *testing.T) {
	t.Parallel()
	const record = "docs/record.jsonl"
	root, _ := scratchRepository(t, map[string]string{record: ""})
	absolute := filepath.Join(root, filepath.FromSlash(record))
	for _, path := range []string{absolute, filepath.ToSlash(absolute), "../record.jsonl", "docs/../../record.jsonl"} {
		if _, err := readWorkflowSource(root, path); err == nil || !strings.Contains(err.Error(), path) {
			t.Errorf("the source check took the record %s (%v); want it refused, naming it", path, err)
		}
	}
	testcheck.Must(readWorkflowSource(root, record))(t)
}

// TestAWorkflowTimingSaysWhetherItRanItsCommitsCode holds the record to telling a timing of its commit's code from one
// beside changed code (clock-committed, SQA-441): over a scratch repository, a timing whose tree holds only a changed
// path under docs/ (the register a session edits beside a run) ran its commit's code, and one whose tree holds changed
// code did not, though both name the commit, so a clock check that requires it accepts the first alone. The record
// had named a timing beside changed code by its commit alone, the change among its changed paths, and the clock checks
// read the commit.
func TestAWorkflowTimingSaysWhetherItRanItsCommitsCode(t *testing.T) {
	t.Parallel()
	const record = "docs/verification/record.jsonl"
	root, git := scratchRepository(t, map[string]string{"code.go": "package code\n", "docs/register.json": "{}\n", record: ""})
	for _, beside := range []struct {
		path               string
		committed, tracked bool
	}{{"docs/register.json", true, true}, {"code.go", false, true}, {"local_perf.go", false, false}} {
		testutil.WriteTextFile(t, root, beside.path, "package code // changed beside the run\n", 0o600)
		testcheck.NoError(t, appendWorkflowTiming(root, record, testcheck.Must(readWorkflowSource(root, record))(t), workflowTiming{At: beside.path}))
		if timing := lastWorkflowTiming(t, root, record); timing.Committed != beside.committed || !slices.Contains(timing.Changed, beside.path) {
			t.Errorf("a timing beside a changed %s (tracked %t) says it ran its commit's code %t, %v changed; want %t",
				beside.path, beside.tracked, timing.Committed, timing.Changed, beside.committed)
		}
		// Each case alone: the tree goes back to the commit's code, the record excepted.
		if beside.tracked {
			git("checkout", "--", beside.path)
		} else {
			testcheck.NoError(t, os.Remove(filepath.Join(root, beside.path)))
		}
	}
}
