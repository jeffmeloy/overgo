package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/plan"
	"overgo/internal/processlock"
)

// TestCloseoutRefusesFinishedLeftovers builds, under a temporary directory, a repository with a
// finished entry on every line of the table whose action is delete: a helper copy, a merged
// branch, an experiment's output, a temporary build folder and a scratch folder nothing holds, its
// name the row of no plan. Beside them lie what must not refuse: a helper copy and a branch that
// are held, an output that is unknown, a finished executable on a line that only reports, a
// directory no line names, and a second scratch folder nothing holds, which holds a link, so the
// listing cannot call it finished.
//
// The close-out mode of the listing command fails while a finished entry of a deleting line
// remains. It prints the listing without sizes, then each refusing entry once more so they are the
// last entry lines, then one line of counts, and writes the same text as a receipt under a name of
// its own, so the measured listing of the same commit keeps its receipt. Its error names no entry:
// it ends with the total, the receipt and the command that removes them, and stays shorter than
// what a commit's checks keep of a failed check however many entries refuse. It removes nothing.
// With every refusing entry cleared it passes and still counts the unknown and the undeclared;
// with one merged branch made again it refuses for that branch alone; and a source that fails is
// one more unknown entry, never a refusal.
func TestCloseoutRefusesFinishedLeftovers(t *testing.T) {
	t.Parallel()
	// Git names a worktree by its resolved path, so the fixture's root is resolved the same way.
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	at := func(slash string) string { return filepath.Join(root, filepath.FromSlash(slash)) }
	folder := func(slash string) string {
		t.Helper()
		if err := os.MkdirAll(at(slash), 0o755); err != nil {
			t.Fatal(err)
		}
		return at(slash)
	}
	write := func(slash, content string) {
		t.Helper()
		folder(filepath.ToSlash(filepath.Dir(filepath.FromSlash(slash))))
		if err := os.WriteFile(at(slash), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// The trunk's checkout: a record that names one experiment, and a plan of one open row. The two
	// scratch folders below are named for rows no plan holds, which is all their line asks: no
	// landing is read.
	repo := folder("repo")
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	write("repo/.gitignore", "/bin/\n/tmp/\n")
	write("repo/docs/verification/run-record.txt", "outputs: read once from experiments/done.\n")
	document := plan.Plan{Campaign: "hygiene", Doctrine: []plan.Paragraph{{Governs: "standing", Text: "the plan of a fixture"}}, Lane: "master-lead",
		Items: []plan.Item{{ID: "open-row", Title: "a row of the fixture", Status: plan.StatusOpen, Steps: []plan.Step{{
			ID: "do", Title: "its one step", Status: plan.StatusOpen, Verify: "go version", Rationale: "the fixture's open row",
		}}}}}
	if err := plan.Save(filepath.Join(repo, filepath.FromSlash(plan.Path)), document); err != nil {
		t.Fatal(err)
	}
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")
	trunk := gitFixture(t, repo, "rev-parse", "HEAD")
	if err := processlock.EnsureStateDirectory(repo); err != nil {
		t.Fatal(err)
	}
	clean := at("temporary/sessions/s1/scratchpad/clean")
	gitFixture(t, repo, "worktree", "add", "-q", "--detach", clean, "master")
	// A commit ran in the clean copy and left the record of its run in its process state: a copy
	// with none is of unknown state, and unknown never refuses.
	if err := processlock.EnsureStateDirectory(clean); err != nil {
		t.Fatal(err)
	}
	write("temporary/sessions/s1/scratchpad/clean/"+processlock.StateDirectory+"/landing_waits.json", "[]\n")
	ahead := at("temporary/sessions/s1/scratchpad/ahead")
	gitFixture(t, repo, "worktree", "add", "-q", "--detach", ahead, "master")
	gitFixture(t, ahead, "commit", "-q", "--allow-empty", "-m", "ahead of the trunk")
	gitFixture(t, repo, "branch", "unmerged-free", gitFixture(t, ahead, "rev-parse", "HEAD"))
	gitFixture(t, repo, "branch", "merged-free", "master")
	write("repo/bin/loose.exe", "7 bytes")
	write("repo/tmp/landed-row/probe.go", "package probe\n")
	write("repo/stray/x", "stray\n")
	write("data/experiments/done/weights/step.bin", "weights")
	folder("data/experiments/orphan/out")
	write("temporary/go-build1/x", "object")
	folder("cache")
	write("outside/large.bin", "bytes behind a link")
	if !fixtureLink(at("outside"), filepath.Join(folder("repo/tmp/linked-row"), "out")) {
		t.Log("this machine makes neither a symbolic link nor a junction, so nothing about links is held here")
		if err := os.RemoveAll(at("repo/tmp/linked-row")); err != nil {
			t.Fatal(err)
		}
	}

	h := hygiene{
		checkout: repo, data: at("data"), sessions: at("temporary/sessions"), temporary: at("temporary"), cache: at("cache"),
		executable: at("elsewhere/loop.exe"),
		stores:     func() ([]leftover, error) { return nil, nil },
		registered: func() ([]string, error) { return nil, nil },
		processes:  func() ([]string, error) { return nil, nil },
	}
	receipt := filepath.Join(repo, processlock.StateDirectory, "closeout-"+trunk[:12]+".txt")
	measuredReceipt := filepath.Join(repo, processlock.StateDirectory, "leftovers-"+trunk[:12]+".txt")
	const removal = "go run ./cmd/loop -leftovers -delete <place>"

	snapshot := func() map[string]int64 {
		t.Helper()
		seen := map[string]int64{}
		if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			seen[path] = -1
			if entry.Type().IsRegular() {
				info, err := entry.Info()
				if err != nil {
					return err
				}
				seen[path] = info.Size()
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		for _, arguments := range [][]string{{"worktree", "list", "--porcelain"}, {"branch", "--list"}, {"stash", "list"}} {
			seen["git "+strings.Join(arguments, " ")+"\n"+gitFixture(t, repo, arguments...)] = 0
		}
		delete(seen, receipt)
		delete(seen, measuredReceipt)
		return seen
	}
	// closeout runs the close-out mode and returns its error, what it printed without the last
	// line, which names the receipt, and the entry lines it printed after the table: the refusing ones.
	closeout := func(h hygiene) (text string, last []string, failure error) {
		t.Helper()
		var output bytes.Buffer
		failure = leftoversCommand(h, true, &output)
		text = output.String()
		cut := strings.LastIndex(text, "loop: wrote ")
		if cut < 0 || !strings.Contains(text[cut:], receipt) || strings.Count(text[cut:], "\n") != 1 {
			t.Fatalf("the close-out printed\n%s\nwant one last line naming the receipt %s", text, receipt)
		}
		text = text[:cut]
		written, err := os.ReadFile(receipt)
		if err != nil || string(written) != text {
			t.Fatalf("the close-out printed\n%s\nand its receipt holds\n%s\n(%v), want the same text in both", text, written, err)
		}
		table := false
		for line := range strings.Lines(text) {
			fields := strings.Split(strings.TrimSuffix(line, "\n"), "\t")
			switch {
			case strings.HasPrefix(line, "# "):
				table = true
			case len(fields) != 6:
			case fields[2] != "-":
				t.Errorf("the close-out printed the size %s for %s, and a close-out measures nothing", fields[2], fields[4])
			case table:
				last = append(last, fields[3]+"|"+strings.TrimPrefix(filepath.ToSlash(fields[4]), filepath.ToSlash(root)+"/"))
			}
		}
		slices.Sort(last)
		return text, last, failure
	}
	// counted is the line of counts a close-out of h prints, from a listing of its own.
	counted := func(h hygiene) string {
		t.Helper()
		found, err := listLeftovers(h, false)
		if err != nil {
			t.Fatal(err)
		}
		states := map[string]int{}
		for _, entry := range found {
			states[entry.state]++
		}
		return fmt.Sprintf("listed %d entries: %d unknown, %d undeclared", len(found), states[stateUnknown], states[stateUndeclared])
	}
	unknownOf := func(h hygiene) (unknown int) {
		t.Helper()
		found, err := listLeftovers(h, false)
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range found {
			if entry.state == stateUnknown {
				unknown++
			}
		}
		return unknown
	}

	// The measured listing of the same commit is written first; the close-out must leave it.
	var listing bytes.Buffer
	if err := leftoversCommand(h, false, &listing); err != nil {
		t.Fatal(err)
	}
	measured, err := os.ReadFile(measuredReceipt)
	if err != nil || !strings.Contains(string(measured), "report\tfinished\t7\tbin\t") {
		t.Fatalf("the measured listing's receipt holds\n%s\n(%v), want the listing with its sizes", measured, err)
	}
	before := snapshot()

	text, last, failure := closeout(h)
	want := []string{
		"branch|refs/heads/merged-free",
		"experiment|data/experiments/done/weights",
		"helper-copy|temporary/sessions/s1/scratchpad/clean",
		"temporary-build|temporary/go-build1",
		"tmp|repo/tmp/landed-row",
	}
	if failure == nil || !slices.Equal(last, want) {
		t.Fatalf("the close-out ended as %v and its last entry lines name\n%s\nwant it refused, naming\n%s\nin\n%s", failure, strings.Join(last, "\n"), strings.Join(want, "\n"), text)
	}
	for _, line := range leftoverPlaces {
		if refuses := slices.ContainsFunc(want, func(entry string) bool { return strings.HasPrefix(entry, line.name+"|") }); refuses != (line.action == actionDelete) {
			t.Errorf("the fixture's refusing entries on the line %s, whose action is %s: %v; want one on every line that deletes and on no other", line.name, line.action, refuses)
		}
	}
	for _, field := range []string{fmt.Sprintf("%d in all", len(want)), receipt, removal} {
		if !strings.Contains(failure.Error(), field) {
			t.Errorf("the refusal is %q, which lacks %q", failure, field)
		}
	}
	if !strings.HasSuffix(failure.Error(), removal) || strings.Contains(failure.Error(), "merged-free") {
		t.Errorf("the refusal is %q, want it to end with the command that removes the entries and to name none of them", failure)
	}
	if !strings.Contains(text, counted(h)) {
		t.Errorf("the close-out's text lacks the line of counts %q:\n%s", counted(h), text)
	}
	// A listing that does not measure reads the files only of what a delete could take: the walk
	// of the build cache and the data root is what a close-out is spared.
	unmeasured, err := listLeftovers(h, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range unmeasured {
		if deletable := entry.action == actionDelete && entry.state == stateFinished; entry.measured || entry.bytes != 0 && !deletable {
			t.Errorf("a listing that does not measure read %d bytes of %s %s (measured: %v), which no delete could take", entry.bytes, entry.place, entry.path, entry.measured)
		}
	}
	if kept, err := os.ReadFile(measuredReceipt); err != nil || !bytes.Equal(kept, measured) {
		t.Errorf("after the close-out the measured listing's receipt holds\n%s\n(%v), want it as it was written", kept, err)
	}
	if after := snapshot(); !maps.Equal(before, after) {
		var moved []string
		for path, size := range before {
			if now, kept := after[path]; !kept || now != size {
				moved = append(moved, fmt.Sprintf("%s was %d and is %d (still there: %v)", path, size, now, kept))
			}
		}
		for path := range after {
			if _, was := before[path]; !was {
				moved = append(moved, path+" is new")
			}
		}
		slices.Sort(moved)
		t.Fatalf("the close-out changed the fixture, and it removes and writes nothing but its receipt:\n%s", strings.Join(moved, "\n"))
	}

	// The refusal of many entries is still shorter than what a commit's checks keep of a failed check.
	var many []leftover
	for index := range 200 {
		many = append(many, leftover{place: leftoverPlaces[index%len(leftoverPlaces)].name, path: at(fmt.Sprintf("made-up/entry-%d", index))})
	}
	long := closeoutRefusal(many, receipt).Error()
	total, named, remedy := strings.Index(long, "200 in all"), strings.Index(long, receipt), strings.Index(long, removal)
	if len(long) >= clioptions.DiagnosticTailBytes || total < 0 || named < total || remedy < named || !strings.HasSuffix(long, removal) || strings.Contains(long, "made-up") {
		t.Fatalf("the refusal of 200 entries is %d bytes: %q; want fewer than %d, ending with the total, the receipt and the command that removes them, and naming no entry", len(long), long, clioptions.DiagnosticTailBytes)
	}
	if closeoutRefusal(nil, receipt) != nil {
		t.Fatal("a close-out that no entry refuses is refused")
	}

	// Each refusing entry is cleared by the means of its kind; the close-out then passes, and counts.
	gitFixture(t, repo, "worktree", "remove", clean)
	gitFixture(t, repo, "branch", "-d", "merged-free")
	for _, slash := range []string{"data/experiments/done/weights", "temporary/go-build1", "repo/tmp/landed-row"} {
		if err := os.RemoveAll(at(slash)); err != nil {
			t.Fatal(err)
		}
	}
	text, last, failure = closeout(h)
	if failure != nil || len(last) != 0 || !strings.Contains(text, counted(h)) {
		t.Fatalf("with every refusing entry cleared the close-out ended as %v, naming %q, want it passed with the line of counts %q:\n%s", failure, last, counted(h), text)
	}
	if unknown := unknownOf(h); unknown == 0 || !strings.Contains(text, "undeclared\t-\tundeclared\t"+at("repo/stray")) {
		t.Fatalf("the passed close-out lists %d unknown entries and its text is\n%s\nwant an unknown entry and the undeclared directory still listed", unknown, text)
	}

	// A source that fails is one more unknown entry, and unknown never refuses.
	failing := h
	failing.stores = func() ([]leftover, error) { return nil, errors.New("the store's own listing failed") }
	text, _, failure = closeout(failing)
	if failure != nil || unknownOf(failing) != unknownOf(h)+1 || !strings.Contains(text, counted(failing)) {
		t.Fatalf("with the store's listing failed the close-out ended as %v with %d unknown entries where there were %d, want it passed with one more, in the line of counts %q:\n%s", failure, unknownOf(failing), unknownOf(h), counted(failing), text)
	}

	// One merged branch made again refuses, and it alone is named.
	gitFixture(t, repo, "branch", "merged-again", "master")
	text, last, failure = closeout(h)
	if failure == nil || !slices.Equal(last, []string{"branch|refs/heads/merged-again"}) || !strings.Contains(failure.Error(), "branch 1; 1 in all") {
		t.Fatalf("with one merged branch made again the close-out ended as %v, naming %q, want it refused for that branch alone:\n%s", failure, last, text)
	}
}
