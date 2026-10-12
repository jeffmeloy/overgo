package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/guard"
	"overgo/internal/landing"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/repoanalysis"
	"overgo/internal/testutil"
)

// TestLeftoversListsEveryDeclaredPlaceWithItsRule builds, under a temporary directory, a repository
// with one entry of each kind in each state: helper copies that are clean, changed, part deleted,
// emptied, holding ignored content, ahead of the trunk, held by a session and gone from disk; lane
// worktrees, branches and a stash; experiment outputs that a committed record names, that a store
// registration names and that nothing names; checkpoints, scratch, session folders, temporary build
// folders, executables, and directories no line of the table names. The listing gives exactly those
// verdicts, each with the rule that decided it. It measures bytes only where a line allows a delete
// or a report, reads a source that failed as unknown and never as nothing, on a line that deletes
// calls nothing finished that holds a link a recursive delete could follow, calls no entry finished
// for want of a committed line that names it when the search could not have found its name, prints
// one line an entry and writes the same text as a receipt named by the trunk's commit. And it
// removes nothing: the files, the worktrees, the branches and the stash are the same after it as
// before, but for the receipt.
func TestLeftoversListsEveryDeclaredPlaceWithItsRule(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
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
	git := func(directory string, arguments ...string) string {
		t.Helper()
		return gitFixture(t, directory, arguments...)
	}
	// A link a recursive delete could follow: a symbolic link, or a junction where that needs a privilege.
	link := func(target, name string) bool {
		return os.Symlink(target, name) == nil || exec.Command("cmd", "/c", "mklink", "/J", name, target).Run() == nil
	}
	relative := func(path string) string {
		t.Helper()
		if !filepath.IsAbs(path) {
			return path
		}
		inside, err := filepath.Rel(root, path)
		if err != nil {
			t.Fatal(err)
		}
		return filepath.ToSlash(inside)
	}
	verdicts := func(found []leftover) []string {
		t.Helper()
		lines := make([]string, len(found))
		for index, entry := range found {
			lines[index] = strings.Join([]string{entry.place, relative(entry.path), entry.action, entry.state}, "|")
		}
		slices.Sort(lines)
		return lines
	}
	entryOf := func(t *testing.T, found []leftover, place, slash string) leftover {
		t.Helper()
		for _, entry := range found {
			if entry.place == place && relative(entry.path) == slash {
				return entry
			}
		}
		t.Fatalf("the listing holds no %s entry %s:\n%s", place, slash, strings.Join(verdicts(found), "\n"))
		return leftover{}
	}
	list := func(t *testing.T, h hygiene) []leftover {
		t.Helper()
		found, err := listLeftovers(h)
		if err != nil {
			t.Fatal(err)
		}
		return found
	}

	// The trunk's checkout: what it commits, then what accumulates in it.
	repo := folder("repo")
	git(repo, "init", "-q", "-b", "master")
	git(repo, "config", "core.autocrlf", "false")
	write("repo/.gitignore", "/bin/\n/tmp/\n/dist/\n/build/*\n!/build/held.dat\novergodb-store/\n/scripts/python_golden_references/\n")
	write("repo/.claude/tracked.fixture", "{}\n")
	write("repo/build/held.dat", "declared\n")
	write("repo/scripts/hook.txt", "run bin/named.exe\nrun bin\\slashed.exe\nrun bin/two words.exe\n")
	// A record names a path by whole elements: a longer name before it, after it or around it is another path.
	// The last line, like the hook's, names an entry with a space in it, which the search reads as the end of a path.
	write("repo/docs/verification/run-record.txt", strings.Join([]string{
		"outputs: read once from Experiments/done.",
		"store: D:/data/experiments/kept/store",
		`reward: D:\data\experiments\kept\run\reward`,
		"model: checkpoints/cited, kept",
		"other: myexperiments/loose/out, experiments/orphan-old/out and experiments/coded.tar",
		"whole: (experiments/whole)",
		"final: experiments/done/final run",
		"",
	}, "\n"))
	for _, record := range repoanalysis.WorkRecords {
		write("repo/"+record, "experiments/planned/out bin/loose.exe\n")
	}
	write("repo/internal/x/x.go", "// Package x names experiments/coded/out.\npackage x\n")
	write("repo/docs/media_samples/sample.dat", "sample\n")
	git(repo, "add", "-A")
	git(repo, "commit", "-q", "-m", "base")
	trunk := git(repo, "rev-parse", "HEAD")
	write("repo/scripts/hook.txt", "work a stash holds\n")
	git(repo, "stash", "push", "-q", "-m", "kept work")
	git(repo, "branch", "merged-free", "master")
	if err := processlock.EnsureStateDirectory(repo); err != nil {
		t.Fatal(err)
	}
	copyAt := func(slash string, arguments ...string) string {
		t.Helper()
		git(repo, append(append([]string{"worktree", "add", "-q"}, arguments...), at(slash), "master")...)
		return at(slash)
	}
	copyAt("temporary/sessions/s1/scratchpad/clean", "--detach")
	copyAt("temporary/sessions/s1/scratchpad/dirty", "--detach")
	write("temporary/sessions/s1/scratchpad/dirty/note.txt", "uncommitted\n")
	partial := copyAt("temporary/sessions/s1/scratchpad/partial", "--detach")
	if err := os.Remove(filepath.Join(partial, "scripts", "hook.txt")); err != nil {
		t.Fatal(err)
	}
	emptied := copyAt("temporary/sessions/s1/scratchpad/emptied", "--detach")
	tracked := strings.Fields(git(emptied, "ls-files"))
	for _, file := range tracked {
		if err := os.Remove(filepath.Join(emptied, filepath.FromSlash(file))); err != nil {
			t.Fatal(err)
		}
	}
	// As many changed paths as tracked files, one of them a file added: not an emptied copy.
	replaced := copyAt("temporary/sessions/s1/scratchpad/replaced", "--detach")
	for _, file := range tracked[1:] {
		if err := os.Remove(filepath.Join(replaced, filepath.FromSlash(file))); err != nil {
			t.Fatal(err)
		}
	}
	write("temporary/sessions/s1/scratchpad/replaced/added.txt", "uncommitted\n")
	copyAt("temporary/sessions/s1/scratchpad/ignored", "--detach")
	write("temporary/sessions/s1/scratchpad/ignored/overgodb-store/blob", "ignored\n")
	ahead := copyAt("temporary/sessions/s2/scratchpad/ahead", "--detach")
	git(ahead, "commit", "-q", "--allow-empty", "-m", "ahead of the trunk")
	aheadCommit := git(ahead, "rev-parse", "HEAD")
	git(repo, "branch", "unmerged-free", aheadCommit)
	held := copyAt("temporary/sessions/s2/scratchpad/held", "--detach")
	if err := processlock.EnsureStateDirectory(held); err != nil {
		t.Fatal(err)
	}
	write("temporary/sessions/s2/scratchpad/held/"+processlock.StateDirectory+"/"+guard.WorktreeHolder, "s9\n")
	if err := os.RemoveAll(copyAt("temporary/sessions/s2/scratchpad/gone", "--detach")); err != nil {
		t.Fatal(err)
	}
	copyAt("repo/.claude/worktrees/agent-clean", "--detach")
	folder("repo/.claude/worktrees/agent-unlinked")
	folder("repo/.claude/other")
	git(copyAt("lanes/feature", "-b", "lane/feature"), "commit", "-q", "--allow-empty", "-m", "lane work")
	copyAt("lanes/merged", "-b", "merged-held")
	copyAt("repo/"+processlock.StateDirectory+"/cycle", "--detach")
	write("repo/bin/named.exe", "named")
	write("repo/bin/slashed.exe", "slashed")
	write("repo/bin/loose.exe", "7 bytes")
	write("repo/bin/two words.exe", "spaced")
	write("repo/tmp/probe.go", "package probe\n")
	write("repo/tmp/tool.exe", "tool")
	folder("repo/build/scrap")
	folder("repo/dist")
	folder("repo/scripts/python_golden_references")
	folder("repo/overgodb-store")
	write("repo/docs/local.json", "{}\n")
	write("repo/stray/x", "stray\n")

	// The data root: a real store that registers two locations, and what runs left beside it.
	storePath := folder("data/overgodb-store")
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	register := func(name string, kind artifact.LocationKind, value string) {
		t.Helper()
		id := testutil.ArtifactID(t, artifact.KindTensorSet, name)
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:       "leftovers-fixture/" + name,
			Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(name))}},
			Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: id, Kind: kind, Value: value}, Action: artifact.LocationAdd}},
		}); err != nil {
			t.Fatal(err)
		}
	}
	write("data/experiments/done/weights/step.bin", "weights")
	write("data/experiments/done/run.log", "log\n")
	folder("data/experiments/done/final run")
	folder("data/experiments/kept/store")
	write("data/experiments/kept/store.lock", "")
	folder("data/experiments/kept/run")
	write("data/experiments/kept/model/weights.bin", "model")
	for _, experiment := range []string{"orphan", "planned", "coded", "loose", "whole"} {
		folder("data/experiments/" + experiment + "/out")
	}
	write("data/experiments/notes.txt", "beside the experiments\n")
	folder("data/checkpoints/proto")
	folder("data/checkpoints/cited")
	folder("data/checkpoints/loose")
	write("data/scratch/log.txt", "scratch\n")
	folder("data/build")
	folder("data/models")
	folder("data/mystery")
	folder("data/overgodb-store.superseded-1")
	// Spelled through another directory, so only a comparison by file identity places it under the entry.
	register("model", artifact.LocationFile, strings.Join([]string{at("data/experiments/done"), "..", "kept", "model", "weights.bin"}, string(filepath.Separator)))
	register("proto", artifact.LocationDirectory, at("data/checkpoints/proto"))
	// A registered directory holds what lies in it: the record names this experiment, and nothing names its output.
	register("whole", artifact.LocationDirectory, at("data/experiments/whole"))
	// A remote location is no file on this machine, whatever its value reads as.
	register("elsewhere", artifact.LocationRemote, at("data/experiments/done/weights"))
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	// A second repository whose main worktree is on no branch, with nothing tracked beside its agent
	// folder, and which is its own data root: the directories it tracks are no leftovers.
	second := folder("second")
	git(second, "init", "-q", "-b", "master")
	write("second/source/tracked.dat", "tracked\n")
	git(second, "add", "-A")
	git(second, "commit", "-q", "-m", "base")
	git(second, "checkout", "-q", "--detach")
	git(second, "worktree", "add", "-q", "--detach", at("second/.claude/worktrees/agent"), "master")
	git(second, "worktree", "add", "-q", "-b", "side", at("second-lane"), "master")
	folder("second/.claude/other")
	folder("second-temporary")

	// The temporary directory, the session folders in it and the build cache.
	write("temporary/go-build1/x", "object")
	executable := at("temporary/go-build2/b001/exe/loop.exe")
	write("temporary/go-build2/b001/exe/loop.exe", "this command")
	folder("temporary/overgo-gate-1")
	folder("temporary/other")
	folder("cache")
	write("outside/large.bin", "bytes behind a link")
	linked := link(at("outside"), filepath.Join(folder("temporary/go-build4"), "b001")) &&
		link(at("outside"), filepath.Join(folder("data/experiments/done/linked"), "out")) &&
		link(at("outside"), at("data/scratch/out"))
	if !linked {
		t.Log("this machine makes neither a symbolic link nor a junction, so nothing about links is held here")
		for _, slash := range []string{"temporary/go-build4", "data/experiments/done/linked", "data/scratch/out"} {
			if err := os.RemoveAll(at(slash)); err != nil {
				t.Fatal(err)
			}
		}
	}

	printed := fmt.Sprintf(`[
		{"path": %q, "kind": "sealed-backup", "sequence": 7, "modified": "2026-10-01T00:00:00Z", "packed": true, "keep": true},
		{"path": %q, "kind": "superseded-store", "modified": "2026-10-02T00:00:00Z", "packed": false, "keep": true}
	]`, filepath.ToSlash(at("backups/sealed")), filepath.ToSlash(at("data/overgodb-store.superseded-1")))
	var storesReturned, processesReadEarly bool
	h := hygiene{
		checkout: repo, data: at("data"), sessions: at("temporary/sessions"), temporary: at("temporary"), cache: at("cache"),
		executable: executable,
		kept:       []string{storePath, at("data/models"), ""},
		stores: func() ([]leftover, error) {
			defer func() { storesReturned = true }()
			return decodeStoreLeftovers([]byte(printed))
		},
		registered: func() ([]string, error) { return storeLocations(storePath, 1) },
		processes: func() ([]string, error) {
			processesReadEarly = processesReadEarly || !storesReturned
			return nil, nil
		},
	}

	snapshot := func(t *testing.T) map[string]int64 {
		t.Helper()
		seen := map[string]int64{}
		if err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			seen[relative(path)] = -1
			if entry.Type().IsRegular() {
				info, err := entry.Info()
				if err != nil {
					return err
				}
				seen[relative(path)] = info.Size()
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		for _, arguments := range [][]string{{"worktree", "list", "--porcelain"}, {"branch", "--list"}, {"stash", "list"}} {
			seen["git "+strings.Join(arguments, " ")+"\n"+gitFixture(t, repo, arguments...)] = 0
		}
		return seen
	}
	before := snapshot(t)
	registeredBefore, err := landing.Worktrees(ctx, repo)
	if err != nil || len(registeredBefore) != 14 || registeredBefore[0].Branch != "refs/heads/master" || registeredBefore[0].Head != trunk {
		t.Fatalf("the fixture registers %d worktrees, the first %+v (%v); want 14 with the trunk's checkout first, on master at %s", len(registeredBefore), registeredBefore[:min(len(registeredBefore), 1)], err, trunk)
	}

	found := list(t, h)
	t.Run("every entry has the verdict its state gives it", func(t *testing.T) {
		want := []string{
			"helper-copy|temporary/sessions/s1/scratchpad/clean|delete|finished",
			"helper-copy|temporary/sessions/s1/scratchpad/dirty|delete|held",
			"helper-copy|temporary/sessions/s1/scratchpad/partial|delete|held",
			"helper-copy|temporary/sessions/s1/scratchpad/emptied|delete|finished",
			"helper-copy|temporary/sessions/s1/scratchpad/replaced|delete|held",
			"helper-copy|temporary/sessions/s1/scratchpad/ignored|delete|held",
			"helper-copy|temporary/sessions/s2/scratchpad/ahead|delete|held",
			"helper-copy|temporary/sessions/s2/scratchpad/held|delete|held",
			"helper-copy|temporary/sessions/s2/scratchpad/gone|delete|unknown",
			"helper-copy|repo/.claude/worktrees/agent-clean|delete|finished",
			"helper-copy|repo/.claude/worktrees/agent-unlinked|delete|unknown",
			"worktree|lanes/feature|report|held",
			"worktree|lanes/merged|report|finished",
			"branch|refs/heads/lane/feature|delete|held",
			"branch|refs/heads/merged-free|delete|finished",
			"branch|refs/heads/merged-held|delete|held",
			"branch|refs/heads/unmerged-free|delete|held",
			"stash|stash@{0}|report|unknown",
			"store-backup|backups/sealed|never|kept",
			"store-backup|data/overgodb-store.superseded-1|never|kept",
			"experiment|data/experiments/done/weights|delete|finished",
			"experiment|data/experiments/done/run.log|delete|unknown",
			"experiment|data/experiments/done/final run|delete|unknown",
			"experiment|data/experiments/kept/store|delete|held",
			"experiment|data/experiments/kept/store.lock|delete|unknown",
			"experiment|data/experiments/kept/run|delete|held",
			"experiment|data/experiments/kept/model|delete|held",
			"experiment|data/experiments/orphan/out|delete|unknown",
			"experiment|data/experiments/whole/out|delete|held",
			"experiment|data/experiments/notes.txt|delete|unknown",
			"experiment|data/experiments/planned/out|delete|unknown",
			"experiment|data/experiments/coded/out|delete|unknown",
			"experiment|data/experiments/loose/out|delete|unknown",
			"checkpoint|data/checkpoints/proto|report|held",
			"checkpoint|data/checkpoints/cited|report|held",
			"checkpoint|data/checkpoints/loose|report|unknown",
			"scratch|data/scratch/log.txt|report|unknown",
			"data-build|data/build|report|unknown",
			"session-scratch|temporary/sessions/s1|report|unknown",
			"session-scratch|temporary/sessions/s2|report|unknown",
			"temporary-build|temporary/go-build1|delete|finished",
			"temporary-build|temporary/go-build2|delete|held",
			"build-cache|cache|report|unknown",
			"bin|repo/bin/named.exe|report|held",
			"bin|repo/bin/slashed.exe|report|held",
			"bin|repo/bin/loose.exe|report|finished",
			"bin|repo/bin/two words.exe|report|unknown",
			"tmp|repo/tmp/probe.go|report|unknown",
			"tmp|repo/tmp/tool.exe|report|unknown",
			"build|repo/build|report|unknown",
			"release|repo/dist|report|unknown",
			"python-place|repo/scripts/python_golden_references|never|kept",
			"checkout-store|repo/overgodb-store|never|kept",
			"docs-untracked|repo/docs/local.json|never|kept",
			"media-samples|repo/docs/media_samples|report|unknown",
			"verification|repo/docs/verification|report|unknown",
			"process-state|repo/.overgo-runtime|never|kept",
			"undeclared|repo/.claude/other|report|undeclared",
			"undeclared|repo/stray|report|undeclared",
			"undeclared|data/mystery|report|undeclared",
		}
		if linked {
			want = append(want, "temporary-build|temporary/go-build4|delete|unknown", "experiment|data/experiments/done/linked|delete|unknown", "scratch|data/scratch/out|report|unknown")
		}
		slices.Sort(want)
		if got := verdicts(found); !slices.Equal(got, want) {
			t.Fatalf("the listing's verdicts are\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
		if processesReadEarly {
			t.Fatal("the running processes were read before the store's own listing returned, while its go command could still run")
		}
	})

	t.Run("every line of the table is whole and has an entry", func(t *testing.T) {
		names := map[string]bool{}
		for _, line := range leftoverPlaces {
			word, _, cut := strings.Cut(line.at, ":")
			if line.name == "" || names[line.name] || !cut || word == "" || line.holds == "" || line.finished == "" || line.list == nil ||
				!slices.Contains([]string{actionDelete, actionReport, actionNever}, line.action) {
				t.Errorf("line %+v is not whole: it needs a name of its own, a root word, what it holds, its finished state and an action", line)
			}
			names[line.name] = true
			if !slices.ContainsFunc(found, func(entry leftover) bool { return entry.place == line.name }) {
				t.Errorf("the fixture holds no entry of line %s, so nothing holds its verdicts", line.name)
			}
		}
	})

	t.Run("every entry carries the rule that decided it", func(t *testing.T) {
		for _, entry := range found {
			if strings.TrimSpace(entry.rule) == "" {
				t.Errorf("%s %s is %s by no rule", entry.place, relative(entry.path), entry.state)
			}
		}
		for _, named := range []struct{ place, slash, rule string }{
			{"experiment", "data/experiments/kept/store", "docs/verification/run-record.txt:2"},
			{"experiment", "data/experiments/kept/run", "docs/verification/run-record.txt:3"},
			{"experiment", "data/experiments/kept/model", "weights.bin"},
			{"checkpoint", "data/checkpoints/cited", "docs/verification/run-record.txt:4"},
			{"checkpoint", "data/checkpoints/proto", "proto"},
			{"experiment", "data/experiments/whole/out", filepath.ToSlash(at("data/experiments/whole"))},
			{"experiment", "data/experiments/orphan/out", "no committed record names its experiment"},
			{"experiment", "data/experiments/coded/out", "no committed record names its experiment"},
			{"experiment", "data/experiments/done/run.log", "a file"},
			{"experiment", "data/experiments/done/final run", "no line that names it could be found"},
			{"bin", "repo/bin/named.exe", "scripts/hook.txt:1"},
			{"bin", "repo/bin/slashed.exe", "scripts/hook.txt:2"},
			{"bin", "repo/bin/two words.exe", "no line that names it could be found"},
			{"helper-copy", "temporary/sessions/s2/scratchpad/ahead", aheadCommit[:12]},
			{"helper-copy", "temporary/sessions/s2/scratchpad/held", "s9"},
			{"helper-copy", "temporary/sessions/s1/scratchpad/ignored", "overgodb-store/"},
			{"helper-copy", "temporary/sessions/s1/scratchpad/emptied", fmt.Sprintf("every one of its %d tracked files is missing", len(tracked))},
			{"helper-copy", "temporary/sessions/s1/scratchpad/partial", fmt.Sprintf("1 of them deletions among its %d tracked files", len(tracked))},
			{"worktree", "lanes/merged", "on the trunk"},
			{"branch", "refs/heads/merged-held", filepath.ToSlash(at("lanes/merged"))},
			{"branch", "refs/heads/unmerged-free", "commits the trunk lacks"},
			{"helper-copy", "temporary/sessions/s1/scratchpad/dirty", "note.txt"},
			{"stash", "stash@{0}", "kept work"},
			{"store-backup", "backups/sealed", "sealed-backup"},
			{"tmp", "repo/tmp/tool.exe", "an executable outside bin"},
			{"temporary-build", "temporary/go-build2", "this command's own executable"},
		} {
			if entry := entryOf(t, found, named.place, named.slash); !strings.Contains(filepath.ToSlash(entry.rule), named.rule) {
				t.Errorf("%s %s is %s by the rule %q, which does not name %q", named.place, named.slash, entry.state, entry.rule, named.rule)
			}
		}
	})

	t.Run("bytes are measured where a line deletes or reports and nowhere else", func(t *testing.T) {
		if loose := entryOf(t, found, "bin", "repo/bin/loose.exe"); !loose.measured || loose.bytes != 7 {
			t.Fatalf("bin/loose.exe measures %d bytes (measured %v), want its 7", loose.bytes, loose.measured)
		}
		if weights := entryOf(t, found, "experiment", "data/experiments/done/weights"); !weights.measured || weights.bytes != 7 {
			t.Fatalf("the finished experiment output measures %d bytes (measured %v), want the 7 of its one file", weights.bytes, weights.measured)
		}
		for _, entry := range found {
			if unmeasured := entry.action == actionNever || !filepath.IsAbs(entry.path); unmeasured && entry.measured {
				t.Errorf("%s %s is measured, and a line that never deletes, a branch and a stash are not", entry.place, relative(entry.path))
			}
		}
		if gone := entryOf(t, found, "helper-copy", "temporary/sessions/s2/scratchpad/gone"); gone.measured {
			t.Fatalf("the copy whose directory is gone measures %d bytes, and what cannot be walked is not measured", gone.bytes)
		}
		// A report's entry is measured without what lies behind a link, which the walk does not follow.
		if linked {
			if behind := entryOf(t, found, "scratch", "data/scratch/out"); !behind.measured || behind.bytes != 0 {
				t.Fatalf("the scratch entry that is a link measures %d bytes (measured %v), want none of what lies behind it", behind.bytes, behind.measured)
			}
		}
	})

	t.Run("a temporary build folder is held while a go command or a test binary runs", func(t *testing.T) {
		running := h
		running.processes = func() ([]string, error) { return []string{"loop.test.exe"}, nil }
		if entry := entryOf(t, list(t, running), "temporary-build", "temporary/go-build1"); entry.state != stateHeld || !strings.Contains(entry.rule, "loop.test.exe") {
			t.Fatalf("with a test binary running the folder is %s by %q, want held by the binary's name", entry.state, entry.rule)
		}
		unread := h
		unread.processes = func() ([]string, error) { return nil, errors.New("the process list is unreadable") }
		listed := list(t, unread)
		if entry := entryOf(t, listed, "temporary-build", "temporary/go-build1"); entry.state != stateUnknown || !strings.Contains(entry.rule, "the process list is unreadable") {
			t.Fatalf("with the process list unreadable the folder is %s by %q, want unknown by the failure", entry.state, entry.rule)
		}
		if entry := entryOf(t, listed, "temporary-build", "temporary/go-build2"); entry.state != stateHeld {
			t.Fatalf("the folder that holds this command's executable is %s when the process list is unreadable, want held", entry.state)
		}
		const sample = "\"System Idle Process\",\"0\",\"Services\",\"0\",\"8 K\"\r\n\"go.exe\",\"4100\",\"Console\",\"1\",\"21,004 K\"\r\n" +
			"\"GO.EXE\",\"4200\",\"Console\",\"1\",\"19,112 K\"\r\n\"loop.test.exe\",\"4300\",\"Console\",\"1\",\"9,000 K\"\r\n\"gopls.exe\",\"4400\",\"Console\",\"1\",\"9,000 K\"\r\n"
		if got, err := goProcesses(sample, 4100); err != nil || !slices.Equal(got, []string{"GO.EXE", "loop.test.exe"}) {
			t.Fatalf("the go commands and test binaries beside the parent 4100 are %q (%v), want the second go command and the test binary", got, err)
		}
		if got, err := goProcesses("\"go.exe\",\"4100\",\"Console\",\"1\",\"21,004 K\"\r\n", 4100); err != nil || len(got) != 0 {
			t.Fatalf("with only this command's parent running the list is %q (%v), want none", got, err)
		}
		for _, unreadable := range []string{"", "\"go.exe\"\r\n", "\"go.exe\",\"41\"00\"\r\n"} {
			if got, err := goProcesses(unreadable, 4100); err == nil {
				t.Fatalf("the process list %q read as %q, want it refused: an empty or malformed list says nothing about what runs", unreadable, got)
			}
		}
	})

	t.Run("a source that fails reads as unknown and never as nothing", func(t *testing.T) {
		failing := h
		failing.registered = func() ([]string, error) { return nil, errors.New("the store's registrations are unreadable") }
		failing.stores = func() ([]leftover, error) { return nil, errors.New("the store's own listing failed") }
		failing.sessions = at("temporary/absent")
		listed := list(t, failing)
		if entry := entryOf(t, listed, "experiment", "data/experiments/done/weights"); entry.state != stateUnknown || !strings.Contains(entry.rule, "the store's registrations are unreadable") {
			t.Fatalf("with the registrations unreadable the output is %s by %q, want unknown by the failure", entry.state, entry.rule)
		}
		if entry := entryOf(t, listed, "experiment", "data/experiments/kept/store"); entry.state != stateHeld {
			t.Fatalf("the output a committed record names is %s with the registrations unreadable, want held", entry.state)
		}
		var backups, sessions []leftover
		for _, entry := range listed {
			switch entry.place {
			case "store-backup":
				backups = append(backups, entry)
			case "session-scratch":
				sessions = append(sessions, entry)
			}
		}
		if len(backups) != 1 || backups[0].state != stateUnknown || !strings.Contains(backups[0].rule, "the store's own listing failed") {
			t.Fatalf("with the store's listing failed its line holds %+v, want one unknown entry carrying the failure", backups)
		}
		if len(sessions) != 1 || sessions[0].state != stateUnknown || sessions[0].path != at("temporary/absent") {
			t.Fatalf("with no session folder its line holds %+v, want one unknown entry naming the folder tried", sessions)
		}
		for _, malformed := range []string{`[{"path": "", "kind": "sealed-backup"}]`, `[{"path": "p"}]`, `{"path": "p"}`} {
			if decoded, err := decodeStoreLeftovers([]byte(malformed)); err == nil {
				t.Fatalf("the store listing %s decoded as %+v, want it refused", malformed, decoded)
			}
		}
		if decoded, err := decodeStoreLeftovers([]byte("null\n")); err != nil || len(decoded) != 0 {
			t.Fatalf("an empty store listing decoded as %+v (%v), want none", decoded, err)
		}
	})

	t.Run("with the main worktree on no branch nothing is on the trunk, and a checkout that is its data root is read once", func(t *testing.T) {
		detached := hygiene{
			checkout: second, data: second, sessions: at("second-sessions"), temporary: at("second-temporary"), cache: at("second-cache"),
			stores:     func() ([]leftover, error) { return nil, nil },
			registered: func() ([]string, error) { return nil, nil },
			processes:  func() ([]string, error) { return nil, nil },
		}
		want := []string{
			"branch|refs/heads/master|delete|unknown",
			"branch|refs/heads/side|delete|unknown",
			"helper-copy|second/.claude/worktrees/agent|delete|unknown",
			"session-scratch|second-sessions|report|unknown",
			"undeclared|second/.claude/other|report|undeclared",
			"worktree|second-lane|report|unknown",
		}
		if got := verdicts(list(t, detached)); !slices.Equal(got, want) {
			t.Fatalf("with the main worktree detached the verdicts are\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
		}
	})

	t.Run("the command prints the listing, writes it as a receipt and removes nothing", func(t *testing.T) {
		var output bytes.Buffer
		if err := leftoversCommand(h, &output); err != nil {
			t.Fatal(err)
		}
		receipt := filepath.Join(repo, processlock.StateDirectory, "leftovers-"+trunk[:12]+".txt")
		written, err := os.ReadFile(receipt)
		if err != nil {
			t.Fatal(err)
		}
		text, last := output.String(), ""
		if cut := strings.LastIndex(text, "loop: "); cut >= 0 {
			text, last = text[:cut], text[cut:]
		}
		if text != string(written) || !strings.Contains(last, receipt) || strings.Count(last, "\n") != 1 {
			t.Fatalf("the command printed\n%s\nand wrote\n%s\nwant the same listing in both and one last line naming %s", output.String(), written, receipt)
		}
		entries, places := 0, 0
		for line := range strings.Lines(text) {
			switch fields := strings.Split(strings.TrimSuffix(line, "\n"), "\t"); {
			case strings.HasPrefix(line, "# "):
				places++
			case len(fields) != 6 || slices.Contains(fields, ""):
				t.Errorf("the line %q is not an entry's six fields: action, state, bytes, place, path, rule", line)
			default:
				entries++
			}
		}
		if entries != len(found) || places != len(leftoverPlaces) {
			t.Fatalf("the command printed %d entries and %d places, want the listing's %d and the table's %d", entries, places, len(found), len(leftoverPlaces))
		}
		for _, line := range []string{
			"report\tfinished\t7\tbin\t" + at("repo/bin/loose.exe") + "\t",
			"never\tkept\t-\tprocess-state\t" + at("repo/.overgo-runtime") + "\t",
			"delete\theld\t-\tbranch\trefs/heads/merged-held\t",
			"# bin at checkout:bin/: ",
			"; a finished entry: report; 4 entries, 25 bytes measured, 1 finished\n",
			"# branch at git:refs/heads: ",
			"; a finished entry: delete; 4 entries, 0 bytes measured, 1 finished\n",
		} {
			if !strings.Contains(text, line) {
				t.Errorf("the printed listing lacks the line opening %q:\n%s", line, text)
			}
		}

		after := snapshot(t)
		delete(after, relative(receipt))
		if !maps.Equal(before, after) {
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
			t.Fatalf("the listings changed the fixture, and a listing removes and writes nothing but its receipt:\n%s", strings.Join(moved, "\n"))
		}
		if registered, err := landing.Worktrees(ctx, repo); err != nil || !slices.Equal(registered, registeredBefore) {
			t.Fatalf("after the listings git registers %+v (%v), want the worktrees it registered before", registered, err)
		}
	})
}
