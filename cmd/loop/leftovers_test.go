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
	"overgo/internal/plan"
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
	linked := fixtureLink(at("outside"), filepath.Join(folder("temporary/go-build4"), "b001")) &&
		fixtureLink(at("outside"), filepath.Join(folder("data/experiments/done/linked"), "out")) &&
		fixtureLink(at("outside"), at("data/scratch/out"))
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
			"tmp|repo/tmp/probe.go|delete|unowned",
			"tmp|repo/tmp/tool.exe|delete|misplaced",
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
			{"tmp", "repo/tmp/tool.exe", "an executable belongs in bin/"},
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

// fixtureLink makes a link a recursive delete could follow, a symbolic link, or a junction where
// that needs a privilege, and reports whether this machine made one.
func fixtureLink(target, name string) bool {
	return os.Symlink(target, name) == nil || exec.Command("cmd", "/c", "mklink", "/J", name, target).Run() == nil
}

// TestTmpEntryBelongsToAnOpenRowOrIsFinished builds, under a temporary directory, a repository whose
// scratch folder holds one entry of each kind; the real scratch folder is never read. A folder named
// for a row is held while either plan holds the row, the working plan or the plan of the trunk's
// commit, in any letter case, and while a committed line or a store registration names it. It is
// finished only when neither plan holds its row and a commit on the trunk's first-parent chain
// landed a step of a row of exactly its name, in a trailer git itself reads as one. A landing on a
// branch nobody merged, one a merge brought in, one named in a message's prose, one under another
// trailer's key and one that names no step finish nothing: such a folder is unowned, as is a folder
// that was never a row and a file outside a row's folder. An executable and a Python file are
// misplaced, each with the place it belongs, and a folder's rule names those directly inside it.
// The scratch module's two files are kept. What cannot be established is unknown and kept: every
// folder when the working plan cannot be read, a landed folder when the store's registrations
// cannot be read or the main worktree is on no branch, and a finished folder that holds a link.
// Every entry's action is delete, an entry's bytes are measured, and no listing removes anything.
func TestTmpEntryBelongsToAnOpenRowOrIsFinished(t *testing.T) {
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
	const trailer = plan.LandedRowTrailer + ": "
	// land makes an empty commit whose message ends with the paragraphs given, and returns the commit.
	land := func(repository string, paragraphs ...string) string {
		t.Helper()
		arguments := []string{"commit", "-q", "--allow-empty", "-m", "an empty landing"}
		for _, paragraph := range paragraphs {
			arguments = append(arguments, "-m", paragraph)
		}
		gitFixture(t, repository, arguments...)
		return gitFixture(t, repository, "rev-parse", "HEAD")
	}
	// savePlan writes the working plan of a checkout: the rows named, each with one open step whose
	// reasoning names a scratch folder, which holds nothing since the plan is a work record.
	savePlan := func(checkout string, rows ...string) {
		t.Helper()
		document := plan.Plan{Campaign: "hygiene", Doctrine: "the plan of a fixture", Lane: "master-lead"}
		for _, row := range rows {
			document.Items = append(document.Items, plan.Item{ID: row, Title: "a row of the fixture", Status: plan.StatusOpen, Steps: []plan.Step{{
				ID: "do", Title: "its one step", Status: plan.StatusOpen, Verify: "go version", Rationale: "its probes lie under tmp/landed-row/x",
			}}})
		}
		if err := plan.Save(filepath.Join(checkout, filepath.FromSlash(plan.Path)), document); err != nil {
			t.Fatal(err)
		}
	}

	// The trunk's checkout: a record and a source file that name scratch entries, the plan, and
	// then the landings, each an empty commit.
	repo := folder("repo")
	gitFixture(t, repo, "init", "-q", "-b", "master")
	gitFixture(t, repo, "config", "core.autocrlf", "false")
	write("repo/.gitignore", "/tmp/\n")
	write("repo/docs/verification/scratch-record.note", "overlay: tmp/cited-row/overlay.data\n")
	// The second name has no boundary on its left, so it names nothing in the scratch folder.
	write("repo/internal/x/x.go", "// Package x reads tmp/cited-file.go and mytmp/loose.go, and its probes build under tmp/go.mod.\npackage x\n")
	savePlan(repo, "open-row", "reopened-row", "refiled-row", "Case-Row", "tip-row")
	gitFixture(t, repo, "add", "-A")
	gitFixture(t, repo, "commit", "-q", "-m", "base")
	older := land(repo, trailer+"landed-row/first")
	land(repo, trailer+"pair-a/do\n"+trailer+"pair-b/do")
	for _, step := range []string{"cited-row/do", "reopened-row/first", "refiled-row/do", "case-row/do", "loud-row/do", "tip-row/first", "filed-row/do", "registered-row/do"} {
		land(repo, trailer+step)
	}
	newest := land(repo, trailer+"landed-row/do")
	land(repo, trailer+"prose-row/do", "The line above names a row in the prose of a message, and this paragraph ends it")
	land(repo, "Reviewed-Row: reviewed-row/do")
	land(repo, trailer+"stepless-row")
	gitFixture(t, repo, "checkout", "-q", "-b", "lane/other")
	land(repo, trailer+"lane-row/do")
	gitFixture(t, repo, "checkout", "-q", "-b", "lane/merged", "master")
	land(repo, trailer+"merged-row/do")
	gitFixture(t, repo, "checkout", "-q", "master")
	gitFixture(t, repo, "merge", "-q", "--no-ff", "-m", "merge the lane", "lane/merged")
	// The working plan after the last commit, uncommitted: one row landed whole and left it, one was filed.
	savePlan(repo, "open-row", "reopened-row", "refiled-row", "Case-Row", "filed-row")

	folders := []string{
		"open-row", "reopened-row", "refiled-row", "case-row", "tip-row", "filed-row", "cited-row", "registered-row", "outputs",
		"landed-row", "pair-a", "pair-b", "LOUD-ROW", "lane-row", "merged-row", "prose-row", "reviewed-row", "stepless-row", "prep", "folder.exe",
	}
	for _, name := range folders {
		folder("repo/tmp/" + name)
	}
	write("repo/tmp/open-row/probe.go", "package probe\n")
	write("repo/tmp/open-row/built.exe", "built")
	write("repo/tmp/open-row/packed.exe/inner.exe", "inner")
	write("repo/tmp/prep/old.py", "print('old')\n")
	write("repo/tmp/landed-row/probe.go", "package probe\n")
	for _, name := range []string{"loose.go", "cited-file.go", "weights.bin", "tool.exe", "TOOL2.EXE", "script.py", "UPPER.PY", "go.mod", "go.sum"} {
		write("repo/tmp/"+name, "a file of the fixture\n")
	}
	write("repo/tmp/notes.txt", "notes")
	write("outside/large.bin", "bytes behind a link")

	// A second repository whose main worktree is on no branch: nothing says what its trunk is.
	second := folder("second")
	gitFixture(t, second, "init", "-q", "-b", "master")
	gitFixture(t, second, "config", "core.autocrlf", "false")
	write("second/.gitignore", "/tmp/\n")
	folder("second/docs")
	savePlan(second, "open-row")
	gitFixture(t, second, "add", "-A")
	gitFixture(t, second, "commit", "-q", "-m", "base")
	land(second, trailer+"landed-row/do")
	gitFixture(t, second, "checkout", "-q", "--detach")
	folder("second/tmp/open-row")
	folder("second/tmp/landed-row")

	// The store registers a location in a landed row's folder, one in a folder that was never a
	// row's, a file of the scratch folder itself, and one that is no scratch entry's.
	locations := []string{at("repo/tmp/registered-row/model.bin"), at("repo/tmp/outputs/run.bin"), at("repo/tmp/weights.bin"), at("data/elsewhere.bin")}
	h := hygiene{
		checkout: repo, data: folder("data"), sessions: folder("sessions"), temporary: folder("temporary"), cache: folder("cache"),
		executable: at("temporary/loop.exe"),
		stores:     func() ([]leftover, error) { return nil, nil },
		registered: func() ([]string, error) { return locations, nil },
		processes:  func() ([]string, error) { return nil, nil },
	}

	// scratch lists a checkout and returns the entries of its scratch folder by name, and their
	// states as sorted lines. Every entry is a child of the folder, and its action is delete.
	scratch := func(t *testing.T, h hygiene) (map[string]leftover, []string) {
		t.Helper()
		found, err := listLeftovers(h)
		if err != nil {
			t.Fatal(err)
		}
		entries, states := map[string]leftover{}, []string{}
		for _, entry := range found {
			if entry.place != "tmp" {
				continue
			}
			name := filepath.Base(entry.path)
			if entry.path != filepath.Join(h.checkout, "tmp", name) {
				t.Errorf("the scratch entry %s is no child of the scratch folder of %s", entry.path, h.checkout)
			}
			if entry.action != "delete" {
				t.Errorf("the scratch entry %s has the action %s, want delete", name, entry.action)
			}
			if strings.TrimSpace(entry.rule) == "" {
				t.Errorf("the scratch entry %s is %s by no rule", name, entry.state)
			}
			entries[name] = entry
			states = append(states, name+"|"+entry.state)
		}
		slices.Sort(states)
		return entries, states
	}
	base := map[string]string{
		"open-row": "held", "reopened-row": "held", "refiled-row": "held", "case-row": "held", "tip-row": "held", "filed-row": "held",
		"cited-row": "held", "registered-row": "held", "outputs": "held",
		"landed-row": "finished", "pair-a": "finished", "pair-b": "finished",
		"LOUD-ROW": "unowned", "lane-row": "unowned", "merged-row": "unowned", "prose-row": "unowned", "reviewed-row": "unowned", "stepless-row": "unowned",
		"prep": "unowned", "folder.exe": "unowned", "loose.go": "unowned", "notes.txt": "unowned",
		"cited-file.go": "held", "weights.bin": "held",
		"tool.exe": "misplaced", "TOOL2.EXE": "misplaced", "script.py": "misplaced", "UPPER.PY": "misplaced",
		"go.mod": "kept", "go.sum": "kept",
	}
	// wanted is the states of the fixture's entries as sorted lines, with those given in place of the usual ones.
	wanted := func(changed map[string]string) []string {
		states := maps.Clone(base)
		maps.Copy(states, changed)
		lines := make([]string, 0, len(states))
		for name, state := range states {
			lines = append(lines, name+"|"+state)
		}
		slices.Sort(lines)
		return lines
	}
	// named fails unless an entry has the state given and its rule carries the text given, with either slash.
	named := func(t *testing.T, entries map[string]leftover, name, state, text string) {
		t.Helper()
		if entry := entries[name]; entry.state != state || !strings.Contains(filepath.ToSlash(entry.rule), filepath.ToSlash(text)) {
			t.Errorf("%s is %s by the rule %q, want %s by a rule that names %q", name, entry.state, entry.rule, state, text)
		}
	}
	snapshot := func(t *testing.T) map[string]int64 {
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
		for _, repository := range []string{repo, second} {
			for _, arguments := range [][]string{{"log", "--all", "--format=%H"}, {"branch", "--list"}} {
				seen[repository+" git "+strings.Join(arguments, " ")+"\n"+gitFixture(t, repository, arguments...)] = 0
			}
		}
		return seen
	}
	before := snapshot(t)
	// unchanged fails when a listing removed, added or resized anything in the fixture, or moved a commit or a branch.
	unchanged := func(t *testing.T) {
		t.Helper()
		if after := snapshot(t); !maps.Equal(before, after) {
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
			t.Fatalf("the listing changed the fixture, and a listing removes and writes nothing:\n%s", strings.Join(moved, "\n"))
		}
	}

	t.Run("every entry has the state its row gives it, by a rule that names what decided it", func(t *testing.T) {
		entries, states := scratch(t, h)
		if want := wanted(nil); !slices.Equal(states, want) {
			t.Fatalf("the scratch entries are\n%s\nwant\n%s", strings.Join(states, "\n"), strings.Join(want, "\n"))
		}
		for _, entry := range []struct{ name, state, text string }{
			{"open-row", "held", "built.exe"},
			{"open-row", "held", "bin/"},
			{"prep", "unowned", "old.py"},
			{"prep", "unowned", "scripts/python_golden_references/"},
			{"landed-row", "finished", newest[:12]},
			{"cited-row", "held", "docs/verification/scratch-record.note:1"},
			{"cited-file.go", "held", "internal/x/x.go:1"},
			{"registered-row", "held", locations[0]},
			{"outputs", "held", locations[1]},
			{"weights.bin", "held", locations[2]},
			{"script.py", "misplaced", "scripts/python_golden_references/"},
			{"UPPER.PY", "misplaced", "scripts/python_golden_references/"},
			{"tool.exe", "misplaced", "bin/"},
			{"TOOL2.EXE", "misplaced", "bin/"},
		} {
			named(t, entries, entry.name, entry.state, entry.text)
		}
		// The check of what lies in a folder goes one level and names files, and the rule names the newest landing.
		if rule := entries["open-row"].rule; strings.Contains(rule, "inner.exe") || strings.Contains(rule, "packed.exe") || strings.Contains(rule, "probe.go") {
			t.Errorf("the rule of open-row is %q, and it names a file that is in its place, a directory, or a file that lies deeper than the folder", rule)
		}
		if rule := entries["landed-row"].rule; strings.Contains(rule, older[:12]) {
			t.Errorf("the rule of landed-row is %q, and it names the older landing %s, not the newest", rule, older[:12])
		}
		if notes := entries["notes.txt"]; !notes.measured || notes.bytes != 5 {
			t.Errorf("notes.txt measures %d bytes (measured %v), want its 5", notes.bytes, notes.measured)
		}
		unchanged(t)
	})

	t.Run("with the store's registrations unreadable a landed folder is unknown, and nothing reads as registered", func(t *testing.T) {
		failing := h
		failing.registered = func() ([]string, error) { return nil, errors.New("the store's registrations are unreadable") }
		entries, states := scratch(t, failing)
		want := wanted(map[string]string{"landed-row": "unknown", "pair-a": "unknown", "pair-b": "unknown", "registered-row": "unknown", "outputs": "unowned", "weights.bin": "unowned"})
		if !slices.Equal(states, want) {
			t.Fatalf("with the registrations unreadable the scratch entries are\n%s\nwant\n%s", strings.Join(states, "\n"), strings.Join(want, "\n"))
		}
		for _, name := range []string{"landed-row", "pair-a", "pair-b", "registered-row"} {
			named(t, entries, name, "unknown", "the store's registrations are unreadable")
		}
		unchanged(t)
	})

	t.Run("with the main worktree on no branch a row's folder is held and a landed one is unknown", func(t *testing.T) {
		detached := h
		detached.checkout = second
		entries, states := scratch(t, detached)
		if want := []string{"landed-row|unknown", "open-row|held"}; !slices.Equal(states, want) {
			t.Fatalf("with the main worktree detached the scratch entries are %q, want %q", states, want)
		}
		named(t, entries, "landed-row", "unknown", "on no branch")
		unchanged(t)
	})

	t.Run("with the working plan unreadable every folder is unknown and every file keeps its state", func(t *testing.T) {
		working := filepath.Join(repo, filepath.FromSlash(plan.Path))
		saved, err := os.ReadFile(working)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(working, []byte("no plan\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		entries, states := scratch(t, h)
		if err := os.WriteFile(working, saved, 0o644); err != nil {
			t.Fatal(err)
		}
		unreadable := map[string]string{}
		for _, name := range folders {
			unreadable[name] = "unknown"
		}
		if want := wanted(unreadable); !slices.Equal(states, want) {
			t.Fatalf("with the working plan unreadable the scratch entries are\n%s\nwant\n%s", strings.Join(states, "\n"), strings.Join(want, "\n"))
		}
		for _, name := range folders {
			named(t, entries, name, "unknown", working)
		}
		unchanged(t)
	})

	// Last, since it changes the fixture: a finished folder that holds a link is not finished.
	t.Run("a finished folder that holds a link a recursive delete could follow is unknown", func(t *testing.T) {
		if !fixtureLink(at("outside"), at("repo/tmp/pair-a/out")) {
			t.Log("this machine makes neither a symbolic link nor a junction, so nothing about links is held here")
			return
		}
		entries, states := scratch(t, h)
		if want := wanted(map[string]string{"pair-a": "unknown"}); !slices.Equal(states, want) {
			t.Fatalf("with a link in pair-a the scratch entries are\n%s\nwant\n%s", strings.Join(states, "\n"), strings.Join(want, "\n"))
		}
		named(t, entries, "pair-a", "unknown", "link")
	})
}
