package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/plan"
	"overgo/internal/processlock"
	"overgo/internal/scorecard"
	"overgo/internal/testutil"
)

// TestLoopStartsACycleWhenTheMeasuredSurfaceMoved holds the trigger to what moves a cycle's
// measurement: a change to a non-test Go file in the closure of a command a cycle runs, to a file
// one of those packages embeds by name (changed, deleted or renamed), or to the kernels, counts as
// a landing that moved it, and a test, a doc (beside an embedded file or not), a package outside
// the closure does not; and an empty plan runs a cycle only when something moved, so a failing
// cycle is not started again for the code it already judged.
func TestLoopStartsACycleWhenTheMeasuredSurfaceMoved(t *testing.T) {
	t.Parallel()
	repository, records := t.TempDir(), filepath.Join(t.TempDir(), "scorecards")
	git := func(arguments ...string) string { return gitFixture(t, repository, arguments...) }
	land := func(file, message string) {
		t.Helper()
		path := filepath.Join(repository, filepath.FromSlash(file))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(message+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		git("add", "-A")
		git("commit", "-q", "-m", message)
	}
	git("init", "-q", "-b", "master")
	land("internal/evaluation/a.go", "measured: first")
	measured := git("rev-parse", "HEAD")
	if _, err := scorecard.Publish(records, time.Date(2026, time.October, 5, 12, 0, 0, 0, time.UTC), scorecard.Record{Commit: measured}); err != nil {
		t.Fatal(err)
	}
	// The closure as go list reports it: each package, and beside it the files it embeds by name.
	surface := func(context.Context, string) (map[string]bool, error) {
		return map[string]bool{
			"internal/evaluation": true, "cmd/train": true,
			"internal/model": true, "internal/model/architecture_profiles.json": true, "internal/model/architectures.json": true, "internal/model/space name.json": true,
			"internal/trainingprogram": true, "internal/trainingprogram/optimizer_policy.json": true,
		}, nil
	}
	world := &improvement{watch: true, root: repository, records: records, surface: surface}
	check := func(name string, wantMoved int) {
		t.Helper()
		if _, moved, err := world.movedSince(); err != nil || moved != wantMoved {
			t.Fatalf("%s: %d landings moved what a cycle measures (%v), want %d", name, moved, err, wantMoved)
		}
		if worth, reason, err := world.CycleWorthRunning(); err != nil || worth != (wantMoved > 0) {
			t.Fatalf("%s: an empty plan's cycle is worth running = %t (%s, %v), want %t", name, worth, reason, err, wantMoved > 0)
		}
	}
	check("nothing landed since the cycle", 0)

	land("docs/notes.md", "a doc")
	land("internal/evaluation/a_test.go", "a test")
	land("internal/server/x.go", "a package outside the closure")
	check("landings that move nothing a cycle measures", 0)

	land("internal/evaluation/a.go", "moves the surface once")
	check("one landing that moved it", 1)

	land("kernels/k.cu", "moves it again through a kernel")
	check("a kernel moves it too", 2)

	// A cycle that completed at HEAD leaves nothing moved, whatever its outcome.
	if _, err := scorecard.Publish(records, time.Date(2026, time.October, 5, 13, 0, 0, 0, time.UTC), scorecard.Record{Commit: git("rev-parse", "HEAD")}); err != nil {
		t.Fatal(err)
	}
	check("the cycle that completed at HEAD", 0)
	land("cmd/train/main.go", "moves the surface")
	check("a command a cycle runs", 1)

	// A file a closure package embeds by name moves the surface as its Go does, once a landing,
	// and a doc beside it does not. The deletion and the rename hold the selector alone; in a
	// tree that loads, the same landing also edits the directive that named the file.
	if _, err := scorecard.Publish(records, time.Date(2026, time.October, 5, 14, 0, 0, 0, time.UTC), scorecard.Record{Commit: git("rev-parse", "HEAD")}); err != nil {
		t.Fatal(err)
	}
	land("internal/model/notes.md", "a doc beside an embedded input")
	check("a doc beside an embedded input", 0)
	land("internal/model/architecture_profiles.json", "the architecture catalog changes")
	check("one embedded input changed", 1)
	land("internal/trainingprogram/optimizer_policy.json", "the optimizer policy changes")
	git("rm", "-q", "internal/trainingprogram/optimizer_policy.json")
	git("commit", "-q", "-m", "the optimizer policy is deleted")
	check("a change and a deletion more", 3)
	git("mv", "internal/model/architecture_profiles.json", "internal/model/architectures.json")
	git("commit", "-q", "-m", "the architecture catalog is renamed")
	check("the rename", 4)
	land("internal/model/space name.json", "an embedded path containing spaces")
	closure, _ := surface(t.Context(), repository)
	if _, moved, err := landingsThatMovedTheSurface(t.Context(), repository, records, closure); err != nil || moved != 5 {
		t.Fatalf("embedded paths must remain whole: moved %d landings, want 5 (%v)", moved, err)
	}
}

// TestCycleSurfaceNamesTheFilesItsCommandsEmbed holds the closure go list reports to name, beside each
// package, the files its non-test sources embed by name, a production input under testdata among
// them, and nothing a test embeds: the fixture is a module of its own whose commands depend on
// one package that embeds, so the answer is the Go tool's and not a map a test wrote.
func TestCycleSurfaceNamesTheFilesItsCommandsEmbed(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	testutil.WriteTextFile(t, root, "go.mod", "module overgo\n\ngo 1.26\n")
	for _, command := range surfaceCommands {
		testutil.WriteTextFile(t, root, command+"/main.go", "package main\n\nimport _ \"overgo/internal/model\"\n\nfunc main() {}\n")
	}
	// Both sources import embed: the Go tool reads a directive only in a file that does, so a
	// test file without it would keep its embed out for the wrong reason.
	testutil.WriteTextFile(t, root, "internal/model/catalog.go", "package model\n\nimport \"embed\"\n\n//go:embed architecture_profiles.json\nvar catalog []byte\n\n//go:embed webui\nvar webui embed.FS\n\n//go:embed assets/*.json\nvar assets embed.FS\n\n//go:embed all:private\nvar private embed.FS\n\n//go:embed testdata/profile.json\nvar profile []byte\n")
	testutil.WriteTextFile(t, root, "internal/model/catalog_test.go", "package model\n\nimport _ \"embed\"\n\n//go:embed testdata/oracle.json\nvar oracle []byte\n")
	for _, name := range []string{"architecture_profiles.json", "testdata/profile.json", "testdata/oracle.json", "webui/composer.js", "webui/keep.js", "webui/space name.js", "webui/_ignored.js", "assets/old.json", "assets/keep.json", "assets/notes.md", "private/_kept.js", "notes.md"} {
		testutil.WriteTextFile(t, root, "internal/model/"+name, "{}\n")
	}
	closure, err := surfaceClosure(t.Context(), root)
	if err != nil {
		t.Fatal(err)
	}
	for file, want := range map[string]bool{
		"internal/model/webui/composer.js":          true,
		"internal/model/catalog.go":                 true,
		"kernels/cuda/ops.cu":                       true,
		"internal/model/architecture_profiles.json": true,
		"internal/model/testdata/profile.json":      true,
		"internal/model/testdata/oracle.json":       false,
		"internal/model/catalog_test.go":            false,
		"internal/model/notes.md":                   false,
		"docs/notes.md":                             false,
	} {
		if got := touchesSurface(root, file, closure); got != want {
			t.Errorf("%s moves what a cycle measures = %t, want %t (closure %v)", file, got, want, closure)
		}
	}
	for _, change := range []struct{ name, old, next string }{
		{"delete", "webui/composer.js", ""},
		{"rename", "webui/keep.js", "webui/renamed.js"},
		{"glob deletion", "assets/old.json", ""},
	} {
		old := filepath.Join(root, "internal/model", change.old)
		if change.next == "" {
			err = os.Remove(old)
		} else {
			err = os.Rename(old, filepath.Join(root, "internal/model", change.next))
		}
		if err != nil {
			t.Fatal(err)
		}
		closure, err = surfaceClosure(t.Context(), root)
		if err != nil {
			t.Fatal(err)
		}
		if !touchesSurface(root, "internal/model/"+change.old, closure) {
			t.Errorf("%s of an embedded directory child was excluded", change.name)
		}
	}

}

// TestAHandStartedRunWaitsForTheTrigger holds -improve, started by hand, to the trigger's terms with the dispatched step
// beside them: a cycle is due when none has completed, never while nothing it measures moved, and once something moved
// only when the plan is empty; a dispatched step starts none, a fix its own check
// judges nor a decline its own model's cells judge, and a cycle a run left unfinished continues whatever the plan holds.
func TestAHandStartedRunWaitsForTheTrigger(t *testing.T) {
	t.Parallel()
	repository, records := t.TempDir(), filepath.Join(t.TempDir(), "scorecards")
	git := func(arguments ...string) string { return gitFixture(t, repository, arguments...) }
	land := func(file, message string) {
		t.Helper()
		testutil.WriteTextFile(t, repository, file, message+"\n")
		git("add", "-A")
		git("commit", "-q", "-m", message)
	}
	surface := func(context.Context, string) (map[string]bool, error) {
		return map[string]bool{"internal/evaluation": true}, nil
	}
	world := &improvement{watch: true, root: repository, records: records, surface: surface}
	fix := plan.Dispatch{Item: "run-lanes", Step: "do", Verify: "go test ./internal/evaluation -run '^TestTheFix$'"}
	gap := plan.Dispatch{Item: "scorecard-regressed-x", Step: proposalStep, Verify: "go run ./cmd/validate -scorecard -gap scorecard-regressed-x -target 1"}
	check := func(name string, dispatch plan.Dispatch, want bool) {
		t.Helper()
		due, reason, err := world.HandStartDue(dispatch)
		if err != nil || due != want || reason == "" {
			t.Fatalf("%s: a hand-started cycle is due = %t (%q, %v), want %t with a reason", name, due, reason, err, want)
		}
	}
	git("init", "-q", "-b", "master")
	land("internal/evaluation/a.go", "measured: first")
	check("no cycle has completed", fix, true)

	if _, err := scorecard.Publish(records, time.Date(2026, time.October, 31, 6, 0, 0, 0, time.UTC), scorecard.Record{Commit: git("rev-parse", "HEAD")}); err != nil {
		t.Fatal(err)
	}
	check("nothing moved since the cycle, a gap dispatched", gap, false)

	land("internal/evaluation/a.go", "a fix to what the cycle measures")
	check("a fix its own check judges", fix, false)
	check("a decline its own model's cells judge", gap, false)
	check("an empty plan", plan.Dispatch{Complete: true}, true)

	testutil.WriteTextFile(t, repository, filepath.Join(processlock.StateDirectory, cycleCreditFile), `{"Commit":"`+git("rev-parse", "HEAD")+`"}`)
	check("a cycle a run left unfinished", fix, true)
}
