package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/scorecard"
)

// TestLoopStartsACycleWhenTheMeasuredSurfaceMoved holds the trigger to what moves a cycle's
// measurement: a change to a non-test Go file in the closure of a command a cycle runs, or to the
// kernels, counts as a landing that moved it, and a test, a doc, a package outside the closure
// does not; no cycle is due while nothing moved, so a failing cycle is not started again for the
// code it already judged, nor before the landings that moved the surface reach the limit; the
// limit starts one, once for a commit; and an empty plan runs a cycle only when something moved.
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
	surface := func(context.Context, string) (map[string]bool, error) {
		return map[string]bool{"internal/evaluation": true, "cmd/train": true}, nil
	}
	world := &improvement{watch: true, root: repository, records: records, surface: surface, limit: 2}
	check := func(name string, wantDue, wantWorth bool) {
		t.Helper()
		due, reason, err := world.CycleDue()
		if err != nil || due != wantDue {
			t.Fatalf("%s: a cycle is due = %t (%s, %v), want %t", name, due, reason, err, wantDue)
		}
		if worth, reason, err := world.CycleWorthRunning(); err != nil || worth != wantWorth {
			t.Fatalf("%s: an empty plan's cycle is worth running = %t (%s, %v), want %t", name, worth, reason, err, wantWorth)
		}
	}
	check("nothing landed since the cycle", false, false)

	land("docs/notes.md", "a doc")
	land("internal/evaluation/a_test.go", "a test")
	land("internal/server/x.go", "a package outside the closure")
	check("landings that move nothing a cycle measures", false, false)

	land("internal/evaluation/a.go", "moves the surface once")
	check("one landing that moved it, below the limit", false, true)

	land("kernels/k.cu", "moves it again through a kernel")
	check("the second landing reaches the limit", true, true)
	if due, _, err := world.CycleDue(); err != nil || due {
		t.Fatalf("a cycle was due twice for one commit (%t, %v), want it begun once", due, err)
	}

	// A cycle that completed at HEAD leaves nothing moved, whatever its outcome.
	if _, err := scorecard.Publish(records, time.Date(2026, time.October, 5, 13, 0, 0, 0, time.UTC), scorecard.Record{Commit: git("rev-parse", "HEAD")}); err != nil {
		t.Fatal(err)
	}
	world.attempted = ""
	check("the cycle that completed at HEAD", false, false)

	// No limit set: only an empty plan starts one, and only when something moved.
	world.limit = 0
	land("cmd/train/main.go", "moves the surface")
	check("no limit", false, true)
}

// TestDecideCycleNamesItsReason holds the decision's words: each outcome says why, for the
// operator who reads the loop's output.
func TestDecideCycleNamesItsReason(t *testing.T) {
	t.Parallel()
	for name, test := range map[string]struct {
		last         string
		moved, limit int
		due          bool
	}{
		"no cycle yet":         {"", 5, 3, false},
		"nothing moved":        {"abcdef0123456789", 0, 3, false},
		"below the limit":      {"abcdef0123456789", 2, 3, false},
		"at the limit":         {"abcdef0123456789", 3, 3, true},
		"no limit set":         {"abcdef0123456789", 9, 0, false},
		"above the limit":      {"abcdef0123456789", 7, 3, true},
		"one landing, limit 1": {"abcdef0123456789", 1, 1, true},
	} {
		if got := decideCycle(test.last, test.moved, test.limit); got.Due != test.due || got.Reason == "" {
			t.Errorf("%s: decideCycle = %+v, want due %t with a reason", name, got, test.due)
		}
	}
}
