package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/plan"
	"overgo/internal/testcheck"
)

// TestAReplayedCheckRunsAtItsLandedCommitFirst holds a replay to running a step's check where it passes before where it
// fails (replay-bound/do): a command's memory is bounded only once it has passed in the process (processcontrol's job
// bound), and the replay ran its check at the parent first, where by design it fails, so a replay of empty-added-token
// there ran unbounded and took 279 GB of commit on 2026-11-15. The fixture's check says which commit it ran at and passes
// at the landed one alone: a replay runs it there first, then at the parent, then after the worker's attempt; and a step
// whose check fails at its landed commit is no task, said so, with nothing run at the parent.
func TestAReplayedCheckRunsAtItsLandedCommitFirst(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := t.TempDir()
	git := func(arguments ...string) string {
		t.Helper()
		output, err := exec.Command("git", append([]string{"-C", root, "-c", "user.name=fixture", "-c", "user.email=fixture@example.com"}, arguments...)...).CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", arguments, err, output)
		}
		return strings.TrimSpace(string(output))
	}
	commit := func(marker string) string {
		t.Helper()
		testcheck.NoError(t, os.WriteFile(filepath.Join(root, "marker.txt"), []byte(marker+"\n"), 0o644))
		git("add", "marker.txt")
		git("commit", "-q", "-m", marker)
		return git("rev-parse", "HEAD")
	}
	git("init", "-q")
	parent, landed := commit("parent"), commit("landed")
	testcheck.NoError(t, os.WriteFile(filepath.Join(root, dataroot.ConfigFile), []byte("{}"), 0o644))
	ran := filepath.ToSlash(filepath.Join(t.TempDir(), "ran.log"))
	hold := testcheck.Must(holdReplayCheckout(root))(t)
	t.Cleanup(func() { testcheck.NoError(t, hold.lock.Close()) })
	worker := replayWorker{name: "fixture", work: func(context.Context, string, replayTask) (localWork, error) { return localWork{}, nil }}
	step := func(verify string) replayed {
		return replayed{reference: "row/do", landed: landed, parent: parent, step: plan.Step{ID: "do", Title: "do", Verify: verify, Files: []string{"marker.txt"}}}
	}

	record := testcheck.Must(replay(ctx, root, hold.checkout, step("cat marker.txt >> '"+ran+"' && grep -q landed marker.txt"), worker, 1))(t)
	runs := strings.Fields(string(testcheck.Must(os.ReadFile(ran))(t)))
	if want := []string{"landed", "parent", "parent"}; !slices.Equal(runs, want) || record.Skipped != "" || record.Attempts != 1 || record.Passed {
		t.Errorf("the replay ran its check at %v and recorded %+v; want %v, one attempt that left it failing", runs, record, want)
	}

	testcheck.NoError(t, os.Remove(filepath.FromSlash(ran)))
	broken := testcheck.Must(replay(ctx, root, hold.checkout, step("cat marker.txt >> '"+ran+"' && grep -q never marker.txt"), worker, 1))(t)
	runs = strings.Fields(string(testcheck.Must(os.ReadFile(ran))(t)))
	if !slices.Equal(runs, []string{"landed"}) || !strings.Contains(broken.Skipped, "fails at the landed commit") || broken.Attempts != 0 {
		t.Errorf("a step whose check fails at its landed commit ran at %v and recorded %+v; want it run there alone and skipped", runs, broken)
	}
}
