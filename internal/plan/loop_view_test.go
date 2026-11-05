package plan

import (
	"strings"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// TestLoopViewAndStop: a worktree's loop view names the row it works next
// and its last landing, with no supervisor running; a stop from the
// workbench holds it as a user stop while the next row stays shown, and a
// second stop leaves the first in place.
func TestLoopViewAndStop(t *testing.T) {
	t.Parallel()
	document := Plan{Campaign: "loop", Items: []Item{
		{ID: "next", Status: StatusOpen, Steps: []Step{{ID: "do", Title: "Do the next thing", Status: StatusOpen, Verify: "go test ./..."}}},
	}}
	repository := planRepository(t, document)
	git(t, repository, "commit", "-q", "--allow-empty", "-m", "Land the earlier row\n\nCause: fixture.\n\n"+LandedRowTrailer+": earlier/do")
	landed := git(t, repository, "rev-parse", "HEAD")
	git(t, repository, "commit", "-q", "--allow-empty", "-m", "a commit that lands no row")
	view := func() LoopView {
		t.Helper()
		read, err := ReadLoopView(t.Context(), repository)
		testcheck.NoError(t, err)
		return read
	}
	first := view()
	if first.Running || first.Stopped || first.Row != "next/do" || first.RowTitle != "Do the next thing" ||
		first.Landing.Commit != landed || first.Landing.Row != "earlier/do" || first.Landing.Subject != "Land the earlier row" || first.Landing.At == "" {
		t.Fatalf("view = %+v", first)
	}
	// A live supervisor reads as running, and the stop wakes it.
	campaign, err := processcontrol.BeginCampaign(repository, "loop-worker")
	testcheck.NoError(t, err)
	defer campaign.Close()
	if !view().Running {
		t.Fatal("a live supervisor did not read as running")
	}
	testcheck.NoError(t, StopLoop(t.Context(), repository, "stopped from the workbench"))
	select {
	case <-campaign.Stops():
	default:
		t.Fatal("the stop did not wake the live supervisor")
	}
	stopped := view()
	if !stopped.Stopped || stopped.StopReason != "requested: stopped from the workbench" || stopped.Row != "next/do" {
		t.Fatalf("stopped view = %+v", stopped)
	}
	held, err := ReadStop(t.Context(), repository, "", ExecutionUnattended)
	if err != nil || !held.Blocked || held.Event.Worker != workbenchWorker {
		t.Fatalf("stop = %+v, %v", held, err)
	}
	testcheck.NoError(t, StopLoop(t.Context(), repository, "a second stop"))
	if again, err := ReadStop(t.Context(), repository, "", ExecutionUnattended); err != nil || again.ID != held.ID {
		t.Fatalf("a second stop replaced the first: %+v, %v", again, err)
	}
}

// TestAParkedRowWaitsForACommit: dispatch passes over a parked row for the one behind it and says when only that row is
// left, and a commit brings it back.
func TestAParkedRowWaitsForACommit(t *testing.T) {
	t.Parallel()
	row := func(id string) Item {
		return Item{ID: id, Title: id, Status: StatusOpen, Steps: []Step{{ID: "do", Title: id, Status: StatusOpen, Verify: "true"}}}
	}
	repository := planRepository(t, Plan{Campaign: "loop", Items: []Item{row("failing"), row("behind")}})
	resolve := func() Dispatch {
		t.Helper()
		dispatch, err := ResolveDispatch(t.Context(), repository, DispatchRequest{Mode: ExecutionInteractive})
		testcheck.NoError(t, err)
		return dispatch
	}
	if dispatch := resolve(); dispatch.Item != "failing" {
		t.Fatalf("before parking, dispatch = %+v", dispatch)
	}
	for _, parked := range []string{"failing/do", "failing/do"} {
		testcheck.NoError(t, ParkRow(t.Context(), repository, parked))
	}
	if dispatch := resolve(); dispatch.Item != "behind" {
		t.Fatalf("with failing parked, dispatch = %+v", dispatch)
	}
	testcheck.NoError(t, ParkRow(t.Context(), repository, "behind/do"))
	if dispatch := resolve(); dispatch.Item != "" || !strings.Contains(dispatch.Waiting, "parked until a commit: failing/do, behind/do") {
		t.Fatalf("with both parked, dispatch = %+v", dispatch)
	}
	git(t, repository, "commit", "-q", "--allow-empty", "-m", "a commit")
	if dispatch := resolve(); dispatch.Item != "failing" {
		t.Fatalf("after a commit, dispatch = %+v", dispatch)
	}
}
