package plan

import (
	"os"
	"strings"
	"testing"
	"time"

	"overgo/internal/testcheck"
	"overgo/internal/worklease"
)

// TestDispatchSkipsCardStepsWhileARunHoldsTheCard holds dispatch to host work while a measurement run holds the card:
// on Oct 31 the run held it from 1:55 AM to 11:00 AM and the lead sat idle 4.9 hours with a card landing queued. With
// the card free the first step, which needs it, is dispatched; while a run holds it the host step behind it is; with
// only card steps left dispatch waits and says why, not that the plan is done. A run holds the card while it has not
// ended and its process lives.
func TestDispatchSkipsCardStepsWhileARunHoldsTheCard(t *testing.T) {
	t.Parallel()
	card := Item{ID: "bridge", Title: "bridge", Status: StatusOpen, Steps: []Step{{ID: "capture", Title: "capture", Status: StatusOpen, Verify: "true", NeedsCard: true}}}
	host := Item{ID: "host", Title: "host", Status: StatusOpen, Steps: []Step{{ID: "do", Title: "do", Status: StatusOpen, Verify: "true"}}}
	document := Plan{Campaign: "fixture", Items: []Item{card, host}}
	free := testcheck.Must(selectDispatch(document, worklease.UnassignedRole, ""))(t)
	document.cardHeld = true
	held := testcheck.Must(selectDispatch(document, worklease.UnassignedRole, ""))(t)
	if free.Item != "bridge" || held.Item != "host" {
		t.Fatalf("dispatch gave %s with the card free and %s with it held; want the card step, then the host step", free.Item, held.Item)
	}
	document.Items = []Item{card}
	waiting := testcheck.Must(selectDispatch(document, worklease.UnassignedRole, ""))(t)
	if waiting.Complete || !strings.Contains(waiting.Waiting, "holds the graphics card") {
		t.Fatalf("with only a card step left and the card held, dispatch gave %+v; want it waiting for the card", waiting)
	}

	running := MeasurementRun{Process: os.Getpid(), Started: time.Now()}
	ended := running
	ended.Ended = time.Now()
	if !running.HoldsTheCard() || ended.HoldsTheCard() {
		t.Fatalf("a running run holds the card %v and an ended one %v; want the running one alone", running.HoldsTheCard(), ended.HoldsTheCard())
	}
}
