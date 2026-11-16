package plan

import (
	"bufio"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// TestALandingIsAdmittedAsDispatchGivesItWhileTheCardIsHeld holds admission to the dispatch a worker is given
// (admit-card/do): on 2026-11-16, while the readings campaign held the card's turn, -next gave space-week/do, the first
// step that needs no card, and the gate refused its landing for the card step before it, as admission read neither the
// card's holder nor the parked rows that dispatch reads beside the plan. With the card free a worker is admitted to
// the first step, marked as needing the card, and refused the step after it; while another process holds the card's
// turn, the worker is admitted to the step that needs none and refused the marked one.
func TestALandingIsAdmittedAsDispatchGivesItWhileTheCardIsHeld(t *testing.T) {
	t.Parallel()
	marked := Item{ID: "bridge", Title: "bridge", Steps: []Step{{ID: "capture", Title: "capture", Verify: "true", NeedsCard: true}}}
	host := Item{ID: "host", Title: "host", Steps: []Step{{ID: "do", Title: "do", Verify: "true"}}}
	document := Plan{Campaign: "fixture", Items: []Item{marked, host}}
	repository := planRepository(t, document)
	// The fixture's card is its own, so no card another process holds on the machine is part of the test.
	turn := processcontrol.CardTurnPrefix + "fixture:" + t.Name() + ":" + strconv.Itoa(os.Getpid())
	request := DispatchRequest{Mode: ExecutionInteractive, Worker: "fixture-worker"}
	admitted := func(ref Ref) error {
		t.Helper()
		return admitLanding(t.Context(), repository, document, request, []Ref{ref}, turn)
	}
	card, other := Ref{Item: "bridge", Step: "capture"}, Ref{Item: "host", Step: "do"}
	if err := admitted(card); err != nil {
		t.Errorf("with the card free a worker was refused the first step: %v", err)
	}
	if err := admitted(other); err == nil {
		t.Error("with the card free a worker was admitted to the step after the first")
	}

	holder := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestCardTurnHolder$", "-test.timeout=0")
	holder.Env = append(os.Environ(), cardTurnEnvironment+"="+turn)
	holder.Stderr = os.Stderr
	input := testcheck.Must(holder.StdinPipe())(t)
	output := bufio.NewReader(testcheck.Must(holder.StdoutPipe())(t))
	testcheck.NoError(t, holder.Start())
	if said := strings.TrimSpace(testcheck.Must(output.ReadString('\n'))(t)); said != "taken" {
		t.Fatalf("the holder said %q; want taken", said)
	}
	if err := admitted(other); err != nil {
		t.Errorf("while another process holds the card's turn a worker was refused the step that needs no card: %v", err)
	}
	if err := admitted(card); err == nil || !strings.Contains(err.Error(), "a worker lands only the step dispatch gives it") {
		t.Errorf("while another process holds the card's turn a worker's landing of the card step answered %v; want it refused", err)
	}
	testcheck.NoError(t, input.Close())
	testcheck.NoError(t, holder.Wait())
}
