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

// TestALandingAcceptsTheStepEitherCardReadingGives holds admission to the steps dispatch gives a worker with the card
// held or free (dispatch-steady/do, after admit-card/do). Admission had followed the one step dispatch gave at the
// landing's own reading of the card. The lanes runner takes the card's turn package by package, so on 2026-11-24 the
// step whose check is on the card was refused while the runner held the turn, and the step that needed none was
// refused minutes later while it held none. Over a first step marked as needing the card and two after it that need
// none, a worker is admitted to the first and the second, with the card free and while another process holds its
// turn, and refused the third either way.
func TestALandingAcceptsTheStepEitherCardReadingGives(t *testing.T) {
	t.Parallel()
	marked := Item{ID: "bridge", Title: "bridge", Steps: []Step{{ID: "capture", Title: "capture", Verify: "true", NeedsCard: true}}}
	host := Item{ID: "host", Title: "host", Steps: []Step{{ID: "do", Title: "do", Verify: "true"}}}
	later := Item{ID: "later", Title: "later", Steps: []Step{{ID: "do", Title: "do", Verify: "true"}}}
	document := Plan{Campaign: "fixture", Items: []Item{marked, host, later}}
	repository := planRepository(t, document)
	// The fixture's card is its own, so no card another process holds on the machine is part of the test.
	turn := processcontrol.CardTurnPrefix + "fixture:" + t.Name() + ":" + strconv.Itoa(os.Getpid())
	request := DispatchRequest{Mode: ExecutionInteractive, Worker: "fixture-worker"}
	hold := func(reading string) {
		t.Helper()
		for _, ref := range []Ref{{Item: "bridge", Step: "capture"}, {Item: "host", Step: "do"}} {
			if err := admitLanding(t.Context(), repository, document, request, []Ref{ref}, turn); err != nil {
				t.Errorf("%s a worker was refused %s: %v", reading, ref, err)
			}
		}
		err := admitLanding(t.Context(), repository, document, request, []Ref{{Item: "later", Step: "do"}}, turn)
		if err == nil || !strings.Contains(err.Error(), "a worker lands only the step dispatch gives it") {
			t.Errorf("%s a worker's landing of the third step answered %v; want it refused", reading, err)
		}
	}
	hold("with the card free")

	holder := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestCardTurnHolder$", "-test.timeout=0")
	holder.Env = append(os.Environ(), cardTurnEnvironment+"="+turn)
	holder.Stderr = os.Stderr
	input := testcheck.Must(holder.StdinPipe())(t)
	output := bufio.NewReader(testcheck.Must(holder.StdoutPipe())(t))
	testcheck.NoError(t, holder.Start())
	if said := strings.TrimSpace(testcheck.Must(output.ReadString('\n'))(t)); said != "taken" {
		t.Fatalf("the holder said %q; want taken", said)
	}
	hold("while another process holds the card's turn")
	testcheck.NoError(t, input.Close())
	testcheck.NoError(t, holder.Wait())
}
