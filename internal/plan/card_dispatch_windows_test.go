package plan

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
)

// cardTurnEnvironment names the card's turn a process of this test binary holds (TestCardTurnHolder).
const cardTurnEnvironment = "OVERGO_TEST_CARD_TURN"

// TestCardTurnHolder is the process TestWhileTheCardIsHeldDispatchGivesAStepThatNeedsNone starts to hold a card's turn:
// it takes the turn, says so, and gives it up when its input closes. It passes at once in any other run.
func TestCardTurnHolder(t *testing.T) {
	t.Parallel()
	turn := os.Getenv(cardTurnEnvironment)
	if turn == "" {
		return
	}
	release := testcheck.Must(processcontrol.TakeTurn(turn, "the fixture's render", os.Stderr))(t)
	fmt.Println("taken")
	_, _ = io.Copy(io.Discard, os.Stdin)
	testcheck.NoError(t, release())
}

// TestWhileTheCardIsHeldDispatchGivesAStepThatNeedsNone holds dispatch to host work while another holds the card
// (card-free-dispatch/do): the card counted held only while a measurement run held it (on Oct 31 from 1:55 AM to
// 11:00 AM, the lead idle 4.9 hours), never while a render, a run of the lanes or a landing held the card's turn, which
// on 2026-11-14 held it for hours. A step needs the card where it is marked so or its check runs on the card. With the
// card free the first step, marked, is dispatched; while another process holds the card's turn, read from its record,
// the first step whose mark and check need none is, past one whose check runs on the card; with only card steps left
// the first of them is (card-order/do), since its code needs no card and its card work queues at the card's turn,
// where dispatch had waited on the holder from 10:58 to 19:32 on 2026-11-16; once the turn is given up the marked step
// is dispatched again. A measurement run holds the card while it has not ended and its process lives.
func TestWhileTheCardIsHeldDispatchGivesAStepThatNeedsNone(t *testing.T) {
	t.Parallel()
	marked := Item{ID: "bridge", Title: "bridge", Steps: []Step{{ID: "capture", Title: "capture", Verify: "true", NeedsCard: true}}}
	checked := Item{ID: "render", Title: "render", Steps: []Step{{ID: "do", Title: "do", Verify: CardCheckSetting + " go test ./internal/p -run '^TestA$'"}}}
	host := Item{ID: "host", Title: "host", Steps: []Step{{ID: "do", Title: "do", Verify: "go test ./internal/p -run '^TestB$'"}}}
	document := Plan{Campaign: "fixture"}
	// The fixture's card is its own, so no card another process holds on the machine is part of the test.
	turn := processcontrol.CardTurnPrefix + "fixture:" + t.Name() + ":" + strconv.Itoa(os.Getpid())
	dispatched := func(items ...Item) Dispatch {
		t.Helper()
		document.Items = items
		document.cardHolder = testcheck.Must(cardHolder(MeasurementRun{}, false, turn))(t)
		return testcheck.Must(selectDispatch(document, ""))(t)
	}
	if free := dispatched(marked, checked, host); free.Item != "bridge" {
		t.Fatalf("with the card free dispatch gave %q; want bridge, the first step", free.Item)
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
	if held := dispatched(marked, checked, host); held.Item != "host" {
		t.Errorf("while another process holds the card's turn dispatch gave %q; want host, the step that needs no card", held.Item)
	}
	if carded := dispatched(marked, checked); carded.Item != "bridge" || carded.Waiting != "" {
		t.Errorf("with only card steps left and the card's turn held, dispatch gave %+v; want bridge, the first of them, and no wait", carded)
	}
	testcheck.NoError(t, input.Close())
	testcheck.NoError(t, holder.Wait())
	if given := dispatched(marked, checked, host); given.Item != "bridge" {
		t.Errorf("once the turn is given up dispatch gave %q; want bridge, the card step", given.Item)
	}

	running := MeasurementRun{Process: os.Getpid(), Started: time.Now()}
	ended := running
	ended.Ended = time.Now()
	if run, over := testcheck.Must(cardHolder(running, true, turn))(t), testcheck.Must(cardHolder(ended, true, turn))(t); run != "the measurement run" || over != "" {
		t.Errorf("a running measurement run reads as holder %q and an ended one %q; want the measurement run, then none", run, over)
	}
}
