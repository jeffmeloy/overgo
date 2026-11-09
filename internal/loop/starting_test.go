package loop

import (
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// startingWorld is a ladderWorld whose steps start at a rung of their own, by item.
type startingWorld struct {
	ladderWorld
	starts map[string]int
}

func (w *startingWorld) StartLevel(step Step) int { return w.starts[step.Item] }

// TestTheLocalWorkerGoesFirstWhereItsReplayPays holds the driver to the rung each step starts at: a step whose kind the
// local worker's replays pay for starts with it, and climbs to the Claude worker when it cannot pass; a step of a kind
// they do not pay for starts with the Claude worker, the local worker never tried on it.
func TestTheLocalWorkerGoesFirstWhereItsReplayPays(t *testing.T) {
	t.Parallel()
	world := &startingWorld{
		ladderWorld: ladderWorld{
			passes:    map[string][]bool{"paying": {false, true}, "costly": {true, true}, "easy": {true, true}},
			fakeWorld: fakeWorld{steps: []Step{{Item: "paying", ID: "do"}, {Item: "costly", ID: "do"}, {Item: "easy", ID: "do"}}},
		},
		starts: map[string]int{"paying": 0, "costly": 1, "easy": 0},
	}
	outcome := testcheck.Must(Run(world, Config{MaxAttemptsPerStep: 2, MaxInvocations: 10}))(t)
	want := []string{"paying:0", "paying:0", "paying:1", "costly:1", "easy:0"}
	if !slices.Equal(world.used, want) || outcome.Reason != ReasonPlanComplete {
		t.Fatalf("the runs went %v, ending %+v; want %v", world.used, outcome, want)
	}
}
