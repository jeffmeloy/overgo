package main

import (
	"bytes"
	"slices"
	"testing"

	"overgo/internal/testcheck"
)

// TestACardLaneRunsInTheCardsTurn holds a run of the cycle's lanes to holding the card's turn while a lane on the card
// runs, and only then (cycle-lanes/lane-turn): a direct run of the lanes beside a Wan render held none, so each of its
// card tests created its context in the turn alone and waited out one of the render's pictures, every diffusionimage
// test about four minutes of it, where a landing's lanes hold the turn once.
func TestACardLaneRunsInTheCardsTurn(t *testing.T) {
	t.Parallel()
	root, _ := fixtureRepository(t)
	var held []string
	lanes := Plan{
		Suites: []SelectedSuite{
			{Lane: "card", Arguments: []string{"env", "GOFLAGS"}, Environment: []string{"OVERGO_CUDA_TEST=1"}},
			{Lane: "host", Arguments: []string{"env", "GOFLAGS"}},
		},
		HoldsCard: func(purpose string, work func() error) error {
			held = append(held, purpose)
			return work()
		},
	}
	var out bytes.Buffer
	testcheck.NoError(t, executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lanes, scopeLanes))
	if !slices.Equal(held, []string{"the cycle lane card"}) {
		t.Fatalf("the run held the card's turn for %q; want the card lane's alone", held)
	}
}
