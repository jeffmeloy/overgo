package main

import (
	"bytes"
	"slices"
	"testing"

	"overgo/internal/quietmachine"
	"overgo/internal/testcheck"
)

// TestALaneStepSharesTheQuietMachineBeforeTheCard holds a run of the cycle's lanes to the quiet machine's order
// (moss-first-run/quiet): each lane's step takes the machine's shared side first, the card's turn inside it for a lane
// on the card, as a commit's checks take them, and its commands are marked held, so a workload its test starts takes no
// side of its own; a run that shares nothing marks nothing. A timing of the workflow holding the machine alone waited
// for the card the audio lane held, while the two evaluate commands the lane's test had started waited for the timing,
// on 2026-11-21: the lane had taken the card and its workloads the machine after it, the timing the machine and then
// the card.
func TestALaneStepSharesTheQuietMachineBeforeTheCard(t *testing.T) {
	t.Parallel()
	root, _ := fixtureRepository(t)
	var held []string
	lanes := Plan{
		Suites: []SelectedSuite{
			{Lane: "card", Arguments: []string{"env", "GOFLAGS"}, Environment: []string{"OVERGO_CUDA_TEST=1"}},
			{Lane: "host", Arguments: []string{"env", "GOFLAGS"}},
		},
		HoldsCard: func(purpose string, work func() error) error {
			held = append(held, "card for "+purpose)
			return work()
		},
		SharesMachine: func(purpose string, work func() error) error {
			held = append(held, "machine for "+purpose)
			return work()
		},
	}
	var out bytes.Buffer
	testcheck.NoError(t, executeRun(t.Context(), &out, root, "C:/stores/selected", nil, lanes, scopeLanes))
	if want := []string{"machine for the cycle lane card", "card for the cycle lane card", "machine for the cycle lane host"}; !slices.Equal(held, want) {
		t.Errorf("the lanes' steps took %q; want %q", held, want)
	}
	marked := quietmachine.HeldEnvironment + "=1"
	if environment := stepEnvironment(lanes, []string{"OVERGO_CUDA_TEST=1"}); !slices.Equal(environment, []string{"OVERGO_CUDA_TEST=1", marked}) {
		t.Errorf("a step that shares the machine runs its command under %q; want the lane's environment and %s", environment, marked)
	}
	if environment := stepEnvironment(Plan{}, []string{"OVERGO_CUDA_TEST=1"}); slices.Contains(environment, marked) {
		t.Errorf("a step that shares nothing runs its command under %q; want it unmarked", environment)
	}
}
