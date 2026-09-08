package speechrecognition

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"slices"
	"strconv"
	"testing"

	"overgo/internal/binaryschema"
)

func TestCTCAlignmentOracle(t *testing.T) {
	data, err := os.ReadFile("testdata/ctc_alignment.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Commit string
		Cases  []struct {
			Name                      string
			Frames, Vocabulary, Blank int
			Targets, States           []int
			Emissions                 []float32
			Score                     float32
		}
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Commit != "3497b7cc44753e2c141d8fe60ac42cec433e3281" || len(fixture.Cases) != 5 {
		t.Fatal("missing pinned oracle cases")
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			var workspace CTCAlignmentWorkspace
			_, _, budget, err := alignmentStorage(test.Frames, len(test.Targets))
			if err != nil {
				t.Fatal(err)
			}
			for range 2 {
				got, err := workspace.Align(t.Context(), test.Emissions, test.Targets, test.Frames, test.Vocabulary, test.Blank, budget)
				if err != nil || !slices.Equal(got.States, test.States) || math.Float32bits(got.Score) != math.Float32bits(test.Score) {
					t.Fatalf("path=%v score=%g; oracle=%v %g: %v", got.States, got.Score, test.States, test.Score, err)
				}
			}
			allocations := testing.AllocsPerRun(10, func() {
				if _, err := workspace.Align(t.Context(), test.Emissions, test.Targets, test.Frames, test.Vocabulary, test.Blank, budget); err != nil {
					t.Fatal(err)
				}
			})
			if allocations != 0 {
				t.Fatalf("admitted workspace reuse allocated %g times", allocations)
			}
		})
	}
}

func TestCTCAlignmentRefusals(t *testing.T) {
	for _, test := range []struct {
		name                      string
		emissions                 []float32
		targets                   []int
		frames, vocabulary, blank int
		budget                    uint64
	}{
		{"empty-target", []float32{0, 0}, nil, 1, 2, 0, 128},
		{"blank-target", []float32{0, 0}, []int{0}, 1, 2, 0, 128},
		{"unknown-target", []float32{0, 0}, []int{2}, 1, 2, 0, 128},
		{"unknown-blank", []float32{0, 0}, []int{1}, 1, 2, 2, 128},
		{"impossible-repeat", []float32{0, 0, 0, 0}, []int{1, 1}, 2, 2, 0, 128},
		{"short-emissions", []float32{0}, []int{1}, 1, 2, 0, 128},
		{"zero-frames", nil, []int{1}, 0, 2, 0, 128},
		{"overflow", nil, []int{1}, math.MaxInt, 2, 0, 128},
		{"nan", []float32{0, float32(math.NaN())}, []int{1}, 1, 2, 0, 128},
		{"positive-infinity", []float32{0, float32(math.Inf(1))}, []int{1}, 1, 2, 0, 128},
		{"positive-emission", []float32{0, 1}, []int{1}, 1, 2, 0, 128},
		{"unreachable-target", []float32{0, float32(math.Inf(-1))}, []int{1}, 1, 2, 0, 128},
		{"byte-admission", []float32{0, 0}, []int{1}, 1, 2, 0, 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			var workspace CTCAlignmentWorkspace
			if _, err := workspace.Align(t.Context(), test.emissions, test.targets, test.frames, test.vocabulary, test.blank, test.budget); err == nil {
				t.Fatal("invalid alignment accepted")
			}
		})
	}
	var workspace CTCAlignmentWorkspace
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(nil)
	if _, err := workspace.Align(ctx, []float32{0, 0}, []int{1}, 1, 2, 0, 128); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation=%v", err)
	}
	if _, err := workspace.Align(nil, []float32{0, 0}, []int{1}, 1, 2, 0, 128); err == nil {
		t.Fatal("nil context accepted")
	}
	if _, err := (*CTCAlignmentWorkspace)(nil).Align(t.Context(), []float32{0, 0}, []int{1}, 1, 2, 0, 128); err == nil {
		t.Fatal("nil workspace accepted")
	}
	for _, shape := range [][2]int{{0, 1}, {1, 0}, {1, math.MaxInt}, {math.MaxInt, 1}} {
		if _, _, _, err := alignmentStorage(shape[0], shape[1]); err == nil {
			t.Fatalf("invalid storage geometry accepted: %v", shape)
		}
	}
}

func TestCTCAlignmentWorkspaceAdmission(t *testing.T) {
	var workspace CTCAlignmentWorkspace
	for _, shape := range [][2]int{{32, 16}, {100, 2}, {1, 1}} {
		frames, count := shape[0], shape[1]
		targets := make([]int, count)
		for i := range targets {
			targets[i] = 1 + i%2
		}
		_, _, budget, err := alignmentStorage(frames, count)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := workspace.Align(t.Context(), make([]float32, frames*3), targets, frames, 3, 0, budget); err != nil {
			t.Fatal(err)
		}
		retained := uint64(cap(workspace.scores))*binaryschema.Uint32Bytes + uint64(cap(workspace.moves)) + uint64(cap(workspace.path))*(strconv.IntSize/8)
		if retained > budget {
			t.Fatalf("retained %d exceeds %d", retained, budget)
		}
	}
	if _, err := workspace.Align(t.Context(), []float32{0, 0}, workspace.path, 1, 2, 0, 128); err == nil {
		t.Fatal("target/path overlap accepted")
	}
	_, _, budget, err := alignmentStorage(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	workspace.discardExcess(budget)
	if cap(workspace.path) != 1 {
		t.Fatal("fitting workspace was discarded")
	}
	workspace.discardExcess(budget - 1)
	if cap(workspace.path) != 0 || cap(workspace.scores) != 0 || cap(workspace.moves) != 0 {
		t.Fatal("prior workspace exceeds composing reservation")
	}
}
