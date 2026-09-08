package speechactivity

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"slices"
	"testing"
)

// Small declared linear maps make every state element observable. The two
// blocks use different contexts, including dilation, without a model fixture.
func testNetwork() *Network {
	identity := affine{weight: []float32{1}, in: 1, out: 1}
	return &Network{
		blocks: []memoryBlock{
			{expand: identity, project: identity, past: []float32{.25, .5}, pastDilation: 2, context: 2},
			{expand: identity, project: identity, past: []float32{.125, .25, .5, 1}, pastDilation: 1, context: 3, residual: true},
		},
		dense: []affine{identity}, output: affine{weight: []float32{.125}, in: 1, out: 1},
		inputWidth: 1, scratchWidth: 1, memoryBytes: 1 << 20, causal: true,
	}
}

func cloneState(state [][]float32) [][]float32 {
	result := make([][]float32, len(state))
	for i := range state {
		result[i] = slices.Clone(state[i])
	}
	return result
}

func TestNetworkChunkAndRestart(t *testing.T) {
	n := testNetwork()
	features := []float32{.25, .5, .125, .75, 1, .375, .5, .25}
	var whole Workspace
	want, wantState, err := n.Evaluate(t.Context(), features, len(features), nil, &whole, nil)
	if err != nil {
		t.Fatal(err)
	}
	for split := 1; split < len(features); split++ {
		var first, restarted Workspace
		left, state, err := n.Evaluate(t.Context(), features[:split], split, nil, &first, nil)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(state)
		if err != nil {
			t.Fatal(err)
		}
		var restored [][]float32
		if err := json.Unmarshal(encoded, &restored); err != nil {
			t.Fatal(err)
		}
		before := cloneState(restored)
		right, final, err := n.Evaluate(t.Context(), features[split:], len(features)-split, restored, &restarted, nil)
		if err != nil {
			t.Fatal(err)
		}
		got := append(slices.Clone(left), right...)
		if !slices.Equal(got, want) || !reflect.DeepEqual(final, wantState) || !reflect.DeepEqual(restored, before) {
			t.Fatalf("split %d changed probabilities, state, or caller-owned restart", split)
		}
	}
	var streaming Workspace
	var state [][]float32
	var got []float32
	for _, value := range features {
		output, next, err := n.Evaluate(t.Context(), []float32{value}, 1, state, &streaming, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, state = append(got, output...), next
	}
	if !slices.Equal(got, want) || !reflect.DeepEqual(state, wantState) {
		t.Fatal("borrowed-state continuation differs")
	}
	if cap(state[0]) != 2 || cap(state[1]) != 3 {
		t.Fatal("state grew beyond declared context")
	}
	allocations := testing.AllocsPerRun(100, func() {
		_, next, err := n.Evaluate(t.Context(), features[:1], 1, state, &streaming, nil)
		if err != nil {
			t.Fatal(err)
		}
		state = next
	})
	// Shared linear dispatch currently allocates closures. Measure its exact
	// call count separately so this assertion detects added state/buffer copies
	// without claiming the underlying primitive is allocation-free.
	commonAllocations := testing.AllocsPerRun(100, func() {
		for _, block := range n.blocks {
			project(streaming.expanded[:1], features[:1], block.expand, 1, true)
			project(streaming.projected[:1], features[:1], block.project, 1, false)
		}
		for _, layer := range n.dense {
			project(streaming.left[:1], features[:1], layer, 1, true)
		}
		project(streaming.output[:1], features[:1], n.output, 1, false)
	})
	if allocations != commonAllocations {
		t.Fatalf("steady allocations = %g, shared linear baseline = %g", allocations, commonAllocations)
	}
	t.Logf("steady allocations: %g shared linear, %g additional state/composition", commonAllocations, allocations-commonAllocations)
}

func TestNetworkRefusals(t *testing.T) {
	for _, name := range []string{"zero", "nil-context", "canceled", "shape", "nan", "cache-count", "cache-width", "cache-nan", "budget", "owner", "noncausal", "overlap"} {
		t.Run(name, func(t *testing.T) {
			n, ctx, input, frames := testNetwork(), t.Context(), []float32{1}, 1
			var w Workspace
			var state [][]float32
			switch name {
			case "zero":
				n = &Network{}
			case "nil-context":
				ctx = nil
			case "canceled":
				var cancel context.CancelCauseFunc
				ctx, cancel = context.WithCancelCause(ctx)
				cancel(context.Canceled)
			case "shape":
				frames = 2
			case "nan":
				input[0] = float32(math.NaN())
			case "cache-count":
				state = [][]float32{{1}}
			case "cache-width":
				state = [][]float32{{1, 2, 3}, {}}
			case "cache-nan":
				state = [][]float32{{float32(math.NaN())}, {}}
			case "budget":
				n.memoryBytes = 1
			case "owner":
				w.owner = testNetwork()
			case "noncausal":
				n.causal = false
				state = [][]float32{{}, {}}
			case "overlap":
				state = [][]float32{input, input}
			}
			output, state, err := n.Evaluate(ctx, input, frames, state, &w, nil)
			if err == nil || output != nil || state != nil {
				t.Fatal("invalid execution returned usable output")
			}
		})
	}
}

func TestNetworkCancellationAndStateOwnership(t *testing.T) {
	n := testNetwork()
	var w Workspace
	_, state, err := n.Evaluate(t.Context(), []float32{1}, 1, nil, &w, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := n.Evaluate(t.Context(), state[0], 1, nil, &w, nil); err == nil {
		t.Fatal("accepted input aliasing returned state")
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	calls := 0
	output, result, err := n.Evaluate(ctx, []float32{1}, 1, nil, &w, func(Trace) error { calls++; cancel(context.Canceled); return nil })
	if !errors.Is(err, context.Canceled) || calls != 1 || output != nil || result != nil {
		t.Fatal("cancellation was not honored between operations")
	}
}
