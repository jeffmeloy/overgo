package densecausal

import (
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/hostmath"
	"overgo/internal/testutil"
)

// TestDenseBlockAdapterLearnsThroughTheProgram holds an adapter trained on a
// dense block through a tensor program, whose backward pass is derived and
// not written, to the model's own block and to what training owes.
//
// The reference. With its up factors at nothing the adapted block is the
// block: for a state, the program's output is the output of the model's own
// layer forward, to within the two arithmetics, on a layer with grouped
// attention.
//
// A controlled task. The target is the block's output under weights moved by
// an update of the adapter's own rank on the projections it adapts, so the
// adapter can reach it. Stepped toward it, the loss falls to a small part of
// where it began, and the adapted block's output moves from the block's
// toward the target's.
//
// Resuming. An adapter that keeps its factors and its optimizer's state
// after some updates, and another over the same block that resumes from
// them, take the same next updates: their factors are the same values, to
// the bit.
func TestDenseBlockAdapterLearnsThroughTheProgram(t *testing.T) {
	const seq, rank, alpha, layerIndex, heads, headWidth = 6, 2, 4.0, 1, 2, 8
	spec := testutil.DenseCausalSpec{Vocab: 32, Hidden: 16, Heads: heads, HeadDim: headWidth, KVHeads: 1, Intermediate: 32, Layers: 2, Seed: 3}
	weights, shapes := testutil.DenseCausalWeights(t, spec)
	model, err := NewModel(weights, shapes, heads, headWidth, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	names := model.layers[layerIndex].names
	random := rand.New(rand.NewPCG(5, 9))
	state := make([]float32, seq*spec.Hidden)
	for index := range state {
		state[index] = float32(random.NormFloat64())
	}
	// through is a state through the model's own forward of the layer.
	through := func(model *Model) []float32 {
		t.Helper()
		out := slices.Clone(state)
		if err := model.layerForward(out, model.layers[layerIndex], hostmath.RopeInvFreq(model.Dims.RopeTheta, model.Dims.HeadDim), seq); err != nil {
			t.Fatal(err)
		}
		return out
	}
	apart := func(left, right []float32) (most float64) {
		for index := range left {
			most = max(most, math.Abs(float64(left[index])-float64(right[index])))
		}
		return most
	}
	adapted := []string{names.Q, names.V}
	adapter, err := newBlockAdapter(model, layerIndex, seq, rank, alpha, adapted, 7)
	if err != nil {
		t.Fatal(err)
	}
	own := through(model)
	started, err := adapter.outputOf(state)
	if err != nil || apart(started, own) > 1e-5 {
		t.Fatalf("the program's block stands %v from the model's own (%v), want them one block", apart(started, own), err)
	}

	// The target: the block under weights moved by an update of the adapter's rank.
	moved := maps.Clone(weights)
	for _, name := range adapted {
		out, in := shapes[name][0], shapes[name][1]
		left, right := make([]float64, out*rank), make([]float64, rank*in)
		for index := range left {
			left[index] = random.NormFloat64() / math.Sqrt(float64(out))
		}
		for index := range right {
			right[index] = random.NormFloat64() / math.Sqrt(float64(in))
		}
		changed := slices.Clone(weights[name])
		for row := range out {
			for column := range in {
				for component := range rank {
					changed[row*in+column] += float32(left[row*rank+component] * right[component*in+column])
				}
			}
		}
		moved[name] = changed
	}
	teacher, err := NewModel(moved, shapes, heads, headWidth, 10000, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	target := through(teacher)
	if apart(target, own) < 1e-2 {
		t.Fatalf("the target stands %v from the block's own output: the task asks nothing", apart(target, own))
	}
	const rate, momentum, updates = 0.05, 0.9, 60
	first, last := 0.0, 0.0
	for update := range updates {
		loss, err := adapter.step(state, target, rate, momentum)
		if err != nil || math.IsNaN(loss) {
			t.Fatalf("update %d: loss %v (%v)", update, loss, err)
		}
		if update == 0 {
			first = loss
		}
		last = loss
	}
	learnt, err := adapter.outputOf(state)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("the loss fell from %.6g to %.6g over %d updates; the adapted block stands %.4g from the target, where the block stood %.4g", first, last, updates, apart(learnt, target), apart(own, target))
	if last > first/10 || apart(learnt, target) > apart(own, target)/2 {
		t.Fatalf("the loss went from %v to %v and the output stands %v from the target, where the block stood %v: want the task learnt", first, last, apart(learnt, target), apart(own, target))
	}

	// Resuming: the same next updates from what was kept.
	fresh := func() *blockAdapter {
		t.Helper()
		made, err := newBlockAdapter(model, layerIndex, seq, rank, alpha, adapted, 7)
		if err != nil {
			t.Fatal(err)
		}
		return made
	}
	const before, after = 3, 4
	whole := fresh()
	for range before {
		if _, err := whole.step(state, target, rate, momentum); err != nil {
			t.Fatal(err)
		}
	}
	factors, kept, err := whole.kept()
	if err != nil {
		t.Fatal(err)
	}
	resumed := fresh()
	if err := resumed.resume(factors, kept, rate, momentum); err != nil {
		t.Fatal(err)
	}
	for range after {
		wholeLoss, wholeErr := whole.step(state, target, rate, momentum)
		resumedLoss, resumedErr := resumed.step(state, target, rate, momentum)
		if wholeErr != nil || resumedErr != nil || wholeLoss != resumedLoss {
			t.Fatalf("after resuming, an update's loss is %v where the uninterrupted adapter's is %v (%v, %v)", resumedLoss, wholeLoss, resumedErr, wholeErr)
		}
	}
	for name, values := range whole.factors {
		if !slices.Equal(resumed.factors[name], values) {
			t.Fatalf("the resumed adapter's %s differs from the uninterrupted one's", name)
		}
	}
	if _, _, err := fresh().kept(); err == nil {
		t.Error("an adapter that took no update kept something to resume from")
	}
}
