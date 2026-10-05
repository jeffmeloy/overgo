package densecausal

import (
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/hostmath"
)

// TestHeadBackwardWindowsAddUpToTheSequence holds the head's loss and gradients,
// taken a window of rows at a time, to the whole sequence's taken at once, for
// windows that leave none and some rows over: the loss, the gradient of the
// head's weight and each row's gradient into the stream.
func TestHeadBackwardWindowsAddUpToTheSequence(t *testing.T) {
	g := readTinyGolden(t)
	model := modelFromGolden(t, g)
	hidden, vocab := model.Dims.Hidden, model.Dims.Vocab
	head := model.tensors.head.values
	const rows = 7
	random := rand.New(rand.NewPCG(3, 9))
	normed := make([]float32, rows*hidden)
	for index := range normed {
		normed[index] = random.Float32()*2 - 1
	}
	targets := make([]int, rows)
	for index := range targets {
		targets[index] = random.IntN(vocab)
	}

	// The sequence at once: every row's logits, the mean cross-entropy and its gradient.
	logits, dLogits := make([]float32, rows*vocab), make([]float32, rows*vocab)
	hostmath.Linear(logits, normed, head, rows, hidden, vocab)
	wantLoss := hostmath.SoftmaxCrossEntropy(dLogits, logits, targets, rows, vocab)
	wantDNormed, wantGrad := make([]float32, rows*hidden), make([]float32, len(head))
	hostmath.LinearBackward(wantDNormed, wantGrad, nil, normed, head, dLogits, rows, hidden, vocab, false)

	near := func(got, want []float32, what string, window int) {
		t.Helper()
		for index := range want {
			if math.Abs(float64(got[index]-want[index])) > 1e-5*(1+math.Abs(float64(want[index]))) {
				t.Errorf("window %d: %s[%d] = %v, want %v", window, what, index, got[index], want[index])
				return
			}
		}
	}
	for _, window := range []int{1, 2, 3, rows - 1, rows, rows + 5} {
		gotDNormed, gotGrad := make([]float32, rows*hidden), make([]float32, len(head))
		loss := headBackward(normed, targets, head, gotGrad, gotDNormed, hidden, vocab, window)
		if math.Abs(loss-wantLoss) > 1e-9*(1+math.Abs(wantLoss)) {
			t.Errorf("window %d: loss %v, want %v", window, loss, wantLoss)
		}
		near(gotDNormed, wantDNormed, "dNormed", window)
		near(gotGrad, wantGrad, "head gradient", window)
	}
	if headWindowRows(vocab) < 1 || headWindowRows(1<<30) != 1 {
		t.Errorf("the window holds %d rows of this vocabulary and %d of a vocabulary too wide for it, want at least one", headWindowRows(vocab), headWindowRows(1<<30))
	}
}
