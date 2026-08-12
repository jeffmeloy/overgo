package densecausal

import "testing"

// TestTrainBF16SGDLossDecreases runs the BF16-master stochastic-SGD training
// path on the tiny llama golden model and pins that the causal-LM loss falls --
// the optimizer half of the at-scale memory path drives a real model's loss
// down with a 2-byte master and no momentum state.
func TestTrainBF16SGDLossDecreases(t *testing.T) {
	g := readGolden(t, "llama_train_golden.json", "llama_train_golden/v1")
	m := modelFromGolden(t, g)
	// Explicit lr: the derived (Muon-tuned n^-1/2) rate is conservative for plain
	// SGD. Overfitting one fixed batch must drive the loss down decisively.
	const steps = 60
	traj, err := m.TrainBF16SGD(g.Tokens, steps, 1.0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(traj) != steps {
		t.Fatalf("want %d losses, got %d", steps, len(traj))
	}
	first, last := traj[0], traj[len(traj)-1]
	t.Logf("bf16-sgd loss %.5f -> %.5f over %d steps", first, last, len(traj))
	if !(last < first) {
		t.Fatalf("loss did not decrease: %.5f -> %.5f", first, last)
	}
	if last > 0.5*first {
		t.Fatalf("loss decreased too little: %.5f -> %.5f (want < 0.5x)", first, last)
	}
}
