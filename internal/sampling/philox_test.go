package sampling

import "testing"

// TestPositionedDrawsFollowThePosition holds a sampled row's draw to its position (Strata's philox_uniform keyed by
// the row): a sampler that sampled other rows first picks at a position what a fresh one picks there, and the draws at
// distinct positions differ.
func TestPositionedDrawsFollowThePosition(t *testing.T) {
	const (
		seed      = 41
		positions = 64
	)
	logits := []float32{0.3, 0.1, 0.25, 0.2, 0.15, 0.05, 0.35, 0.12}
	config := Config{Temperature: 1, Seed: seed}
	busy, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	history := make([]int, 0, positions)
	draws := map[float64]bool{}
	for position := range positions {
		history = append(history, position%len(logits))
		busied, err := busy.SampleWithHistory(logits, history)
		if err != nil {
			t.Fatal(err)
		}
		fresh, err := New(config)
		if err != nil {
			t.Fatal(err)
		}
		alone, err := fresh.SampleWithHistory(logits, history)
		if err != nil {
			t.Fatal(err)
		}
		if busied != alone {
			t.Fatalf("position %d: a sampler that drew before picks %d, a fresh one %d", position, busied, alone)
		}
		draws[philoxUniform(seed, uint64(position))] = true
	}
	if len(draws) != positions {
		t.Fatalf("%d positions drew %d distinct uniforms", positions, len(draws))
	}
}
