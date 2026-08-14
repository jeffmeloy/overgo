package routedlm

import "testing"

func TestPrefixStateRequiresEveryLayer(t *testing.T) {
	state, err := NewPrefixState(2, 1, 2, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if state.Complete() {
		t.Fatal("empty prefix state is complete")
	}
	for layer := range state.Layers {
		if err := state.SetLayer(layer, make([]float32, 4), make([]float32, 4)); err != nil {
			t.Fatal(err)
		}
	}
	if !state.Complete() {
		t.Fatal("populated prefix state is incomplete")
	}
}
