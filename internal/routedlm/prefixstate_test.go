package routedlm

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestPrefixStateRequiresEveryLayer(t *testing.T) {
	state, err := NewPrefixState(2, 1, 2, 2, 2)
	testcheck.NoError(t, err)
	if state.Complete() {
		t.Fatal("empty prefix state is complete")
	}
	for layer := range state.Layers {
		testcheck.NoError(t, state.SetLayer(layer, make([]float32, 4), make([]float32, 4)))
	}
	if !state.Complete() {
		t.Fatal("populated prefix state is incomplete")
	}
}
