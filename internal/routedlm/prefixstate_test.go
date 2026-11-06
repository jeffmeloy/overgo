package routedlm

import (
	"testing"

	"overgo/internal/testcheck"
)

func TestPrefixStateRequiresEveryLayer(t *testing.T) {
	state := testcheck.Must(NewPrefixState(2, 1, 2, 2, 2))(t)
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
