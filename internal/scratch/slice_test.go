package scratch

import (
	"slices"
	"testing"
)

func TestResizeStorageContract(t *testing.T) {
	storage := []float64{1, 2, 3}
	reused := Resize(storage, 2)
	if &reused[0] != &storage[0] || !slices.Equal(reused, []float64{1, 2}) {
		t.Fatal("shrinking must reuse uncleared storage")
	}
	grown := Resize(reused, 4)
	if len(grown) != 4 || cap(grown) != 4 || &grown[0] == &storage[0] || !slices.Equal(grown, []float64{0, 0, 0, 0}) {
		t.Fatal("growth must allocate exact fresh storage")
	}
	empty := Resize(storage, 0)
	if len(empty) != 0 || cap(empty) != cap(storage) || Resize[byte](nil, 0) != nil {
		t.Fatal("empty resize must preserve capacity without allocation")
	}
}
