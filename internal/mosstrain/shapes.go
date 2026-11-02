package mosstrain

import (
	"fmt"
	"slices"
)

// CheckShape refuses a tensor stored at another shape than shapes, what the transformer's graph reads, gives it. Placed as
// it stands, a config that disagrees with its weights panics or reads past a device buffer. A flat weight states its
// element count alone, and a tensor the graph does not read has no shape to hold.
func CheckShape(shapes map[string][]uint64, name string, stored []uint64) error {
	want, read := shapes[name]
	if !read {
		return nil
	}
	product := func(shape []uint64) uint64 {
		elements := uint64(1)
		for _, extent := range shape {
			elements *= extent
		}
		return elements
	}
	if len(want) == 1 && product(stored) == want[0] || slices.Equal(stored, want) {
		return nil
	}
	return fmt.Errorf("mosstrain: %s is stored at %v, and the model's configuration reads it at %v", name, stored, want)
}
