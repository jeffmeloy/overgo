package mosstrain

import (
	"strings"
	"testing"
)

// TestATensorOfAnotherShapeIsRefused: a tensor that disagrees with what the configuration reads is refused by name, in
// either axis, before it is placed; one that agrees passes, a flat weight by its element count, and one the graph does not
// read has nothing to hold.
func TestATensorOfAnotherShapeIsRefused(t *testing.T) {
	t.Parallel()
	shapes := map[string][]uint64{"matrix": {4, 8}, "flat": {16}}
	for name, read := range map[string]struct {
		tensor string
		stored []uint64
		fails  bool
	}{
		"agrees":         {"matrix", []uint64{4, 8}, false},
		"flat agrees":    {"flat", []uint64{4, 4}, false},
		"not read":       {"other", []uint64{3}, false},
		"wider":          {"matrix", []uint64{4, 9}, true},
		"transposed":     {"matrix", []uint64{8, 4}, true},
		"flat elsewhere": {"flat", []uint64{4, 5}, true},
	} {
		err := CheckShape(shapes, read.tensor, read.stored)
		if read.fails && (err == nil || !strings.Contains(err.Error(), read.tensor)) {
			t.Errorf("%s: %v, want it refused by name", name, err)
		}
		if !read.fails && err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}
