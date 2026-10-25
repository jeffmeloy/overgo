package reference

import (
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

// holdsCells reports whether a node of a cell type is one the reference holds
// as the values its cells round to: a cache of Q8_0 cells fed as an input, or
// appended to, is float data to the reference, rounded as the device rounds it
// (cacheAppend), so a graph reading such a cache runs here unchanged.
func holdsCells(node *tensor.Tensor) bool {
	return node.Type == dtype.Q8_0 && (node.Op == tensor.OpInput || node.Op == tensor.OpCacheAppend)
}
