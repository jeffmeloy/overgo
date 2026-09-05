// Package scratch provides caller-owned reusable working storage.
package scratch

// Resize returns working storage of exactly count elements, reusing capacity
// when possible. Growth allocates exactly count elements and discards prior
// contents. Reused storage is not cleared. The caller must validate count and
// any memory budget before calling; this function does not preserve data.
func Resize[T any](values []T, count int) []T {
	if cap(values) < count {
		return make([]T, count)
	}
	return values[:count]
}
