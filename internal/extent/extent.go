// Package extent owns dependency-neutral shape positions and small cardinalities.
package extent

const (
	// FirstOffset is the first zero-based shape or sequence position.
	FirstOffset = 0
	// SingletonExtent is one element and the multiplicative extent identity.
	SingletonExtent = 1
	// PairedExtent is two elements or rank-two geometry.
	PairedExtent = 2
	// TripleExtent is three elements or rank-three geometry.
	TripleExtent = 3
)
