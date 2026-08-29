package evaluation

// hasComparisonPair reports whether a collection contains a first value and
// at least one value to compare with it. Expressing the invariant through the
// tail keeps callers independent of a duplicated numeric cardinality policy.
func hasComparisonPair[T any](values []T) bool {
	return len(values) != 0 && len(values[1:]) != 0
}
