package dtype

// Float64SliceToFloat32: Go narrowing into owned storage.
func Float64SliceToFloat32(values []float64) []float32 {
	return ConvertSlice[float32](values)
}

// ConvertSlice converts a numeric slice element by element into a new
// slice of another numeric type: the one loop behind every typed
// slice conversion in the tree, so a token id list, a float narrowing,
// and an index widening share it rather than each carrying a copy.
func ConvertSlice[To, From ~int | ~int32 | ~uint32 | ~float32 | ~float64](values []From) []To {
	converted := make([]To, len(values))
	for index, value := range values {
		converted[index] = To(value)
	}
	return converted
}
