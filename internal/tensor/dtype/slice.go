package dtype

// Float64SliceToFloat32: Go narrowing into owned storage.
func Float64SliceToFloat32(values []float64) []float32 {
	converted := make([]float32, len(values))
	for index, value := range values {
		converted[index] = float32(value)
	}
	return converted
}
