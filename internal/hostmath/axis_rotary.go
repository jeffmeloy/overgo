// Axis-partitioned interleaved rotary: one head row splits into contiguous
// per-axis channel spans; each span rotates adjacent pairs (2i, 2i+1) by its
// own axis position over its own inverse-frequency ladder (span-local
// exponent base^(-2i/span)). This is the host mirror of the serving graph's
// per-axis RoPELayoutNormal slices; the backward is rotation by the negated
// angles (the transpose of the orthogonal map).
package hostmath

// AxisRotaryInvFreq: per-span inverse-frequency ladders for the given spans.
func AxisRotaryInvFreq(base float64, spans [3]int) [3][]float64 {
	var out [3][]float64
	for axis, span := range spans {
		out[axis] = RopeInvFreq(base, span)
	}
	return out
}

// axisRotarySpans drives one per-span rotation kernel across the head row.
func axisRotarySpans(row []float32, spans [3]int, invFreq [3][]float64, positions [3]int, rotate func([]float32, []float64, int)) {
	offset := 0
	for axis, span := range spans {
		if span > 0 {
			rotate(row[offset:offset+span], invFreq[axis], positions[axis])
		}
		offset += span
	}
	if offset != len(row) {
		panic("hostmath: axis rotary spans do not cover the head row")
	}
}

// ApplyAxisRotaryInterleaved rotates one head row in place: span[axis]
// channels starting at the running offset rotate by positions[axis] over
// invFreq[axis]. The spans must cover the row exactly.
func ApplyAxisRotaryInterleaved(row []float32, spans [3]int, invFreq [3][]float64, positions [3]int) {
	axisRotarySpans(row, spans, invFreq, positions, ApplyRotaryInterleaved)
}

// AxisRotaryInterleavedBackward: VJP of ApplyAxisRotaryInterleaved, in place
// on the head-row gradient.
func AxisRotaryInterleavedBackward(dRow []float32, spans [3]int, invFreq [3][]float64, positions [3]int) {
	axisRotarySpans(dRow, spans, invFreq, positions, RotaryInterleavedBackward)
}
