package hostmath

import "slices"

// The linear kernels advance a block of independent outputs together. A dot
// product adds each product to one running sum, so its adds wait on each other;
// a block's sums advance side by side and the operand they share is loaded once
// for all of them. Every output element still adds its terms in the order the
// one-output loop did, so a result is that loop's bit for bit.
const (
	// interleave is how many outputs a block holds.
	interleave = 4
	// laneThird and laneFourth index the block's last two outputs, which the
	// kernels name one by one to keep each sum in a register.
	laneThird  = 2
	laneFourth = 3
)

// block slices the interleave consecutive rows of data, width long, that begin
// at row first.
func block(data []float32, first, width int) (rows [interleave][]float32) {
	for lane := range rows {
		rows[lane] = data[(first+lane)*width : (first+lane+1)*width]
	}
	return rows
}

// column reads the interleave values of data a stride apart that begin at
// index first.
func column(data []float32, first, stride int) (values [interleave]float32) {
	for lane := range values {
		values[lane] = data[first+lane*stride]
	}
	return values
}

// dot4 returns the dot products of w with each row of a block, as long as w.
func dot4(w []float32, rows [interleave][]float32) (sums [interleave]float32) {
	x0, x1, x2, x3 := rows[0][:len(w)], rows[1][:len(w)], rows[laneThird][:len(w)], rows[laneFourth][:len(w)]
	var s0, s1, s2, s3 float32
	for index, value := range w {
		s0 += value * x0[index]
		s1 += value * x1[index]
		s2 += value * x2[index]
		s3 += value * x3[index]
	}
	return [interleave]float32{s0, s1, s2, s3}
}

// axpy adds gradient times source into destination, which a zero gradient
// leaves untouched.
func axpy(destination []float32, gradient float32, source []float32) {
	if gradient == 0 {
		return
	}
	source = source[:len(destination)]
	for index := range destination {
		destination[index] += gradient * source[index]
	}
}

// axpy4 adds the block's gradients times its rows into destination in lane
// order, each element once loaded and once stored; the gradients are nonzero,
// which axpy would skip.
func axpy4(destination []float32, gradients [interleave]float32, rows [interleave][]float32) {
	a, b, c, d := rows[0][:len(destination)], rows[1][:len(destination)], rows[laneThird][:len(destination)], rows[laneFourth][:len(destination)]
	g0, g1, g2, g3 := gradients[0], gradients[1], gradients[laneThird], gradients[laneFourth]
	for index := range destination {
		value := destination[index]
		value += g0 * a[index]
		value += g1 * b[index]
		value += g2 * c[index]
		value += g3 * d[index]
		destination[index] = value
	}
}

// axpyShared4 adds each of the block's gradients times source into the
// destination of its lane, the source loaded once for all four; the gradients
// are nonzero, which axpy would skip.
func axpyShared4(source []float32, destinations [interleave][]float32, gradients [interleave]float32) {
	d0, d1, d2, d3 := destinations[0][:len(source)], destinations[1][:len(source)], destinations[laneThird][:len(source)], destinations[laneFourth][:len(source)]
	g0, g1, g2, g3 := gradients[0], gradients[1], gradients[laneThird], gradients[laneFourth]
	for index, value := range source {
		d0[index] += g0 * value
		d1[index] += g1 * value
		d2[index] += g2 * value
		d3[index] += g3 * value
	}
}

// anyZero reports whether a block holds a zero gradient, which the one-row loops
// skip and a block must too.
func anyZero(gradients [interleave]float32) bool { return slices.Contains(gradients[:], 0) }
