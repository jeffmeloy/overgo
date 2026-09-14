package hostoptimizer

import "overgo/internal/hostmath"

// LinearOutputProjection is the training-side form of
// hostmath.LinearInputProjection: an optional square input projection, the
// output linear and an optional bias. Each output weight row is read once
// against every projected input row, which stays cached, instead of once per
// input row; every dot product keeps the serial order, so the logits are
// bit-identical to hostmath's.
func LinearOutputProjection(dst, projected, x, projection, weight, bias []float32, rows, inDim, outDim int) {
	if projection != nil {
		hostmath.Linear(projected, x, projection, rows, inDim, inDim)
		x = projected
	}
	hostmath.ParallelRangeF64(outDim, rows*inDim, func(oStart, oEnd int) {
		for o := oStart; o < oEnd; o++ {
			wRow := weight[o*inDim : (o+1)*inDim]
			for r := range rows {
				xRow := x[r*inDim : (r+1)*inDim]
				var sum float32
				for c := range wRow {
					sum += wRow[c] * xRow[c]
				}
				dst[r*outDim+o] = sum
			}
		}
	})
	if len(bias) != 0 {
		for row := range rows {
			hostmath.AddBias(dst[row*outDim:(row+1)*outDim], bias)
		}
	}
}
