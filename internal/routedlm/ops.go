package routedlm

import (
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// BF16Matrix: row-major [Out, In] bf16 weight storage; linear expands
// exactly and accumulates in f64 (identical arithmetic to materializing f32).
type BF16Matrix struct {
	Data    []uint16
	In, Out int
}

func (m BF16Matrix) empty() bool { return len(m.Data) == 0 }

// linearRounded: out = bf16(x * W^T) — the reference projection discipline
// (f64-accumulate linear followed by bf16 rounding of the outputs).
func linearRounded(out, x []float32, w BF16Matrix, rows int) {
	hostmath.LinearBF16F64(out, x, w.Data, nil, rows, w.In, w.Out)
	bf16RoundSlice(out)
}

func bf16RoundSlice(values []float32) {
	for i, v := range values {
		values[i] = dtype.RoundBF16(v)
	}
}

// rmsNormRounded: out = bf16(rmsnorm(x) * weight) per row.
func rmsNormRounded(out, x, weight []float32, rows, d int, eps float64) {
	hostmath.RMSNormInto(out, x, weight, rows, d, eps)
	bf16RoundSlice(out)
}

func silu64(x float64) float64 {
	return x / (1.0 + math.Exp(-x))
}

// applyRotaryHalfBF16: half-split rotation with the reference bf16 stepping:
// the angle is the f32 product pos*invFreq[i]; cos/sin and every product and
// sum are bf16-rounded.
func applyRotaryHalfBF16(row []float32, invFreq []float64, pos int) {
	half := len(row) / 2
	for i := 0; i < half; i++ {
		ang := float64(float32(pos) * float32(invFreq[i]))
		c := dtype.RoundBF16(float32(math.Cos(ang)))
		s := dtype.RoundBF16(float32(math.Sin(ang)))
		a, b := row[i], row[i+half]
		row[i] = dtype.RoundBF16(dtype.RoundBF16(a*c) - dtype.RoundBF16(b*s))
		row[i+half] = dtype.RoundBF16(dtype.RoundBF16(b*c) + dtype.RoundBF16(a*s))
	}
}
