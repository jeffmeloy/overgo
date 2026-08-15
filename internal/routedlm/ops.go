package routedlm

import (
	"math"

	"overgo/internal/hostmath"
	"overgo/internal/tensor/dtype"
)

// BF16Matrix: row-major [Out, In] BF16. Data serves host math; Raw serves
// device-only streaming without a duplicate word conversion.
type BF16Matrix struct {
	Data    []uint16
	Raw     []byte
	In, Out int
}

func (m BF16Matrix) empty() bool { return len(m.Data) == 0 && len(m.Raw) == 0 }

// linearRounded: out = bf16(x * W^T) — the reference projection discipline
// (f64-accumulate linear followed by bf16 rounding of the outputs).
func linearRounded(out, x []float32, w BF16Matrix, rows int) {
	hostmath.LinearBF16F64(out, x, w.Data, nil, rows, w.In, w.Out)
	dtype.RoundBF16Slice(out)
}

// rmsNormRounded: out = bf16(rmsnorm(x) * weight) per row.
func rmsNormRounded(out, x, weight []float32, rows, d int, eps float64) {
	hostmath.RMSNormInto(out, x, weight, rows, d, eps)
	dtype.RoundBF16Slice(out)
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
