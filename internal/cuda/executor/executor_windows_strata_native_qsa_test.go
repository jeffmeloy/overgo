//go:build windows

package executor

import (
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
)

// TestStrataNativeQSAMatchesStrata holds Strata's native QSA gate and fused per-head norm and rotation
// (strata_native_qsa.cuh, from native_qsa.cu and native_rope.cu) to float64 transcriptions at Flash-Next's geometry
// within fast math's rounding: the norm and rotation reads each row of a query and gate pair (a stride of two heads),
// norms it by its root mean square plus epsilon and gamma, and rotates its first 64 values as NeoX halves at the row's
// position with theta 1e7^(-2 pair / 64); the gate multiplies each attention value by the sigmoid of its head's gate.
func TestStrataNativeQSAMatchesStrata(t *testing.T) {
	const (
		heads    = 24
		width    = 256
		rotation = 64
		tokens   = 3
		epsilon  = float32(1e-6)
		// bound is the relative L1 distance the kernels keep from the transcriptions, fast math's angles included.
		bound = 1e-5
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(61, 67))
	rows := tokens * heads
	qFull := make([]float32, rows*2*width)
	for index := range qFull {
		qFull[index] = float32(draws.NormFloat64())
	}
	gamma := make([]float32, width)
	for index := range gamma {
		gamma[index] = 1 + 0.1*float32(draws.NormFloat64())
	}
	positions := make([]int32, rows)
	for row := range rows {
		positions[row] = int32(17 + 5*(row/heads))
	}
	qBuffer, gammaBuffer := buffers.upload(t, float32Bytes(qFull)), buffers.upload(t, float32Bytes(gamma))
	positionBuffer, out := buffers.upload(t, int32Bytes(positions)), buffers.zeros(t, 4*rows*width)
	attn := make([]float32, rows*width)
	for index := range attn {
		attn[index] = float32(draws.NormFloat64())
	}
	attnBuffer, gated := buffers.upload(t, float32Bytes(attn)), buffers.zeros(t, 4*rows*width)
	scaling := strataRopeScaling{freqBase: strataFreqBaseDefault, factor: 1, attnFactor: 1}
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataNativeNormRope(state, functions, qBuffer, 2*width, gammaBuffer, out, int32(rows), width, rotation,
			epsilon, scaling, positionBuffer, 0); err != nil {
			return err
		}
		return launchStrataNativeQSAGate(state, functions, attnBuffer, qBuffer, gated, int32(rows), width)
	})
	thetaScale := float64(float32(math.Pow(strataFreqBaseDefault, -2.0/rotation)))
	wantNorm, wantGate := make([]float64, rows*width), make([]float64, rows*width)
	for row := range rows {
		in := qFull[row*2*width : row*2*width+width]
		var squares float64
		for _, v := range in {
			squares += float64(v) * float64(v)
		}
		scale := 1 / math.Sqrt(squares/width+float64(epsilon))
		normed := make([]float64, width)
		for col := range width {
			normed[col] = float64(in[col]) * scale * float64(gamma[col])
		}
		copy(wantNorm[row*width:], normed)
		for pair := range rotation / 2 {
			theta := float64(positions[row]) * math.Pow(thetaScale, float64(pair))
			a, b := normed[pair], normed[pair+rotation/2]
			wantNorm[row*width+pair] = a*math.Cos(theta) - b*math.Sin(theta)
			wantNorm[row*width+pair+rotation/2] = a*math.Sin(theta) + b*math.Cos(theta)
		}
		for col := range width {
			gate := float64(qFull[row*2*width+width+col])
			wantGate[row*width+col] = float64(attn[row*width+col]) / (1 + math.Exp(-gate))
		}
	}
	if d := strataRelativeL1(float32To64(float32s(buffers.read(t, out, 4*rows*width))), wantNorm); d > bound {
		t.Fatalf("the norm and rotation are %.3g from the transcription", d)
	}
	if d := strataRelativeL1(float32To64(float32s(buffers.read(t, gated, 4*rows*width))), wantGate); d > bound {
		t.Fatalf("the gate is %.3g from the transcription", d)
	}
}
