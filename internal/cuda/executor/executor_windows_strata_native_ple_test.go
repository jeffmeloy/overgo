//go:build windows

package executor

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/quant"
)

// TestStrataNativePLEMatchesStrata holds Strata's lexical-injection post-ops and history advance
// (strata_native_ple.cuh and strata_round_elementwise.cuh, from native_ple_postops.cu, native_gr_norm.cu and ple.cu) to
// a float64 transcription of native_ple_postops at Flash-Next's geometry, within fast math's rounding: the projected key
// and the hidden rows normed by stream; each stream's gate the sigmoid of the signed square root of their scaled dot;
// the value broadcast by the gates, normed by stream; the dilated conv over the history's rows 9, 6 and 3 tokens back and
// the normed row, its SiLU added to the hidden and gated rows. The history advance shifts each channel's nine rows up
// one and writes the normed row last, exactly.
func TestStrataNativePLEMatchesStrata(t *testing.T) {
	const (
		n       = strataPLEEmbedding
		streams = strataPLEStreams
		width   = strataPLEWidth
		history = 9
		taps    = 4
		// bound is the relative L1 distance the post-ops keep from the transcription.
		bound = 1e-5
	)
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(89, 97))
	gauss := func(count int, scale float64) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(draws.NormFloat64() * scale)
		}
		return values
	}
	norm := func() []float32 {
		values := gauss(width, 0.1)
		for index := range values {
			values[index]++
		}
		return values
	}
	key, hidden, value, rows := gauss(width, 1), gauss(width, 1), gauss(n, 1), gauss(width*history, 1)
	normKey, normQuery, normConv := norm(), norm(), norm()
	conv := make([]byte, 2*width*taps)
	convValues := make([]float64, width*taps)
	for index := range convValues {
		bits := quant.Float32ToFloat16(float32(draws.NormFloat64() * 0.3))
		conv[2*index], conv[2*index+1] = byte(bits), byte(bits>>8)
		convValues[index] = float64(quant.Float16ToFloat32(bits))
	}
	weights := strataPLEWeights{
		NormKey: buffers.upload(t, float32Bytes(normKey)), NormQuery: buffers.upload(t, float32Bytes(normQuery)),
		NormConv: buffers.upload(t, float32Bytes(normConv)), ConvF16: buffers.upload(t, conv),
	}
	out := strataPLEBuffers{
		Key: buffers.zeros(t, 4*width), Query: buffers.zeros(t, 4*width), Gate: buffers.zeros(t, 4*streams),
		Gated: buffers.zeros(t, 4*width), Normalized: buffers.zeros(t, 4*width), Conv: buffers.zeros(t, 4*width),
		Result: buffers.zeros(t, 4*width),
	}
	keyBuffer, hiddenBuffer := buffers.upload(t, float32Bytes(key)), buffers.upload(t, float32Bytes(hidden))
	valueBuffer, historyBuffer := buffers.upload(t, float32Bytes(value)), buffers.upload(t, float32Bytes(rows))
	buffers.run(t, func(state *device.State, functions functionSet) error {
		if err := launchStrataNativePLEPostops(state, functions, keyBuffer, hiddenBuffer, valueBuffer, historyBuffer, weights, out); err != nil {
			return err
		}
		return launchStrataPLEHistoryAdvance(state, functions, historyBuffer, out.Normalized)
	})
	rmsNorm := func(x, gamma []float32) []float64 {
		normed := make([]float64, width)
		for stream := range streams {
			var squares float64
			for d := range n {
				squares += float64(x[stream*n+d]) * float64(x[stream*n+d])
			}
			scale := 1 / math.Sqrt(squares/n+float64(strataPLEEpsilon))
			for d := range n {
				normed[stream*n+d] = float64(x[stream*n+d]) * scale * float64(gamma[stream*n+d])
			}
		}
		return normed
	}
	keyNormed, queryNormed := rmsNorm(key, normKey), rmsNorm(hidden, normQuery)
	gates := make([]float64, streams)
	gated := make([]float32, width)
	gatedValues := make([]float64, width)
	for stream := range streams {
		var dot float64
		for d := range n {
			dot += keyNormed[stream*n+d] * queryNormed[stream*n+d]
		}
		s := dot / math.Sqrt(n)
		sign := 0.0
		if s > 0 {
			sign = 1
		} else if s < 0 {
			sign = -1
		}
		gates[stream] = 1 / (1 + math.Exp(-sign*math.Sqrt(max(math.Abs(s), 1e-6))))
		for d := range n {
			gatedValues[stream*n+d] = float64(value[d]) * gates[stream]
			gated[stream*n+d] = float32(gatedValues[stream*n+d])
		}
	}
	normed := rmsNorm(gated, normConv)
	result := make([]float64, width)
	for c := range width {
		var sum float64
		for k := range taps {
			x := normed[c]
			if k < taps-1 {
				x = float64(rows[c*history+3*k])
			}
			sum += x * convValues[c*taps+k]
		}
		result[c] = float64(hidden[c]) + gatedValues[c] + sum/(1+math.Exp(-sum))
	}
	if d := strataRelativeL1(float32To64(float32s(buffers.read(t, out.Gate, 4*streams))), gates); d > bound {
		t.Fatalf("the gates are %.3g from the transcription", d)
	}
	if d := strataRelativeL1(float32To64(float32s(buffers.read(t, out.Result, 4*width))), result); d > bound {
		t.Fatalf("the result is %.3g from the transcription", d)
	}
	gotNormed := float32s(buffers.read(t, out.Normalized, 4*width))
	advanced := float32s(buffers.read(t, historyBuffer, 4*width*history))
	for c := range width {
		want := append(slices.Clone(rows[c*history+1:(c+1)*history]), gotNormed[c])
		if !slices.Equal(advanced[c*history:(c+1)*history], want) {
			t.Fatalf("channel %d's advanced history differs", c)
		}
	}
}
