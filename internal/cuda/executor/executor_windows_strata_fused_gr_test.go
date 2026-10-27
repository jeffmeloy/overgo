//go:build windows

package executor

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
)

// TestStrataFusedGRReadMatchesStrata holds Strata's hyper-connection read over the GGUF's Q8_0 projections
// (strata_fused_gr.cuh, from fused_gr.cu's S23 read) to gr_parity's contract and to a float64 transcription, at
// Flash-Next's geometry: GrMulti and FusedGrArgs keep their C layout; a read of T tokens gives each token the mixed row,
// stream scales, injection and folded residual that a read of that token alone gives, bit for
// bit, for T of 1 to 8, with the previous half's write folded in and without; and those values are the transcription's
// within float rounding (the Q8 read keeps its bottleneck in shared memory, its lo workspace unwritten) - each
// stream's rows normed by its root mean square, the bottleneck's SiLU over the down
// projection's stream sums, the injection's bf16 rows, the up projection's sigmoid gate on each normed value, the gated
// streams averaged. The native write (native_gr_post) adds each token's block output weighted by twice the sigmoid of
// its stream's injection over the streams.
func TestStrataFusedGRReadMatchesStrata(t *testing.T) {
	const (
		// bound is the relative L1 distance the read keeps from the float64 transcription.
		bound   = 1e-5
		epsilon = float32(1e-6)
		// residualScale, weightScale and normScale size the residual, the projections and the norm as Flash-Next's.
		residualScale = 0.5
		weightScale   = 0.05
		normScale     = 0.1
	)
	if size := unsafe.Sizeof(strataFusedGRArgs{}); size != 152 {
		t.Fatalf("FusedGrArgs is %d bytes, not its C layout's 152", size)
	}
	if size := unsafe.Sizeof(strataGRMulti{}); size != 1232 {
		t.Fatalf("GrMulti is %d bytes, not its C layout's 1232", size)
	}
	n, hc, lr := strataGREmbedding, strataGRStreams, strataGRRank
	d := n * hc
	worker := newFixtureWorker(t)
	cuda := newFixtureExecutorWithWorker(t, worker)
	buffers := newStrataDevice(t, cuda)
	defer buffers.release(t)
	draws := rand.New(rand.NewPCG(29, 31))
	gauss := func(count int, scale float32) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = float32(draws.NormFloat64()) * scale
		}
		return values
	}
	down := strataRandomWeights(t, draws, dtype.Q8_0, int32(d), int32(lr), weightScale)
	up := strataRandomWeights(t, draws, dtype.Q8_0, int32(lr), int32(d), weightScale)
	downValues, err := quant.Dequantize(dtype.Q8_0, down, uint64(lr*d))
	if err != nil {
		t.Fatal(err)
	}
	upValues, err := quant.Dequantize(dtype.Q8_0, up, uint64(d*lr))
	if err != nil {
		t.Fatal(err)
	}
	injectBF16 := make([]uint16, hc*d)
	injectValues := make([]float64, hc*d)
	for index, value := range gauss(hc*d, weightScale) {
		bits := uint16(math.Float32bits(value) >> 16)
		injectBF16[index], injectValues[index] = bits, float64(math.Float32frombits(uint32(bits)<<16))
	}
	norm := gauss(d, normScale)
	for index := range norm {
		norm[index] += 1
	}
	injectBytes := make([]byte, 2*len(injectBF16))
	for index, bits := range injectBF16 {
		injectBytes[2*index], injectBytes[2*index+1] = byte(bits), byte(bits>>8)
	}
	downBuffer, upBuffer := buffers.upload(t, down), buffers.upload(t, up)
	injectBuffer, normBuffer := buffers.upload(t, injectBytes), buffers.upload(t, float32Bytes(norm))
	type token struct {
		r, bo, inj []float32
	}
	type outputs struct{ mixed, rs, inject, r []float32 }
	read := func(tokens []token, apply bool) []outputs {
		args := make([]strataFusedGRArgs, len(tokens))
		buffersOf := make([][5]driver.DevicePtr, len(tokens))
		for index, tok := range tokens {
			r := buffers.upload(t, float32Bytes(tok.r))
			mixed, rs, lo, inject := buffers.zeros(t, 4*n), buffers.zeros(t, 4*hc), buffers.zeros(t, 4*lr), buffers.zeros(t, 4*hc)
			buffersOf[index] = [5]driver.DevicePtr{mixed, rs, lo, inject, r}
			args[index] = strataFusedGRArgs{
				R: r, ROut: r, WNorm: normBuffer, WInject: injectBuffer, Eps: epsilon, Lo: lo, Rs: rs, InjectOut: inject,
				Mixed: mixed, Q8Down: downBuffer, Q8Up: upBuffer,
			}
			if apply {
				args[index].Apply = 1
				args[index].BoPrev, args[index].InjPrev = buffers.upload(t, float32Bytes(tok.bo)), buffers.upload(t, float32Bytes(tok.inj))
			}
		}
		scratch := buffers.zeros(t, 4*len(tokens)*hc*n)
		buffers.run(t, func(state *device.State, functions functionSet) error {
			return launchStrataFusedGRReadQ8(state, functions, args, scratch)
		})
		got := make([]outputs, len(tokens))
		for index, pointers := range buffersOf {
			got[index] = outputs{
				mixed: float32s(buffers.read(t, pointers[0], 4*n)), rs: float32s(buffers.read(t, pointers[1], 4*hc)),
				inject: float32s(buffers.read(t, pointers[3], 4*hc)),
				r:      float32s(buffers.read(t, pointers[4], 4*d)),
			}
		}
		return got
	}
	for _, apply := range []bool{false, true} {
		tokens := make([]token, strataGRMaxTokens)
		for index := range tokens {
			tokens[index] = token{r: gauss(d, residualScale), bo: gauss(n, residualScale), inj: gauss(hc, 1)}
		}
		alone := make([]outputs, len(tokens))
		for index := range tokens {
			alone[index] = read(tokens[index:index+1], apply)[0]
		}
		for count := 2; count <= strataGRMaxTokens; count++ {
			for index, got := range read(tokens[:count], apply) {
				want := alone[index]
				if !slices.Equal(got.mixed, want.mixed) || !slices.Equal(got.rs, want.rs) ||
					!slices.Equal(got.inject, want.inject) || !slices.Equal(got.r, want.r) {
					t.Fatalf("apply=%t: token %d of a %d-token read differs from its read alone", apply, index, count)
				}
			}
		}
		for index, tok := range tokens {
			// The transcription: the folded residual, each stream's scale, the bottleneck, the injection, the gate.
			folded := make([]float64, d)
			for i := range d {
				folded[i] = float64(tok.r[i])
				if apply {
					weight := 2 / (1 + math.Exp(-float64(tok.inj[i/n])/float64(hc)))
					folded[i] += float64(tok.bo[i%n]) * weight
				}
			}
			scales := make([]float64, hc)
			for c := range hc {
				var squares float64
				for i := c * n; i < (c+1)*n; i++ {
					squares += folded[i] * folded[i]
				}
				scales[c] = 1 / math.Sqrt(squares/float64(n)+float64(epsilon))
			}
			normed := make([]float64, d)
			for i := range d {
				normed[i] = folded[i] * float64(norm[i]) * scales[i/n]
			}
			bottleneck := make([]float64, lr)
			for row := range lr {
				var sum float64
				for i := range d {
					sum += float64(downValues[row*d+i]) * normed[i]
				}
				x := sum / float64(hc)
				bottleneck[row] = x / (1 + math.Exp(-x))
			}
			inject := make([]float64, hc)
			for row := range hc {
				for i := range d {
					inject[row] += injectValues[row*d+i] * normed[i]
				}
			}
			mixed := make([]float64, n)
			for i := range d {
				var gate float64
				for v := range lr {
					gate += float64(upValues[i*lr+v]) * bottleneck[v]
				}
				mixed[i%n] += normed[i] / (1 + math.Exp(-gate)) / float64(hc)
			}
			got := alone[index]
			for _, check := range []struct {
				name string
				got  []float32
				want []float64
			}{{"mixed row", got.mixed, mixed}, {"stream scales", got.rs, scales}, {"injection", got.inject, inject}, {"folded residual", got.r, folded}} {
				if dist := strataRelativeL1(float32To64(check.got), check.want); dist > bound {
					t.Fatalf("apply=%t token %d: the %s is %.3g from the transcription", apply, index, check.name, dist)
				}
			}
		}
	}
	// The native write over three tokens, in place.
	const tokens = 3
	residual, blockOut, inject := gauss(tokens*d, residualScale), gauss(tokens*n, residualScale), gauss(tokens*hc, 1)
	residualBuffer := buffers.upload(t, float32Bytes(residual))
	blockBuffer, injectRows := buffers.upload(t, float32Bytes(blockOut)), buffers.upload(t, float32Bytes(inject))
	buffers.run(t, func(state *device.State, functions functionSet) error {
		return launchStrataNativeGRPost(state, functions, residualBuffer, blockBuffer, injectRows, residualBuffer,
			int32(n), int32(hc), tokens)
	})
	want := make([]float64, tokens*d)
	for tok := range tokens {
		for i := range d {
			weight := 2 / (1 + math.Exp(-float64(inject[tok*hc+i/n])/float64(hc)))
			want[tok*d+i] = float64(residual[tok*d+i]) + float64(blockOut[tok*n+i%n])*weight
		}
	}
	if dist := strataRelativeL1(float32To64(float32s(buffers.read(t, residualBuffer, 4*tokens*d))), want); dist > bound {
		t.Fatalf("the native write is %.3g from its formula", dist)
	}
}
