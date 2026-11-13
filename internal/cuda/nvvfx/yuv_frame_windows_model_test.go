//go:build windows && modeltest

package nvvfx

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"

	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
	"overgo/internal/testskip"
)

// TestAYUVFrameIsBT709LimitedRange holds the YUV 4:2:0 descriptor (describeYUV420) to the frame FFmpeg is told it gets
// (clip-encode/device-bytes): a known 8-bit RGBA frame converted on the card by the runtime's transfer reads, plane by
// plane, as BT.709 limited range, luma within a level and each chroma sample within three of its 2 x 2 block's mean.
// Probed on 2026-11-13, pixel format 10 is the runtime's 4:2:0 and descriptor byte 25 its colour space: left at 0 the
// luma read 27 levels off BT.709 and half a level off BT.601, and with cosited chroma a block's chroma was its top-left
// pixel's alone, chroma sample 90's U reading 155 against its block's 142.
func TestAYUVFrameIsBT709LimitedRange(t *testing.T) {
	cudatest.Require(t)
	if os.Getenv(directoryEnvironment) == "" {
		testskip.NotApplicable(t, directoryEnvironment+" names no NVIDIA Video Effects runtime here")
	}
	const width, height = 64, 32
	worker := testcheck.Must(device.New(0))(t)
	defer worker.Close()
	rgba := make([]byte, width*height*rgbaBytes)
	for y := range height {
		for x := range width {
			at := (y*width + x) * rgbaBytes
			rgba[at], rgba[at+1], rgba[at+2], rgba[at+3] = byte(x*4), byte(y*8), byte((x*y)%256), byteScale
		}
	}
	written := make([]byte, YUV420Bytes(width, height))
	testcheck.NoError(t, worker.Do(context.Background(), func(state *device.State) (failed error) {
		runtime, err := Open()
		if err != nil {
			return err
		}
		source, err := state.Driver.MemAlloc(uint64(len(rgba)))
		if err != nil {
			return err
		}
		defer func() { failed = errors.Join(failed, state.Driver.MemFree(source)) }()
		target, err := state.Driver.MemAlloc(uint64(len(written)))
		if err != nil {
			return err
		}
		defer func() { failed = errors.Join(failed, state.Driver.MemFree(target)) }()
		if err := state.Driver.MemcpyHtoD(source, rgba); err != nil {
			return err
		}
		from, err := runtime.describe(source, width, height, false)
		if err != nil {
			return err
		}
		to, err := runtime.describeYUV420(target, width, height)
		if err != nil {
			return err
		}
		if err := errors.Join(runtime.transfer(from, to, 1, state.Stream), state.Driver.StreamSynchronize(state.Stream)); err != nil {
			return err
		}
		return state.Driver.MemcpyDtoH(written, target)
	}))
	// BT.709's luma weights, and its limited range's luma and chroma spans.
	const kr, kb, lumaSpan, chromaSpan = 0.2126, 0.0722, 219, 224
	luma := func(r, g, b float64) float64 { return kr*r + (1-kr-kb)*g + kb*b }
	plane, chroma := width*height, width*height/4
	for index := range plane {
		r, g, b := float64(rgba[index*rgbaBytes])/byteScale, float64(rgba[index*rgbaBytes+1])/byteScale, float64(rgba[index*rgbaBytes+2])/byteScale
		if want := 16 + lumaSpan*luma(r, g, b); math.Abs(float64(written[index])-want) > 1 {
			t.Fatalf("pixel %d's luma is %d, want BT.709 limited range's %.1f", index, written[index], want)
		}
	}
	for y := range height / 2 {
		for x := range width / 2 {
			var cb, cr float64
			for _, offset := range []int{0, 1, width, width + 1} {
				at := ((2*y)*width + 2*x + offset) * rgbaBytes
				r, g, b := float64(rgba[at])/byteScale, float64(rgba[at+1])/byteScale, float64(rgba[at+2])/byteScale
				cb += (b - luma(r, g, b)) / (2 * (1 - kb)) / 4
				cr += (r - luma(r, g, b)) / (2 * (1 - kr)) / 4
			}
			sample := y*width/2 + x
			for _, read := range []struct {
				name  string
				value byte
				want  float64
			}{{"U", written[plane+sample], 128 + chromaSpan*cb}, {"V", written[plane+chroma+sample], 128 + chromaSpan*cr}} {
				if math.Abs(float64(read.value)-read.want) > 3 {
					t.Fatalf("chroma sample %d's %s is %d, want BT.709 limited range's %.1f of its block", sample, read.name, read.value, read.want)
				}
			}
		}
	}
}
