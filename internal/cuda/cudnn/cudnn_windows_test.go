//go:build windows

package cudnn

import (
	"context"
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/quant"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testskip"
)

// convolved is the convolution of shape computed plainly on the host, each
// sum in float64: the value the library's is held to.
func convolved(shape Shape, input, filter []float32) []float32 {
	frames, height, width := shape.output()
	output := make([]float32, shape.Out*frames*height*width)
	at := func(channel, frame, row, column int) float64 {
		if frame < 0 || frame >= shape.Frames || row < 0 || row >= shape.Height || column < 0 || column >= shape.Width {
			return 0
		}
		return float64(input[((channel*shape.Frames+frame)*shape.Height+row)*shape.Width+column])
	}
	for out := range shape.Out {
		for frame := range frames {
			for row := range height {
				for column := range width {
					var sum float64
					for in := range shape.In {
						for kt := range shape.Kernel[overFrames] {
							for kh := range shape.Kernel[overRows] {
								for kw := range shape.Kernel[overColumns] {
									weight := filter[(((out*shape.In+in)*shape.Kernel[overFrames]+kt)*shape.Kernel[overRows]+kh)*shape.Kernel[overColumns]+kw]
									sum += float64(weight) * at(in,
										frame*shape.Stride[overFrames]+kt-shape.Pad[overFrames],
										row*shape.Stride[overRows]+kh-shape.Pad[overRows],
										column*shape.Stride[overColumns]+kw-shape.Pad[overColumns])
								}
							}
						}
					}
					output[((out*frames+frame)*height+row)*width+column] = float32(sum)
				}
			}
		}
	}
	return output
}

// TestConvolutionIsTheSumItIsDefinedAs holds the library's forward
// convolution, through this binding, to the sum a convolution is defined as,
// for the two shapes the video codec convolves in: a kernel of three in every
// extent at stride one, padded in rows and columns and not in frames, where
// the codec stages the frames before a chunk itself, and a kernel of three in
// rows and columns at stride two with no padding. Each is held in the two
// pairs of types the binding plans: single precision throughout, to a
// float32 sum's rounding, and half precision tensors with their sums kept in
// single, the pair the codec runs, to the sum of the operands as half
// precision holds them, within a half's rounding of what is written. A
// machine without cuDNN on its search path has nothing to hold, and the test
// says so.
func TestConvolutionIsTheSumItIsDefinedAs(t *testing.T) {
	cudatest.Require(t)
	worker, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()
	var library *Library
	err = worker.Do(context.Background(), func(state *device.State) (err error) {
		library, err = Open(state.Stream)
		return err
	})
	if errors.Is(err, ErrAbsent) {
		testskip.NotApplicable(t, err.Error())
	}
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := worker.Do(context.Background(), func(*device.State) error { return library.Close() }); err != nil {
			t.Error(err)
		}
	}()
	random := rand.New(rand.NewPCG(1, 2))
	values := func(count int) []float32 {
		drawn := make([]float32, count)
		for index := range drawn {
			drawn[index] = float32(random.NormFloat64())
		}
		return drawn
	}
	// halves gives values as half precision holds them, and the bytes of them.
	halves := func(values []float32) ([]float32, []byte) {
		t.Helper()
		stored, err := quant.Quantize(dtype.F16, values)
		if err != nil {
			t.Fatal(err)
		}
		held, err := quant.Dequantize(dtype.F16, stored, uint64(len(values)))
		if err != nil {
			t.Fatal(err)
		}
		return held, stored
	}
	for name, shape := range map[string]Shape{
		"three in every extent, padded in rows and columns": {In: 5, Out: 7, Frames: 4, Height: 6, Width: 9, Kernel: [extents]int{3, 3, 3}, Stride: [extents]int{1, 1, 1}, Pad: [extents]int{0, 1, 1}},
		"three in rows and columns, at stride two":          {In: 6, Out: 4, Frames: 3, Height: 9, Width: 11, Kernel: [extents]int{1, 3, 3}, Stride: [extents]int{1, 2, 2}},
	} {
		frames, height, width := shape.output()
		input := values(shape.In * shape.Frames * shape.Height * shape.Width)
		filter := values(shape.Out * shape.In * shape.Kernel[overFrames] * shape.Kernel[overRows] * shape.Kernel[overColumns])
		for _, pair := range []struct {
			name string
			data DataType
		}{{"single precision", DataF32}, {"half precision summed in single", DataF16}} {
			operands, stored := [][]float32{input, filter}, [][]byte{driver.Bytes(input), driver.Bytes(filter)}
			if pair.data == DataF16 {
				for index, values := range operands {
					operands[index], stored[index] = halves(values)
				}
			}
			want := convolved(shape, operands[0], operands[1])
			written := make([]byte, len(want)*(len(stored[0])/len(input)))
			if err := worker.Do(context.Background(), func(state *device.State) (failed error) {
				plan, err := library.Plan(shape, pair.data, DataF32)
				if err != nil {
					return err
				}
				defer func() { failed = errors.Join(failed, library.Release(plan)) }()
				var pointers [4]driver.DevicePtr
				defer func() {
					for _, pointer := range pointers {
						if pointer != 0 {
							failed = errors.Join(failed, state.Driver.MemFree(pointer))
						}
					}
				}()
				for index, storage := range stored {
					if pointers[index], err = state.Driver.MemAlloc(uint64(len(storage))); err != nil {
						return err
					}
					if err := state.Driver.MemcpyHtoD(pointers[index], storage); err != nil {
						return err
					}
				}
				if pointers[2], err = state.Driver.MemAlloc(uint64(len(written))); err != nil {
					return err
				}
				if plan.Workspace != 0 {
					if pointers[3], err = state.Driver.MemAlloc(plan.Workspace); err != nil {
						return err
					}
				}
				if err := library.Forward(plan, pointers[0], pointers[1], pointers[3], pointers[2]); err != nil {
					return err
				}
				if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
					return err
				}
				return state.Driver.MemcpyDtoH(written, pointers[2])
			}); err != nil {
				t.Fatalf("%s, %s: %v", name, pair.name, err)
			}
			got := make([]float32, len(want))
			// A float32 sum's rounding, and for a value written as a half the half's own.
			tolerance, relative := 1e-4, 0.0
			if pair.data == DataF16 {
				if got, err = quant.Dequantize(dtype.F16, written, uint64(len(want))); err != nil {
					t.Fatal(err)
				}
				relative = 1.0 / (1 << 10)
			} else {
				copy(driver.Bytes(got), written)
			}
			var worst float64
			for index := range want {
				apart := math.Abs(float64(got[index]) - float64(want[index]))
				if worst = max(worst, apart); apart > tolerance+relative*math.Abs(float64(want[index])) {
					t.Fatalf("%s, %s: value %d is %g, the sum %g", name, pair.name, index, got[index], want[index])
				}
			}
			t.Logf("%s, %s: %d frames of %d by %d, at most %.3g from the sum", name, pair.name, frames, height, width, worst)
		}
	}
}
