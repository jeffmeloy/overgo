//go:build windows

package executor

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"overgo/internal/comfyquant"
	"overgo/internal/cuda/device"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/testcheck"
)

// TestTheCardDecodesConvRotAsTheHost holds the card's int8 ConvRot decode (DecodeInt8ConvRot, moss-first-run) to the
// host's (comfyquant.DecodeRow), bit for bit: over groups of 64, 256 and 1024 columns, rows of seeded int8 values and of
// every value at the int8 extremes, which give the largest sums, under scales of full mantissas and both signs from
// those whose products fall subnormal to 1e30, zero, an infinite scale and a NaN, each BF16 bit of every row is the
// host's. The MOSS Ultra encoder's weights, decoded so on the host, took 41 s of the workflow's first prompt.
func TestTheCardDecodesConvRotAsTheHost(t *testing.T) {
	cudatest.Require(t)
	random := rand.New(rand.NewPCG(20261121, 1))
	worker := testcheck.Must(device.New(0))(t)
	defer func() { testcheck.NoError(t, worker.Close()) }()
	runtime := testcheck.Must(NewWithWorker(worker))(t)
	defer func() { testcheck.NoError(t, runtime.Close()) }()
	scales := []float32{0, float32(math.Inf(1)), float32(math.NaN()), -1e30, 1e-42, -3e-39, 1.1e-38}
	for _, shape := range []struct{ rows, inner, group uint32 }{{40, 192, 64}, {48, 4096, 256}, {24, 3072, 1024}} {
		t.Run(fmt.Sprintf("%dx%d/%d", shape.rows, shape.inner, shape.group), func(t *testing.T) {
			rows, inner := uint64(shape.rows), uint64(shape.inner)
			stored := make([]byte, rows*inner+4*rows)
			for row := range rows {
				values := stored[row*inner : (row+1)*inner]
				switch row {
				case 0:
					for index := range values {
						values[index] = 0x80
					}
				case 1:
					for index := range values {
						values[index] = 0x7f
					}
				default:
					for index := range values {
						values[index] = byte(random.Uint32())
					}
				}
				scale := float32(math.Ldexp(1+random.Float64(), random.IntN(80)-40)) * float32(1-2*random.IntN(2))
				if row < uint64(len(scales)) {
					scale = scales[row]
				}
				binary.LittleEndian.PutUint32(stored[rows*inner+4*row:], math.Float32bits(scale))
			}
			decoded := make([]byte, 2*rows*inner)
			testcheck.NoError(t, worker.Do(t.Context(), func(state *device.State) (err error) {
				output, err := state.Driver.MemAlloc(uint64(len(decoded)))
				if err != nil {
					return err
				}
				defer func() { err = errors.Join(err, state.Driver.MemFree(output)) }()
				if err := runtime.DecodeInt8ConvRot(state, Int8ConvRotDecode{Stored: stored, Rows: shape.rows, Inner: shape.inner, Group: shape.group, Output: output}); err != nil {
					return err
				}
				return state.Driver.MemcpyDtoH(decoded, output)
			}))
			host := make([]uint16, inner)
			parted := 0
			for row := range rows {
				scale := math.Float32frombits(binary.LittleEndian.Uint32(stored[rows*inner+4*row:]))
				comfyquant.DecodeRow(host, stored[row*inner:(row+1)*inner], scale, uint64(shape.group))
				for column, want := range host {
					if got := binary.LittleEndian.Uint16(decoded[2*(row*inner+uint64(column)):]); got != want {
						if parted++; parted <= 4 {
							t.Errorf("row %d (scale %g), column %d decodes to BF16 %#04x on the card, %#04x on the host", row, scale, column, got, want)
						}
					}
				}
			}
			if parted > 0 {
				t.Errorf("%d of %d values part from the host's decode; want none", parted, rows*inner)
			}
		})
	}
}
