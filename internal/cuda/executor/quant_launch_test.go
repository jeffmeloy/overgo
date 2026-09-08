package executor

import (
	"encoding/binary"
	"math"
	"testing"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/tensor/dtype"
)

func TestQuantMulMatSpansAdvanceOnlyColumns(t *testing.T) {
	const inner, width, columns = uint32(256), uint32(17408), uint32(8192)
	trace := &device.LaunchTrace{Probe: true}
	state := &device.State{Trace: trace}
	function := boundKernel{id: kernelMulMatIq4XsF32, argumentCount: 6}
	err := launchQuantMulMatSpans(state, function, dtype.IQ4XS, 16, 32, 48, inner, width, columns)
	if err != nil {
		t.Fatal(err)
	}
	const recordBytes = 4*8 + 3*8 + 3*4
	raw := trace.Data()
	if len(raw) != 2*recordBytes {
		t.Fatalf("launch bytes %d: want two launches", len(raw))
	}
	var covered uint32
	for offset := 0; offset < len(raw); offset += recordBytes {
		record := raw[offset : offset+recordBytes]
		left := binary.LittleEndian.Uint64(record[32:])
		right := binary.LittleEndian.Uint64(record[40:])
		output := binary.LittleEndian.Uint64(record[48:])
		rows := binary.LittleEndian.Uint32(record[64:])
		if left != 16 || right != 32+uint64(covered)*uint64(inner)*4 || output != 48+uint64(covered)*uint64(width)*4 {
			t.Fatalf("column %d: invalid pointers %d/%d/%d", covered, left, right, output)
		}
		if rows == 0 || rows > columns-covered {
			t.Fatalf("column %d: invalid span %d", covered, rows)
		}
		covered += rows
	}
	if covered != columns {
		t.Fatalf("covered %d columns; want %d", covered, columns)
	}
	trace.Reset(true)
	ptr := driver.DevicePtr(0)
	if err := launch1DABI(state, boundKernel{argumentCount: 1}, math.MaxUint32, &ptr); err != nil {
		t.Fatal(err)
	}
	if blocks := binary.LittleEndian.Uint32(trace.Data()[8:]); blocks != 1<<24 {
		t.Fatalf("maximum launch wrapped to %d blocks", blocks)
	}
}

func TestQuantMulMatSpansRespectThreadIndex(t *testing.T) {
	for _, storage := range []dtype.Type{dtype.Q3K, dtype.IQ4XS, dtype.Q8_0} {
		for _, width := range []uint32{1, 17408, math.MaxUint32 / 32} {
			capacity, err := quantMulMatSpanRows(storage, width)
			if err != nil || capacity == 0 {
				t.Fatalf("%s width %d: capacity %d, %v", storage, width, capacity, err)
			}
			if _, err := quantMulMatLaunchCount(storage, width, capacity); err != nil {
				t.Fatal(err)
			}
			if _, err := quantMulMatLaunchCount(storage, width, capacity+1); err == nil {
				t.Fatalf("%s width %d: capacity does not cover the launch boundary", storage, width)
			}
		}
	}
	for _, width := range []uint32{0, math.MaxUint32} {
		if _, err := quantMulMatSpanRows(dtype.Q3K, width); err == nil {
			t.Fatalf("accepted unlaunchable width %d", width)
		}
	}
}
