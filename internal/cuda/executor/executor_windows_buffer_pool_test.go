//go:build windows

package executor

import (
	"testing"
)

// TestDeviceBufferPoolReusesClassesAndTrims: capacity requests of many
// distinct sizes share a bounded set of size classes, and the pool's
// free lists never exceed their share of device memory. A pass whose
// every prompt had a new length used to leave one exact-size page set
// per length in the lists until the device filled.
func TestDeviceBufferPoolReusesClassesAndTrims(t *testing.T) {
	cuda := newFixtureExecutor(t)
	ctx := t.Context()
	// Two hundred sizes between 1.1 and 3 MiB, no two alike, each held
	// briefly: with exact keys every one would be a fresh allocation.
	for round := range 3 {
		for index := range 200 {
			size := uint64(1<<20) + uint64(100_000+index*9_500)
			buffer, err := cuda.AllocateDeviceBuffer(ctx, size)
			if err != nil {
				t.Fatal(err)
			}
			if err := buffer.Release(ctx); err != nil {
				t.Fatal(err)
			}
		}
		cuda.mu.RLock()
		allocations := len(cuda.resources.buffers.allocations)
		cuda.mu.RUnlock()
		// Sizes span the 1-2 MiB and 2-4 MiB octaves: at most eight classes
		// each, and the sequential hold-and-release keeps one buffer per
		// class live at a time.
		if allocations > 2*int(deviceBufferClassSteps) {
			t.Fatalf("round %d: %d pooled allocations for 200 distinct sizes", round, allocations)
		}
	}
	// Large buffers past the free share are trimmed rather than kept: hold
	// and release buffers whose total exceeds the share and check the
	// free lists stayed within it.
	cuda.mu.RLock()
	limit := cuda.resources.buffers.freeLimit
	cuda.mu.RUnlock()
	if limit == 0 {
		t.Fatal("free limit was not derived from the device")
	}
	large := limit/4 + 1<<20
	for index := range 8 {
		buffer, err := cuda.AllocateDeviceBuffer(ctx, large+uint64(index)<<20)
		if err != nil {
			t.Fatal(err)
		}
		if err := buffer.Release(ctx); err != nil {
			t.Fatal(err)
		}
	}
	cuda.mu.RLock()
	freeBytes := cuda.resources.buffers.freeBytes
	cuda.mu.RUnlock()
	if freeBytes > limit {
		t.Fatalf("free lists hold %d bytes, limit %d", freeBytes, limit)
	}
}

// TestDeviceBufferClass: the exact range keeps alignment, the classed
// range rounds up by an eighth of its octave and never below the request.
func TestDeviceBufferClass(t *testing.T) {
	for _, testCase := range []struct {
		size, want uint64
	}{
		{1, 256},
		{1000, 1024},
		{1 << 20, 1 << 20},
		{1<<20 + 1, 1<<20 + 1<<18},
		{1<<20 + 1<<18, 1<<20 + 1<<18},
		{3 << 20, 3 << 20},
		{3<<20 + 1, 3<<20 + 1<<19},
		{1 << 30, 1 << 30},
		{1<<30 + 5, 1<<30 + 1<<28},
	} {
		got, ok := deviceBufferClass(testCase.size)
		if !ok || got != testCase.want || got < testCase.size {
			t.Fatalf("class(%d) = %d, %t; want %d", testCase.size, got, ok, testCase.want)
		}
	}
	if _, ok := deviceBufferClass(0); ok {
		t.Fatal("zero size classed")
	}
}
