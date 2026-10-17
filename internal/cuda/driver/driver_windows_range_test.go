//go:build windows

package driver

import (
	"fmt"
	"testing"
)

func TestOwnsDeviceRange(t *testing.T) {
	library := &Library{allocations: map[DevicePtr]uint64{0x1000: 0x100}}
	for _, test := range []struct {
		name    string
		pointer DevicePtr
		bytes   uint64
		want    bool
	}{
		{"whole", 0x1000, 0x100, true},
		{"base too large", 0x1000, 0x101, false},
		{"overflowing length", 0x1080, ^uint64(0), false},
		{"interior", 0x1080, 0x80, true},
		{"past end", 0x1080, 0x81, false},
		{"before", 0x0fff, 1, false},
		{"end", 0x1100, 1, false},
		{"zero pointer", 0, 1, false},
		{"zero bytes", 0x1000, 0, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := library.ownsDeviceRange(test.pointer, test.bytes); got != test.want {
				t.Fatalf("ownsDeviceRange(%#x, %#x)=%v want %v", test.pointer, test.bytes, got, test.want)
			}
		})
	}
}

func BenchmarkOwnsDeviceRange(b *testing.B) {
	// The large case reflects the retained Flash-Next profile's live allocations.
	for _, count := range []int{16, 40000} {
		library := &Library{allocations: make(map[DevicePtr]uint64, count)}
		for i := range count {
			library.allocations[DevicePtr(i+1)*0x1000] = 0x100
		}
		for _, offset := range []DevicePtr{0, 0x80} {
			b.Run(fmt.Sprintf("allocations=%d/offset=%d", count, offset), func(b *testing.B) {
				for b.Loop() {
					if !library.ownsDeviceRange(DevicePtr(count/2)*0x1000+offset, 0x80) {
						b.Fatal("owned range rejected")
					}
				}
			})
		}
	}
}
