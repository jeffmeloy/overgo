//go:build windows

package driver

import "testing"

func TestOwnsDeviceRange(t *testing.T) {
	library := &Library{allocations: map[DevicePtr]uint64{0x1000: 0x100}}
	for _, test := range []struct {
		name    string
		pointer DevicePtr
		bytes   uint64
		want    bool
	}{
		{"whole", 0x1000, 0x100, true},
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
