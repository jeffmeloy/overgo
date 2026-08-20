package dtype

import (
	"math"
	"testing"
)

// TestF8E4M3EncodeIsExactInverseOnEveryValue: the gold-standard codec test —
// encode(decode(b)) == b for all 256 patterns except the NaN pair (whose
// payload collapses to the canonical NaN pattern) and negative zero.
func TestF8E4M3EncodeIsExactInverseOnEveryValue(t *testing.T) {
	for b := 0; b < 256; b++ {
		value := F8E4M3ToFloat32(byte(b))
		encoded := Float32ToF8E4M3(value)
		if math.IsNaN(float64(value)) {
			if encoded&0x7f != 0x7f {
				t.Fatalf("byte %02x: NaN encoded as %02x", b, encoded)
			}
			continue
		}
		if value == 0 {
			if encoded&0x7f != 0 {
				t.Fatalf("byte %02x: zero encoded as %02x", b, encoded)
			}
			continue
		}
		if encoded != byte(b) {
			t.Fatalf("byte %02x (value %g) re-encoded as %02x", b, value, encoded)
		}
	}
}

// TestF8E4M3EncodeRoundsAndSaturates: midpoints round to even and
// out-of-range magnitudes saturate to 448.
func TestF8E4M3EncodeRoundsAndSaturates(t *testing.T) {
	if got := Float32ToF8E4M3(1000); F8E4M3ToFloat32(got) != 448 {
		t.Fatalf("1000 -> %02x (%g), want 448", got, F8E4M3ToFloat32(got))
	}
	if got := Float32ToF8E4M3(-1000); F8E4M3ToFloat32(got) != -448 {
		t.Fatalf("-1000 -> %02x, want -448", got)
	}
	// Nearest e4m3 neighbors of 3.1 are 3.0 and 3.25; 3.1 rounds to 3.0.
	if got := F8E4M3ToFloat32(Float32ToF8E4M3(3.1)); got != 3.0 {
		t.Fatalf("3.1 -> %g, want 3.0", got)
	}
	// Exact midpoint 3.125 rounds to even mantissa (3.25 has mantissa 0b101,
	// 3.0 has 0b100): round-half-even selects 3.0.
	if got := F8E4M3ToFloat32(Float32ToF8E4M3(3.125)); got != 3.0 {
		t.Fatalf("3.125 -> %g, want 3.0 (round half to even)", got)
	}
}
