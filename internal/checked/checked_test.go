package checked

import (
	"math"
	"testing"
)

func TestCheckedArithmetic(t *testing.T) {
	const signedValue int64 = 7
	if value, ok := Uint64(signedValue); !ok || value != uint64(signedValue) {
		t.Fatalf("unsigned conversion = %d, %v", value, ok)
	}
	if _, ok := Uint64(-signedValue); ok {
		t.Fatal("negative conversion accepted")
	}
	if value, ok := Add64(1, 2, 3); !ok || value != 6 {
		t.Fatalf("sum = %d, %v", value, ok)
	}
	if _, ok := Add64(math.MaxUint64, 1); ok {
		t.Fatal("overflowing sum accepted")
	}
	if value, ok := Mul64(7, 8); !ok || value != 56 {
		t.Fatalf("product = %d, %v", value, ok)
	}
	if _, ok := Mul64(math.MaxUint64, 2); ok {
		t.Fatal("overflowing product accepted")
	}
	if value, ok := Align(33, 32); !ok || value != 64 {
		t.Fatalf("alignment = %d, %v", value, ok)
	}
}
