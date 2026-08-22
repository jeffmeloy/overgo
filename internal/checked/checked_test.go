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
	if value, ok := DivExact64(56, 8); !ok || value != 7 {
		t.Fatalf("quotient = %d, %v", value, ok)
	}
	if _, ok := DivExact64(7, 0); ok {
		t.Fatal("zero divisor accepted")
	}
	if _, ok := DivExact64(7, 2); ok {
		t.Fatal("inexact quotient accepted")
	}
	if value, ok := DivExactInt(56, 8); !ok || value != 7 {
		t.Fatalf("integer quotient = %d, %v", value, ok)
	}
	if _, ok := DivExactInt(-1, 1); ok {
		t.Fatal("negative dividend accepted")
	}
	if _, ok := Mul64(math.MaxUint64, 2); ok {
		t.Fatal("overflowing product accepted")
	}
	if value, ok := MulInt(7, 8); !ok || value != 56 {
		t.Fatalf("integer product = %d, %v", value, ok)
	}
	if _, ok := MulInt(-1, 2); ok {
		t.Fatal("negative integer product accepted")
	}
	if value, ok := Align(33, 32); !ok || value != 64 {
		t.Fatalf("alignment = %d, %v", value, ok)
	}
	if value, ok := RoundUpMultiple(5, 5); !ok || value != 5 {
		t.Fatalf("round-up exact multiple = %d, %v", value, ok)
	}
	if value, ok := RoundUpMultiple(6, 5); !ok || value != 10 {
		t.Fatalf("round-up multiple = %d, %v", value, ok)
	}
	if _, ok := RoundUpMultiple(1, 0); ok {
		t.Fatal("zero round-up multiple accepted")
	}
}
