package processmeasure

import (
	"math"
	"math/big"
	"testing"
	"time"
)

func TestCounterConversion(t *testing.T) {
	for _, pair := range [][2]int64{{0, 1}, {1, 3}, {10_000_001, 10_000_000}, {math.MaxInt64 - 1, math.MaxInt64}, {math.MaxInt64, int64(time.Second)}, {math.MaxInt64, 1}, {-1, 1}, {1, 0}, {1, -1}} {
		got, err := counterDuration(pair[0], pair[1])
		if pair[0] < 0 || pair[1] <= 0 {
			if err == nil {
				t.Fatalf("invalid counter admitted: %v", pair)
			}
			continue
		}
		want := new(big.Int).Mul(big.NewInt(pair[0]), big.NewInt(int64(time.Second)))
		want.Quo(want, big.NewInt(pair[1]))
		if !want.IsInt64() {
			if err == nil {
				t.Fatalf("overflow admitted: %v", pair)
			}
		} else if err != nil || int64(got) != want.Int64() {
			t.Fatalf("counter %v: %v %v, want %s", pair, got, err, want)
		}
	}
}

func TestCounterPrecision(t *testing.T) {
	previous, err := Counter()
	if err != nil {
		t.Fatal(err)
	}
	advanced := false
	for range 1000 {
		next, err := Counter()
		if err != nil || next < previous {
			t.Fatalf("nonmonotonic counter: %v -> %v: %v", previous, next, err)
		}
		advanced = advanced || next > previous
		previous = next
	}
	if !advanced {
		t.Fatal("counter never advanced")
	}
}
