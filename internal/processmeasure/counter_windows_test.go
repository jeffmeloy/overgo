package processmeasure

import (
	"errors"
	"math/big"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

func TestCounterPrecisionFailureWindows(t *testing.T) {
	original := performanceFrequency
	t.Cleanup(func() { performanceFrequency = original })
	failure := errors.New("frequency unavailable")
	performanceFrequency = func() (int64, error) { return 0, failure }
	if _, err := Counter(); !errors.Is(err, failure) {
		t.Fatalf("lost native counter failure: %v", err)
	}
	performanceFrequency = func() (int64, error) { return 0, nil }
	if _, err := Counter(); err == nil {
		t.Fatal("invalid native frequency accepted")
	}
}

func TestCounterPrecisionWindows(t *testing.T) {
	query := syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceCounter")
	frequencyQuery := syscall.NewLazyDLL("kernel32.dll").NewProc("QueryPerformanceFrequency")
	var frequency int64
	if ok, _, err := frequencyQuery.Call(uintptr(unsafe.Pointer(&frequency))); ok == 0 {
		t.Fatal(err)
	}
	native := func() int64 {
		var value int64
		if ok, _, err := query.Call(uintptr(unsafe.Pointer(&value))); ok == 0 {
			t.Fatal(err)
		}
		valueNS := new(big.Int).Mul(big.NewInt(value), big.NewInt(int64(time.Second)))
		return valueNS.Quo(valueNS, big.NewInt(frequency)).Int64()
	}
	for range 10000 {
		coarse := time.Now()
		before := native()
		got, err := Counter()
		after := native()
		if err != nil || int64(got) < before || int64(got) > after {
			t.Fatalf("native bounds [%d,%d], got %d: %v", before, after, got, err)
		}
		if time.Since(coarse) == 0 && after > before {
			t.Logf("native counter advanced %dns inside one unchanged Go clock tick; sample bracket passed", after-before)
			return
		}
	}
	// A future Go runtime may itself use a high-resolution monotonic clock.
	// Native brackets above remain the acceptance authority in that case.
	t.Log("10000 native brackets passed; no unchanged Go clock tick observed")
}
