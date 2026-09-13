package media

import (
	"math"
	"testing"
)

func TestGIFFrameTimingPreservesDuration(t *testing.T) {
	// Exercise every representable positive frame rate and each prefix across
	// two complete seconds. The format quantization bound is half a unit.
	for fps := 1; fps <= 100; fps++ {
		total := 0
		for frame := range fps * 2 {
			delay, err := GIFFrameDelay(fps, frame)
			if err != nil || delay <= 0 {
				t.Fatalf("fps=%d frame=%d delay=%d error=%v", fps, frame, delay, err)
			}
			total += delay
			if math.Abs(float64(total*fps-(frame+1)*100))*2 > float64(fps) {
				t.Fatalf("fps=%d frame=%d total=%d exceeds half-centisecond error", fps, frame, total)
			}
		}
		large, err := GIFFrameDelay(fps, math.MaxInt)
		small, smallErr := GIFFrameDelay(fps, math.MaxInt%fps)
		if err != nil || smallErr != nil || large != small {
			t.Fatal("large frame index changes the timing period")
		}
	}
	for _, value := range [][2]int{{0, 0}, {-1, 0}, {101, 0}, {16, -1}} {
		if _, err := GIFFrameDelay(value[0], value[1]); err == nil {
			t.Fatalf("invalid timing accepted: %v", value)
		}
	}
}
