package processmeasure

import (
	"errors"
	"math"
	"math/bits"
	"time"
)

func counterDuration(ticks, frequency int64) (time.Duration, error) {
	if ticks < 0 || frequency <= 0 {
		return 0, errors.New("measurement: invalid performance counter or frequency")
	}
	seconds := ticks / frequency
	if seconds > math.MaxInt64/int64(time.Second) {
		return 0, errors.New("measurement: performance counter duration overflow")
	}
	hi, lo := bits.Mul64(uint64(ticks%frequency), uint64(time.Second))
	fraction, _ := bits.Div64(hi, lo, uint64(frequency))
	whole := seconds * int64(time.Second)
	if fraction > uint64(math.MaxInt64-whole) {
		return 0, errors.New("measurement: performance counter duration overflow")
	}
	return time.Duration(whole + int64(fraction)), nil
}
