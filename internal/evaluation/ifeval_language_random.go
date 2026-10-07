package evaluation

import (
	"math"
	"math/bits"
)

// MT19937 and CPython's random/choice/gauss transforms. The frozen native
// seed-zero state owns initialization; this is not a configurable RNG API.
type ifevalPythonRandom struct {
	state     []uint32
	index     int
	cached    float64
	hasCached bool
}

// The constants of MT19937's seeding by a key, as CPython's random.seed uses
// it for an integer: a first fill from a fixed seed, then two passes that mix
// the key in.
const (
	pythonStateWords      = 624
	pythonFillSeed        = 19650218
	pythonFillMultiplier  = 1812433253
	pythonKeyMultiplier   = 1664525
	pythonFinalMultiplier = 1566083941
	pythonMixShift        = 30
	pythonTopBit          = 0x80000000
)

// seededPythonRandom is CPython's random.Random(seed) for a seed that fits
// one 32-bit word, so that a set drawn by a Python script with that seed is
// drawn again by the same numbers.
func seededPythonRandom(seed uint32) *ifevalPythonRandom {
	state := make([]uint32, pythonStateWords)
	state[0] = pythonFillSeed
	for index := 1; index < len(state); index++ {
		state[index] = pythonFillMultiplier*(state[index-1]^state[index-1]>>pythonMixShift) + uint32(index)
	}
	at := 1
	advance := func() {
		if at++; at >= len(state) {
			state[0], at = state[len(state)-1], 1
		}
	}
	for range len(state) {
		state[at] = (state[at] ^ (state[at-1]^state[at-1]>>pythonMixShift)*pythonKeyMultiplier) + seed
		advance()
	}
	for range len(state) - 1 {
		state[at] = (state[at] ^ (state[at-1]^state[at-1]>>pythonMixShift)*pythonFinalMultiplier) - uint32(at)
		advance()
	}
	state[0] = pythonTopBit
	return &ifevalPythonRandom{state: state, index: len(state)}
}

// shuffle reorders values as CPython's random.shuffle does: from the last
// position down, each swaps with one drawn at or below it.
func (r *ifevalPythonRandom) shuffle(values []int) {
	for last := len(values) - 1; last > 0; last-- {
		drawn := r.choice(last + 1)
		values[last], values[drawn] = values[drawn], values[last]
	}
}

func (r *ifevalPythonRandom) next() uint32 {
	if r.index >= len(r.state) {
		for i := range r.state {
			y := r.state[i]&0x80000000 | r.state[(i+1)%len(r.state)]&0x7fffffff
			r.state[i] = r.state[(i+397)%len(r.state)] ^ y>>1
			if y&1 != 0 {
				r.state[i] ^= 0x9908b0df
			}
		}
		r.index = 0
	}
	y := r.state[r.index]
	r.index++
	y ^= y >> 11
	y ^= y << 7 & 0x9d2c5680
	y ^= y << 15 & 0xefc60000
	y ^= y >> 18
	return y
}
func (r *ifevalPythonRandom) uniform() float64 {
	return (float64(r.next()>>5)*67108864 + float64(r.next()>>6)) / 9007199254740992
}
func (r *ifevalPythonRandom) choice(n int) int {
	count := bits.Len(uint(n))
	for {
		value := int(r.next() >> uint(32-count))
		if value < n {
			return value
		}
	}
}
func (r *ifevalPythonRandom) gauss() float64 {
	if r.hasCached {
		r.hasCached = false
		return r.cached
	}
	angle := r.uniform() * 2 * math.Pi
	radius := math.Sqrt(-2 * math.Log(1-r.uniform()))
	r.cached = math.Sin(angle) * radius
	r.hasCached = true
	return math.Cos(angle) * radius
}
