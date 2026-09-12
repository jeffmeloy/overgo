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
