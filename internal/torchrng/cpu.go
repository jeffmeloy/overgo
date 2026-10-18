package torchrng

import (
	"fmt"
	"math"
)

// The reference draws its starting noise with torch.randn from a CPU
// generator seeded by manual_seed. The generator is MT19937: its state is
// seeded from the seed's low word, twisted every stateWords draws, and each
// draw is tempered before use.
const (
	stateWords     = 624
	twistOffset    = 397
	seedMultiplier = 1812433253
	seedShift      = 30
	twistMatrix    = 0x9908b0df
	upperBit       = 0x80000000
	lowerBits      = 0x7fffffff
	temperShiftA   = 11
	temperShiftB   = 7
	temperMaskB    = 0x9d2c5680
	temperShiftC   = 15
	temperMaskC    = 0xefc60000
	temperShiftD   = 18
)

// A float32 uniform in [0, 1) is a draw's low mantissa-width bits over two
// to that width; torch fills the whole tensor with them before turning them
// into normals.
const (
	uniformBits  = 24
	uniformScale = 1 << uniformBits
	uniformMask  = uniformScale - 1
)

// Box-Muller turns sixteen uniforms at a time into normals: the first half
// give the radii, the second half the angles, and each pair yields a cosine
// normal in the first half and a sine normal in the second.
const (
	normalChunk = 16
	normalHalf  = normalChunk / 2
	logScale    = -2
	fullTurn    = 2 * math.Pi
)

// mersenne is the reference generator's MT19937 state.
type mersenne struct {
	state [stateWords]uint32
	next  int
}

// newMersenne seeds the state as torch's manual_seed does, from the seed's
// low word.
func newMersenne(seed int64) *mersenne {
	m := &mersenne{next: stateWords}
	m.state[0] = uint32(seed)
	for index := 1; index < stateWords; index++ {
		previous := m.state[index-1]
		m.state[index] = seedMultiplier*(previous^(previous>>seedShift)) + uint32(index)
	}
	return m
}

// word returns the next tempered draw, twisting the state when it is spent.
func (m *mersenne) word() uint32 {
	if m.next == stateWords {
		for index := range stateWords {
			mixed := m.state[index]&upperBit | m.state[(index+1)%stateWords]&lowerBits
			value := m.state[(index+twistOffset)%stateWords] ^ mixed>>1
			if mixed&1 != 0 {
				value ^= twistMatrix
			}
			m.state[index] = value
		}
		m.next = 0
	}
	y := m.state[m.next]
	m.next++
	y ^= y >> temperShiftA
	y ^= y << temperShiftB & temperMaskB
	y ^= y << temperShiftC & temperMaskC
	return y ^ y>>temperShiftD
}

// uniform returns the next float32 uniform in [0, 1).
func (m *mersenne) uniform() float32 {
	return float32(m.word()&uniformMask) / uniformScale
}

// CPUNormal draws contiguous float32 normals with pinned PyTorch CPU AVX2 semantics.
//   - The seed initializes a fresh MT19937 generator from its low word.
//   - Counts below one reference chunk are refused because they use a different Torch algorithm.
//   - Scalar float32 arithmetic preserves the pinned AVX2 result on every host.
//   - Partial chunks redraw the last complete chunk as the reference does.
func CPUNormal(seed int64, count int) ([]float32, error) {
	if count < normalChunk {
		return nil, fmt.Errorf("torchrng: seeded CPU noise of %d values is shorter than torch's %d-value fill", count, normalChunk)
	}
	generator := newMersenne(seed)
	values := make([]float32, count)
	for index := range values {
		values[index] = generator.uniform()
	}
	for start := 0; start+normalChunk <= count; start += normalChunk {
		boxMuller(values[start : start+normalChunk])
	}
	if count%normalChunk != 0 {
		tail := values[count-normalChunk:]
		for index := range tail {
			tail[index] = generator.uniform()
		}
		boxMuller(tail)
	}
	return values, nil
}

// boxMuller turns one chunk of uniforms into normals in place, in the
// reference's float32 arithmetic.
func boxMuller(chunk []float32) {
	for index := range normalHalf {
		radius := float32(math.Sqrt(float64(logScale * cpuLog(1-chunk[index]))))
		theta := float32(fullTurn) * chunk[index+normalHalf]
		s, c := cpuSinCos(theta)
		chunk[index] = radius * c
		chunk[index+normalHalf] = radius * s
	}
}
