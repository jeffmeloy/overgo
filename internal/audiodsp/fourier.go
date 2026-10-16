package audiodsp

import (
	"errors"
	"math"
	"math/bits"
)

// fourierRadix is the binary butterfly width and Bluestein mirror period factor.
// Reconsider if the transform uses a different radix.
const fourierRadix = 2

// fourierPlan owns float64 roots and, for non-power-of-two lengths, a Bluestein convolution.
//   - The convolution has at least fourierRadix*N-1 entries to avoid circular aliasing.
//   - Execution borrows one workspace and allocates nothing.
type fourierPlan struct {
	length, size         int
	roots, chirp, kernel []complex128
}

func fourierExtent(length int) (size, chirps int, err error) {
	if length&(length-1) == 0 {
		return length, 0, nil
	}
	if length > math.MaxInt/fourierRadix {
		return 0, 0, errors.New("audio frontend: Fourier convolution extent overflows")
	}
	power := bits.Len(uint(fourierRadix*length - fourierRadix))
	if power >= bits.UintSize-1 {
		return 0, 0, errors.New("audio frontend: Fourier convolution extent overflows")
	}
	return 1 << power, length, nil
}

func newFourierPlan(length, size, chirps int) fourierPlan {
	p := fourierPlan{length: length, size: size, roots: make([]complex128, size/fourierRadix)}
	for k := range p.roots {
		s, c := math.Sincos(-fourierRadix * math.Pi * float64(k) / float64(size))
		p.roots[k] = complex(c, s)
	}
	if chirps != 0 {
		p.chirp, p.kernel = make([]complex128, chirps), make([]complex128, size)
		// Reduce k squared modulo fourierRadix*N with integer differences before conversion.
		// Budget admission bounds the integer sums as well as the allocations.
		square := 0
		for k := range p.chirp {
			s, c := math.Sincos(-math.Pi * float64(square) / float64(length))
			p.chirp[k] = complex(c, s)
			p.kernel[k] = complex(c, -s)
			if k != 0 {
				p.kernel[size-k] = p.kernel[k]
			}
			square = (square + fourierRadix*k + 1) % (fourierRadix * length)
		}
		p.radixTwo(p.kernel, false)
	}
	return p
}

func (p *fourierPlan) radixTwo(values []complex128, inverse bool) {
	for i, j := 1, 0; i < len(values); i++ {
		bit := len(values) / fourierRadix
		for ; j&bit != 0; bit /= fourierRadix {
			j ^= bit
		}
		j ^= bit
		if i < j {
			values[i], values[j] = values[j], values[i]
		}
	}
	for width := fourierRadix; width <= len(values); width *= fourierRadix {
		half, stride := width/fourierRadix, p.size/width
		for start := 0; start < len(values); start += width {
			for k := range half {
				root := p.roots[k*stride]
				if inverse {
					root = complex(real(root), -imag(root))
				}
				a, b := values[start+k], values[start+k+half]*root
				values[start+k], values[start+k+half] = a+b, a-b
			}
		}
	}
}

func (p *fourierPlan) execute(values []complex128, inverse bool) {
	if inverse {
		for k := range p.length {
			values[k] = complex(real(values[k]), -imag(values[k]))
		}
	}
	if len(p.chirp) == 0 {
		p.radixTwo(values, false)
	} else {
		for k, chirp := range p.chirp {
			values[k] *= chirp
		}
		clear(values[p.length:])
		p.radixTwo(values, false)
		for k, kernel := range p.kernel {
			values[k] *= kernel
		}
		p.radixTwo(values, true)
		for k, chirp := range p.chirp {
			values[k] = values[k] * chirp / complex(float64(p.size), 0)
		}
	}
	if inverse {
		for k := range p.length {
			values[k] = complex(real(values[k]), -imag(values[k])) / complex(float64(p.length), 0)
		}
	}
}
