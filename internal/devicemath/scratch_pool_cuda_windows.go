//go:build windows

package devicemath

import (
	"errors"
	"fmt"

	"overgo/internal/cuda/driver"
)

// Stack scratch: one device region; layer-local lifetime.

// scratchArena: preallocated bump storage; stream-ordered reuse.
type scratchArena struct {
	base     driver.DevicePtr
	capElems int
	cur      int
	high     int // Measurement only.
}

// alloc: next region; fail on extent drift.
func (a *scratchArena) alloc(n int) (driver.DevicePtr, error) {
	if n <= 0 {
		return 0, errors.New("scratchArena: non-positive alloc")
	}
	if a.cur+n > a.capElems {
		return 0, fmt.Errorf("scratchArena overflow: need %d more, cap %d, cursor %d (scratch size helper drifted from ops)", n, a.capElems, a.cur)
	}
	p := ptrAt(a.base, a.cur)
	a.cur += n
	if a.cur > a.high {
		a.high = a.cur
	}
	return p, nil
}

func (a *scratchArena) reset() { a.cur = 0 }

// forwardScratchElems: attention and MLP scratch.
func forwardScratchElems(d layerDims) int {
	return 2 * d.seq * d.hidden
}

// backwardScratchElems: transient VJP storage:
// 4*(seq*inter) + 9*(seq*hidden) + 7*(seq*width) + 6*(seq*kvWidth)
// + 4*(heads*seq*seq) + optional bias-transpose (seq*width).
// Optional bias transpose included.
func backwardScratchElems(d layerDims) int {
	return 4*d.seq*d.inter + 9*d.seq*d.hidden + 8*d.seq*d.width + 6*d.seq*d.kvWidth + 4*d.heads*d.seq*d.seq
}

// maxScratchElems: larger layer pass.
func maxScratchElems(d layerDims) int {
	if f := forwardScratchElems(d); f > backwardScratchElems(d) {
		return f
	}
	return backwardScratchElems(d)
}
