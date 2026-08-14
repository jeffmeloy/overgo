// Package torchrng is a family-neutral, PyTorch-bit-exact CUDA normal source.
//
// It ports the seed-matched TorchCUDARandn generator from adaptive_new
// (go/extmodel NewTorchCUDARandnCUDAStream): a Philox counter-based kernel plus
// the curand float4 Box-Muller fill-order that make the output byte-match
// torch.randn(seed) on CUDA. A single Stream advances a Philox counter across
// successive fills so N draws split across calls equal one contiguous
// torch.randn sequence -- the primitive image/video generation lanes need to
// reproduce a seeded initial latent.
//
// The launch geometry and counter advance in this file are pure and portable
// (no CUDA), so they are unit-testable without a device; the device fill lives
// in stream_cuda_windows.go.
package torchrng

import (
	"fmt"
	"math"
)

const (
	// Block: threads per block. PyTorch's distribution launches use 256 and the
	// Philox subsequence math is tied to it -- not a tunable.
	Block = 256
	// unroll: float4 draws per thread per round (curand_normal4).
	unroll = 4
	// offsetPerRound: Philox counter elements consumed per thread per round.
	// curand advances the 128-bit counter by 4 per curand_normal4 call.
	offsetPerRound = 4
)

// NullaryPolicy derives the launch grid and the Philox counter advance for an
// element count and the device execution limits, matching PyTorch's nullary
// distribution launcher. Ported verbatim from adaptive_new TorchCUDANullaryPolicy;
// the geometry is what makes the counter advance reproduce torch.randn.
func NullaryPolicy(elements int64, smCount, maxThreadsPerSM int) (grid, block int, counterOffset uint64, err error) {
	if elements <= 0 {
		return 0, 0, 0, fmt.Errorf("torchrng policy: elements must be positive, got %d", elements)
	}
	if smCount <= 0 || maxThreadsPerSM <= 0 {
		return 0, 0, 0, fmt.Errorf("torchrng policy: bad device sm=%d max_threads_per_sm=%d", smCount, maxThreadsPerSM)
	}
	block = Block
	blocksPerSM := maxThreadsPerSM / block
	if blocksPerSM <= 0 {
		return 0, 0, 0, fmt.Errorf("torchrng policy: max_threads_per_sm=%d below block=%d", maxThreadsPerSM, block)
	}
	ceilDiv := func(a, b int64) int64 { return (a + b - 1) / b }
	needed := ceilDiv(elements, int64(block))
	if maxGrid := int64(smCount * blocksPerSM); needed > maxGrid {
		needed = maxGrid
	}
	if needed <= 0 {
		needed = 1
	}
	grid = int(needed)
	stride := int64(block * grid * unroll)
	counterOffset = uint64(ceilDiv(elements, stride) * offsetPerRound)
	return grid, block, counterOffset, nil
}

// Stream is a seeded torch.randn-compatible normal source. It holds a Philox
// counter offset that advances across fills so split draws equal one contiguous
// torch.randn(seed) sequence. Zero value is not usable; construct with NewStream.
type Stream struct {
	seed   uint64
	offset uint64

	// device cache: populated lazily by the first device fill and valid only
	// within the worker/context that loaded them (see stream_cuda_windows.go).
	loaded       bool
	module       uintptr
	function     uintptr
	smCount      int
	threadsPerSM int
}

// NewStream returns a Stream seeded like torch.manual_seed(seed) for the CUDA
// generator. The counter starts at offset 0.
func NewStream(seed int64) *Stream {
	return &Stream{seed: uint64(seed)}
}

// Offset returns the current Philox counter offset (advances after each fill).
func (s *Stream) Offset() uint64 {
	if s == nil {
		return 0
	}
	return s.offset
}

// reserve computes the launch grid for elements at the CURRENT counter offset,
// returns that draw offset, and advances the stream's offset by the policy's
// counter step. Pure: no device work, so the counter arithmetic is testable
// without CUDA. Overflow is an error rather than a silent wrap.
func (s *Stream) reserve(elements int64, smCount, maxThreadsPerSM int) (grid int, drawOffset uint64, err error) {
	grid, _, advance, err := NullaryPolicy(elements, smCount, maxThreadsPerSM)
	if err != nil {
		return 0, 0, err
	}
	if math.MaxUint64-s.offset < advance {
		return 0, 0, fmt.Errorf("torchrng: Philox counter overflow (offset=%d advance=%d)", s.offset, advance)
	}
	drawOffset = s.offset
	s.offset += advance
	return grid, drawOffset, nil
}
