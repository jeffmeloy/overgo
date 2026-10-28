package executor

import (
	"cmp"
	"errors"
	"math/bits"
	"slices"

	"overgo/internal/checked"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
)

type deviceBufferLease struct {
	pointer driver.DevicePtr
	size    uint64
}

// deviceBufferPool keeps released device buffers in free lists keyed by
// size class for reuse. Capacity requests round up to a size class so
// requests that differ by a few tokens share a list, and the free lists
// together hold at most a share of the device's memory: allocation and
// release trim them largest-first. Before
// both, a scoring pass whose every prompt had a new length left one set
// of exact-size cache pages per length in the lists and filled a 49 GB
// device within thirty seconds.
type deviceBufferPool struct {
	reserve func(*device.State, uint64) error
	// reclaim takes back from the expert residency a free buffer lent it (lendFree), which lent names, before the
	// buffer is reused or freed.
	reclaim     func(*device.State, driver.DevicePtr, uint64) error
	lent        map[driver.DevicePtr]bool
	free        map[uint64][]driver.DevicePtr
	allocations []deviceBufferLease
	freeBytes   uint64
	freeLimit   uint64
}

const (
	// deviceBufferExactLimit: requests up to this size keep their aligned
	// exact size; above it they round up to a size class.
	deviceBufferExactLimit = uint64(1) << 20
	// deviceBufferClassSteps: size classes per power-of-two octave above
	// the exact limit; a request rounds up to the next multiple of an
	// eighth of its octave, so a class wastes at most an eighth.
	deviceBufferClassSteps = uint64(8)
	// deviceBufferFreeShare: the free lists hold at most this fraction of
	// the device's memory; a fresh allocation trims them past it. Memory
	// they hold is lost to a file-backed expert residency, which fills only
	// what the card reports free: at an eighth, two 16K-token prompts left
	// Flash-Next's pool 5.1 GB of idle buffers and the card 27% fewer experts.
	// It trades the allocations a reuse saves (each cleared by Windows, about 18 ms a GB) against the experts idle
	// buffers hold off the card.
	deviceBufferFreeShare = uint64(64)
)

func (p *deviceBufferPool) acquire(state *device.State, size uint64) (deviceBufferLease, error) {
	bucket, err := deviceBufferBucket(size)
	if err != nil {
		return deviceBufferLease{}, err
	}
	return p.acquireBucket(state, bucket)
}

func (p *deviceBufferPool) acquireExact(state *device.State, size uint64) (deviceBufferLease, error) {
	bucket, ok := deviceBufferClass(size)
	if !ok {
		return deviceBufferLease{}, errors.New("CUDA buffer size is invalid")
	}
	return p.acquireBucket(state, bucket)
}

// acquireSized takes a buffer of the request's aligned size, no size class (AllocateFixedDeviceBuffer).
func (p *deviceBufferPool) acquireSized(state *device.State, size uint64) (deviceBufferLease, error) {
	aligned, ok := checked.Align(size, deviceAllocationAlignment)
	if !ok || aligned == 0 {
		return deviceBufferLease{}, errors.New("CUDA buffer size is invalid")
	}
	return p.acquireBucket(state, aligned)
}

// deviceBufferClass: the size class a capacity request occupies: its
// aligned size up to the exact limit, above it the next multiple of an
// eighth of its power-of-two octave.
func deviceBufferClass(size uint64) (uint64, bool) {
	aligned, ok := checked.Align(size, deviceAllocationAlignment)
	if !ok || aligned == 0 {
		return 0, false
	}
	if aligned <= deviceBufferExactLimit {
		return aligned, true
	}
	octave := uint64(1) << bits.Len64(aligned-1)
	step := octave / deviceBufferClassSteps
	return checked.Align(aligned, step)
}

func (p *deviceBufferPool) acquireBucket(
	state *device.State,
	bucket uint64,
) (deviceBufferLease, error) {
	// The lowest free address, not the last released: a request then
	// takes the same buffers whatever order the last one released them
	// in, so the graphs it captured name them again.
	if available := p.free[bucket]; len(available) > 0 {
		lowest := 0
		for index, pointer := range available {
			if pointer < available[lowest] {
				lowest = index
			}
		}
		pointer := available[lowest]
		if err := p.takeBack(state, pointer, bucket); err != nil {
			return deviceBufferLease{}, err
		}
		p.free[bucket] = slices.Delete(available, lowest, lowest+1)
		p.freeBytes -= bucket
		return deviceBufferLease{pointer: pointer, size: bucket}, nil
	}
	if err := p.trim(state, bucket); err != nil {
		return deviceBufferLease{}, err
	}
	allocate := func() (driver.DevicePtr, error) {
		if p.reserve != nil {
			if err := p.reserve(state, bucket); err != nil {
				return 0, err
			}
		}
		return state.Driver.MemAlloc(bucket)
	}
	pointer, err := allocate()
	if driver.IsOutOfMemory(err) && p.freeBytes > 0 {
		before := p.freeBytes
		if releaseErr := p.trim(state, p.freeLimit); releaseErr != nil {
			return deviceBufferLease{}, errors.Join(err, releaseErr)
		}
		if p.freeBytes < before {
			pointer, err = allocate()
		}
	}
	if err != nil {
		return deviceBufferLease{}, err
	}
	if p.free == nil {
		p.free = make(map[uint64][]driver.DevicePtr)
	}
	lease := deviceBufferLease{pointer: pointer, size: bucket}
	p.allocations = append(p.allocations, lease)
	return lease, nil
}

// trim frees free-list buffers, largest class first, until the lists
// plus the incoming allocation fit the pool's share of device memory.
func (p *deviceBufferPool) trim(state *device.State, incoming uint64) error {
	if p.freeLimit == 0 {
		_, total, err := state.Driver.MemInfo()
		if err != nil || total == 0 {
			return nil
		}
		p.freeLimit = total / deviceBufferFreeShare
	}
	for p.freeBytes > 0 && (incoming >= p.freeLimit || p.freeBytes > p.freeLimit-incoming) {
		largest := uint64(0)
		for size, pointers := range p.free {
			if len(pointers) > 0 && size > largest {
				largest = size
			}
		}
		if largest == 0 {
			return nil
		}
		pointers := p.free[largest]
		pointer := pointers[len(pointers)-1]
		if err := p.takeBack(state, pointer, largest); err != nil {
			return err
		}
		if err := state.Driver.MemFree(pointer); err != nil {
			return err
		}
		p.free[largest] = pointers[:len(pointers)-1]
		p.freeBytes -= largest
		p.allocations = slices.DeleteFunc(p.allocations, func(lease deviceBufferLease) bool {
			return lease.pointer == pointer
		})
	}
	return nil
}

func (p *deviceBufferPool) release(state *device.State, leases ...deviceBufferLease) error {
	for _, lease := range leases {
		if lease.pointer != 0 {
			p.free[lease.size] = append(p.free[lease.size], lease.pointer)
			p.freeBytes += lease.size
		}
	}
	return p.trim(state, 0)
}

func (p *deviceBufferPool) close(state *device.State) error {
	var errs []error
	for _, lease := range p.allocations {
		if err := p.takeBack(state, lease.pointer, lease.size); err != nil {
			errs = append(errs, err)
		}
		if err := state.Driver.MemFree(lease.pointer); err != nil {
			errs = append(errs, err)
		}
	}
	p.free = nil
	p.allocations = nil
	p.lent = nil
	return errors.Join(errs...)
}

// deviceBufferLendBytes is the smallest free buffer the pool lends the expert residency: one expert of Flash-Next's
// widest stride (3.5 MB) fits it.
const deviceBufferLendBytes = 4 << 20

// lendFree lends the expert residency, through lend, the free buffers of deviceBufferLendBytes or more it has not lent,
// in address order, as a prompt's read lends its arena and staging until the next read: the free lists kept a read's
// buffers idle through every decode after it (Flash-Next's round about 390 MiB, some 170 experts). Each comes back
// (takeBack) before the pool reuses or frees it.
func (p *deviceBufferPool) lendFree(lend func(driver.DevicePtr, uint64) error) error {
	var idle []deviceBufferLease
	for size, pointers := range p.free {
		for _, pointer := range pointers {
			if size >= deviceBufferLendBytes && !p.lent[pointer] {
				idle = append(idle, deviceBufferLease{pointer: pointer, size: size})
			}
		}
	}
	slices.SortFunc(idle, func(a, b deviceBufferLease) int { return cmp.Compare(a.pointer, b.pointer) })
	for _, lease := range idle {
		if err := lend(lease.pointer, lease.size); err != nil {
			return err
		}
		if p.lent == nil {
			p.lent = make(map[driver.DevicePtr]bool)
		}
		p.lent[lease.pointer] = true
	}
	return nil
}

// takeBack takes a free buffer lent to the expert residency back, its experts moved out, before the pool reuses or
// frees it; a buffer not lent is the pool's already.
func (p *deviceBufferPool) takeBack(state *device.State, pointer driver.DevicePtr, size uint64) error {
	if !p.lent[pointer] {
		return nil
	}
	if p.reclaim != nil {
		if err := p.reclaim(state, pointer, size); err != nil {
			return err
		}
	}
	delete(p.lent, pointer)
	return nil
}

// graphReserveBytes is the card memory a graph's instantiation is given room for before its capture: up to 16 MiB of
// Flash-Next's card (its 2,673-launch span graphs 12 to 16 MiB).
const graphReserveBytes = 32 << 20

// reserveGraph makes room through the reservation for a graph about to be captured, whose instantiation allocates
// outside the pool: the expert residency leaves the card's memory nearly full, a reservation freeing about what it
// needs; without a reservation it is the card's own.
func (p *deviceBufferPool) reserveGraph(state *device.State) error {
	if p.reserve == nil {
		return nil
	}
	return p.reserve(state, graphReserveBytes)
}

// takeBackAll takes back every buffer lent, as the residency it was lent to is replaced.
func (p *deviceBufferPool) takeBackAll(state *device.State) error {
	for _, lease := range p.allocations {
		if err := p.takeBack(state, lease.pointer, lease.size); err != nil {
			return err
		}
	}
	return nil
}

func deviceBufferBucket(size uint64) (uint64, error) {
	if size == 0 {
		return 0, errors.New("CUDA buffer size is zero")
	}
	if size <= deviceAllocationAlignment {
		return deviceAllocationAlignment, nil
	}
	if size > uint64(1)<<63 {
		return size, nil
	}
	return uint64(1) << bits.Len64(size-1), nil
}
