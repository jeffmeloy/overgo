package executor

import (
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
// together hold at most a share of the device's memory: a fresh
// allocation trims them largest-first before it asks the driver. Before
// both, a scoring pass whose every prompt had a new length left one set
// of exact-size cache pages per length in the lists and filled a 49 GB
// device within thirty seconds.
type deviceBufferPool struct {
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
	// the device's memory; a fresh allocation trims them past it.
	deviceBufferFreeShare = uint64(8)
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
	if available := p.free[bucket]; len(available) > 0 {
		pointer := available[len(available)-1]
		p.free[bucket] = available[:len(available)-1]
		p.freeBytes -= bucket
		return deviceBufferLease{pointer: pointer, size: bucket}, nil
	}
	if err := p.trim(state, bucket); err != nil {
		return deviceBufferLease{}, err
	}
	pointer, err := state.Driver.MemAlloc(bucket)
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
	for p.freeBytes > 0 && p.freeBytes+incoming > p.freeLimit {
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
		p.free[largest] = pointers[:len(pointers)-1]
		p.freeBytes -= largest
		if err := state.Driver.MemFree(pointer); err != nil {
			return err
		}
		p.allocations = slices.DeleteFunc(p.allocations, func(lease deviceBufferLease) bool {
			return lease.pointer == pointer
		})
	}
	return nil
}

func (p *deviceBufferPool) release(lease deviceBufferLease) {
	if lease.pointer != 0 {
		p.free[lease.size] = append(p.free[lease.size], lease.pointer)
		p.freeBytes += lease.size
	}
}

func (p *deviceBufferPool) close(state *device.State) error {
	var errs []error
	for _, lease := range p.allocations {
		if err := state.Driver.MemFree(lease.pointer); err != nil {
			errs = append(errs, err)
		}
	}
	p.free = nil
	p.allocations = nil
	return errors.Join(errs...)
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
