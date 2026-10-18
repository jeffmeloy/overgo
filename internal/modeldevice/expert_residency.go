package modeldevice

import (
	"container/list"
	"context"
	"errors"
	"fmt"
	"unsafe"

	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	"overgo/internal/gguf"
	"overgo/internal/tensor"
)

type expertKey struct {
	bank  string
	index uint32
}

const (
	// expertBankRank is the GGUF layout with input, output and expert axes.
	expertBankRank = 3
	// expertBankAxis is the expert axis of the GGUF expert-bank layout.
	expertBankAxis = expertBankRank - 1
	// expertAllocationFailure is CUDA_ERROR_OUT_OF_MEMORY from the CUDA Driver API.
	// Keep it aligned with the driver result contract used by allocation retry.
	expertAllocationFailure = 2
)

type residentExpert struct {
	key     expertKey
	pointer driver.DevicePtr
	bytes   uint64
}

type expertBank struct {
	info     gguf.TensorInfo
	table    driver.DevicePtr
	pointers []driver.DevicePtr
	bytes    uint64
}

// expertResidency holds encoded experts under one byte limit on the CUDA worker.
// The immutable GGUF is the backing copy for every evicted expert.
type expertResidency struct {
	file                     *gguf.File
	state                    *device.State
	mem                      deviceMemory
	banks                    map[string]*expertBank
	entries                  map[expertKey]*list.Element
	order                    list.List
	limit, bytes, tableBytes uint64
	staging                  []byte
}

// EnableExpertResidency binds file-backed banks after the ordinary weight load succeeds.
// The limit includes pointer tables and encoded experts but excludes other weight stores.
func (w *DeviceWeights) EnableExpertResidency(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo, limit uint64) error {
	if w == nil || w.worker == nil || file == nil || len(infos) == 0 {
		return errors.New("expert residency requires a worker, file and bank descriptors")
	}
	return w.worker.Do(ctx, func(state *device.State) error {
		if w.experts != nil {
			return errors.New("expert residency is already configured")
		}
		if limit == 0 {
			free, _, err := state.Driver.MemInfo()
			if err != nil {
				return err
			}
			limit = free
		}
		cache, err := newExpertResidency(ctx, file, infos, limit, stateMemory{state})
		if err != nil {
			return err
		}
		cache.state = state
		w.experts = cache
		return nil
	})
}

func newExpertResidency(ctx context.Context, file *gguf.File, infos []gguf.TensorInfo, limit uint64, mem deviceMemory) (*expertResidency, error) {
	c := &expertResidency{file: file, mem: mem, limit: limit, banks: make(map[string]*expertBank), entries: make(map[expertKey]*list.Element)}
	for _, info := range infos {
		listed, found := file.Tensor(info.Name)
		if !found || listed != info || info.Dimensions != expertBankRank || info.Shape[expertBankAxis] == 0 || info.Shape[expertBankAxis] > uint64(^uint32(0)) || info.Size%info.Shape[expertBankAxis] != 0 || c.banks[info.Name] != nil {
			return nil, fmt.Errorf("expert bank %q requires one unique rank-three GGUF descriptor", info.Name)
		}
		// Each table entry is a CUDA device address, whose ABI is uint64.
		bytes := info.Shape[expertBankAxis] * uint64(unsafe.Sizeof(driver.DevicePtr(0)))
		if bytes > limit-c.tableBytes {
			return nil, errors.New("expert pointer tables exceed the residency limit")
		}
		c.tableBytes += bytes
		c.banks[info.Name] = &expertBank{info: info, pointers: make([]driver.DevicePtr, info.Shape[expertBankAxis]), bytes: info.Size / info.Shape[expertBankAxis]}
	}
	c.bytes = c.tableBytes
	for _, bank := range c.banks {
		pointer, err := mem.alloc(ctx, uint64(len(bank.pointers))*uint64(unsafe.Sizeof(driver.DevicePtr(0))))
		if err != nil {
			return nil, errors.Join(err, c.close())
		}
		bank.table = pointer
	}
	return c, nil
}

// IsExpertBank reports a bank whose bytes are resolved after routing on the owning worker.
func (w *DeviceWeights) IsExpertBank(name string) bool {
	return w != nil && w.experts != nil && w.experts.banks[name] != nil
}

// HasExpertResidency distinguishes file-backed expert execution from full device residency.
func (w *DeviceWeights) HasExpertResidency() bool { return w != nil && w.experts != nil }

// ExpertMemory reports reclaimable encoded bytes and the routed working-set limit.
// Pointer tables remain resident and are excluded from both values.
func (w *DeviceWeights) ExpertMemory(state *device.State) (reclaimable, capacity uint64, err error) {
	if w == nil || w.experts == nil || w.experts.state != state {
		return 0, 0, errors.New("expert memory query requires its owning worker")
	}
	return w.experts.bytes - w.experts.tableBytes, w.experts.limit - w.experts.tableBytes, nil
}

// ReserveExpertMemory makes room for another executor owner before allocation.
// Only completed expert reads are evicted and the GGUF remains authoritative.
func (w *DeviceWeights) ReserveExpertMemory(state *device.State, required uint64) error {
	if w == nil || w.experts == nil || w.experts.state != state {
		return errors.New("expert memory reservation requires its owning worker")
	}
	free, _, err := state.Driver.MemInfo()
	if err != nil || free >= required {
		return err
	}
	if err = state.Driver.StreamSynchronize(state.Stream); err != nil {
		return err
	}
	for free < required && w.experts.order.Len() > 0 {
		if _, err = w.experts.reclaim(w.experts.order.Front(), 0); err != nil {
			return err
		}
		free, _, err = state.Driver.MemInfo()
		if err != nil {
			return err
		}
	}
	if free < required {
		return fmt.Errorf("expert memory reservation needs %d free bytes, available %d: %w", required, free, &driver.ResultError{Operation: "expert reservation", Code: expertAllocationFailure})
	}
	return nil
}

// Close releases expert tables and encoded experts after their actual device completion.
func (w *DeviceWeights) Close() error {
	if w == nil {
		return nil
	}
	if w.experts != nil {
		err := w.worker.Do(context.Background(), func(state *device.State) error {
			if err := state.Driver.StreamSynchronize(state.Stream); err != nil {
				return err
			}
			if err := w.experts.close(); err != nil {
				return err
			}
			w.experts = nil
			return nil
		})
		if err != nil {
			return err
		}
	}
	return w.deviceTensorStore.Close()
}

func (w *DeviceWeights) expertInput(builder *tensor.Builder, name string) (*tensor.Tensor, driver.DevicePtr, error) {
	bank := w.experts.banks[name]
	shape, err := tensor.NewShape(bank.info.Shape[:bank.info.Dimensions]...)
	if err != nil {
		return nil, 0, err
	}
	node := builder.Input(name, bank.info.Type, shape)
	return node, bank.table, builder.Err()
}

// ResolveExpertBanks returns indexed device tables for the exact routed experts.
// The caller must synchronize the owning stream before resolving or evicting banks.
// The returned tables remain valid through the next resolution on that worker.
func (w *DeviceWeights) ResolveExpertBanks(ctx context.Context, state *device.State, nodes []*tensor.Tensor, indices []uint32) ([]driver.DevicePtr, error) {
	if w == nil || w.experts == nil || w.experts.state != state {
		return nil, errors.New("expert resolution requires its owning CUDA worker")
	}
	names := make([]string, len(nodes))
	for i, node := range nodes {
		bank := w.experts.banks[node.Name]
		if bank == nil || node.Type != bank.info.Type || uint32(node.Shape.Rank) != bank.info.Dimensions {
			return nil, fmt.Errorf("expert bank %q differs from its file descriptor", node.Name)
		}
		for axis := range bank.info.Dimensions {
			if node.Shape.Dims[axis] != bank.info.Shape[axis] {
				return nil, fmt.Errorf("expert bank %q shape differs", node.Name)
			}
		}
		names[i] = node.Name
	}
	return w.experts.resolve(ctx, names, indices)
}

func (c *expertResidency) resolve(ctx context.Context, names []string, indices []uint32) ([]driver.DevicePtr, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// The device's free memory narrows the limit only when an expert must be
	// loaded, so a step whose experts are all resident queries nothing.
	limit := c.limit
	refined := c.state == nil
	needed := make(map[expertKey]bool, len(names)*len(indices))
	var keys []expertKey
	bytes := c.tableBytes
	for _, name := range names {
		bank := c.banks[name]
		if bank == nil {
			return nil, fmt.Errorf("expert bank %q is unavailable", name)
		}
		for _, index := range indices {
			if uint64(index) >= bank.info.Shape[expertBankAxis] {
				return nil, fmt.Errorf("expert %d leaves bank %q", index, name)
			}
			key := expertKey{name, index}
			if needed[key] {
				continue
			}
			if bytes > limit || bank.bytes > limit-bytes {
				return nil, errors.New("routed experts exceed the residency limit")
			}
			bytes += bank.bytes
			needed[key] = true
			keys = append(keys, key)
		}
	}
	reclaimUnused := func(size uint64) (driver.DevicePtr, error) {
		var victim *list.Element
		for entry := c.order.Front(); entry != nil; entry = entry.Next() {
			value := entry.Value.(residentExpert)
			if needed[value.key] {
				continue
			}
			if victim == nil {
				victim = entry
			}
			if value.bytes == size {
				victim = entry
				break
			}
		}
		if victim == nil {
			return 0, errors.New("expert residency has no releasable entry")
		}
		return c.reclaim(victim, size)
	}
	for _, key := range keys {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if entry := c.entries[key]; entry != nil {
			c.order.MoveToBack(entry)
			continue
		}
		if !refined {
			free, _, err := c.state.Driver.MemInfo()
			if err != nil {
				return nil, err
			}
			limit, refined = min(limit, c.bytes+free), true
			if bytes > limit {
				return nil, errors.New("routed experts exceed the residency limit")
			}
		}
		bank := c.banks[key.bank]
		if uint64(cap(c.staging)) < bank.bytes {
			c.staging = make([]byte, bank.bytes)
		}
		data := c.staging[:bank.bytes]
		if err := c.file.ReadTensorRange(bank.info, uint64(key.index)*bank.bytes, data); err != nil {
			return nil, err
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var pointer driver.DevicePtr
		var err error
		for {
			if c.bytes <= limit-bank.bytes {
				pointer, err = c.mem.alloc(ctx, bank.bytes)
				if !driver.IsOutOfMemory(err) {
					break
				}
			}
			// A completed equal-sized victim supplies the allocation directly.
			// Other victims release physical capacity before the next allocation.
			var releaseErr error
			pointer, releaseErr = reclaimUnused(bank.bytes)
			if releaseErr != nil {
				return nil, errors.Join(err, releaseErr)
			}
			if pointer != 0 {
				err = nil
				break
			}
			if err = ctx.Err(); err != nil {
				return nil, err
			}
		}
		if err != nil {
			return nil, err
		}
		if err = c.mem.write(ctx, pointer, data); err == nil {
			err = ctx.Err()
		}
		if err != nil {
			return nil, errors.Join(err, c.mem.free(context.Background(), []driver.DevicePtr{pointer}))
		}
		c.entries[key] = c.order.PushBack(residentExpert{key, pointer, bank.bytes})
		c.bytes += bank.bytes
	}
	tables := make([]driver.DevicePtr, len(names))
	for i, name := range names {
		bank := c.banks[name]
		clear(bank.pointers)
		for _, index := range indices {
			bank.pointers[index] = c.entries[expertKey{name, index}].Value.(residentExpert).pointer
		}
		if err := c.mem.write(ctx, bank.table, driver.Bytes(bank.pointers)); err != nil {
			return nil, err
		}
		tables[i] = bank.table
	}
	return tables, ctx.Err()
}

func (c *expertResidency) reclaim(entry *list.Element, size uint64) (driver.DevicePtr, error) {
	value := entry.Value.(residentExpert)
	if value.bytes != size {
		if err := c.mem.free(context.Background(), []driver.DevicePtr{value.pointer}); err != nil {
			return 0, err
		}
		value.pointer = 0
	}
	delete(c.entries, value.key)
	c.order.Remove(entry)
	c.bytes -= value.bytes
	return value.pointer, nil
}

func (c *expertResidency) close() error {
	if c == nil {
		return nil
	}
	for c.order.Len() != 0 {
		if _, err := c.reclaim(c.order.Front(), 0); err != nil {
			return err
		}
	}
	for _, bank := range c.banks {
		if bank.table == 0 {
			continue
		}
		if err := c.mem.free(context.Background(), []driver.DevicePtr{bank.table}); err != nil {
			return err
		}
		bank.table = 0
	}
	c.staging = nil
	return nil
}

// stateMemory performs synchronous transfers inside the existing worker callback.
type stateMemory struct{ state *device.State }

func (m stateMemory) alloc(ctx context.Context, size uint64) (driver.DevicePtr, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return m.state.Driver.MemAlloc(size)
}
func (m stateMemory) write(ctx context.Context, pointer driver.DevicePtr, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	return m.state.Driver.MemcpyHtoD(pointer, data)
}
func (m stateMemory) free(_ context.Context, pointers []driver.DevicePtr) error {
	var errs []error
	for _, pointer := range pointers {
		errs = append(errs, m.state.Driver.MemFree(pointer))
	}
	return errors.Join(errs...)
}
