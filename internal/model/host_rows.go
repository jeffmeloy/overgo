package model

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sync"
	"sync/atomic"

	"overgo/internal/gguf"
	"overgo/internal/quant"
	"overgo/internal/tensor"
	"overgo/internal/tensor/reference"
)

// hostRowReads is the handles of its table's file a HostRows reads through,
// a read in flight on each. Go reads a Windows file through one handle a
// read at a time: Flash-Next's cold n-gram rows took 458 us each through
// one handle and 5 us each through 32 (2,048 rows), and a decode token's 16
// took 9.5 ms and 0.51 ms. Strata's ple_reader keeps up to 256 reads in
// flight.
const hostRowReads = 32

// hostRowWays is the rows a cache set holds, each replaced in turn, as in
// Strata's row cache.
const hostRowWays = 8

// hostRowEmpty marks a cache way that holds no row, and hostRowMix and
// hostRowShift spread a row index over the sets (Fibonacci hashing, as
// Strata's row cache mixes its keys).
const (
	hostRowEmpty = math.MaxUint32
	hostRowMix   = 0x9E3779B97F4A7C15
	hostRowShift = 29
)

// NGramCacheRows is the rows the n-gram table's reader keeps, Strata's
// default: about 90 MB of Flash-Next's 90-byte IQ4_NL rows, of a 28.8 GB
// table no file cache keeps.
const NGramCacheRows = 1 << 20

// HostRows reads selected rows of a table from its GGUF file. A call's rows
// not yet read are read together, through hostRowReads handles of the file
// that holds the table: Flash-Next's decode read a token's sixteen n-gram
// rows one after another through the model's one handle, 0.6 ms each where
// the file cache lacked them. The rows read are kept in a bounded
// set-associative cache. A HostRows serves one call at a time and holds its
// handles until Close.
type HostRows struct {
	file   *gguf.File
	info   gguf.TensorInfo
	layout gguf.TensorRowLayout
	// handles read the table at base; without them the rows are read through file.
	handles []*os.File
	base    uint64
	mu      sync.Mutex
	sets    uint64
	keys    []uint32
	cached  []byte
	next    []uint8
}

// NewHostRows reads info's rows from file through handles of its own,
// keeping at most cacheRows of them, none for zero.
func NewHostRows(file *gguf.File, info gguf.TensorInfo, cacheRows uint64) (*HostRows, error) {
	rows, err := newHostRows(file, info, cacheRows)
	if err != nil {
		return nil, err
	}
	paths, shards := file.SourcePaths(), file.Shards()
	if int(info.Shard) >= len(paths) || int(info.Shard) >= len(shards) {
		return rows, nil
	}
	rows.base = shards[info.Shard].DataOffset + info.Offset
	if rows.base < info.Offset || rows.base > math.MaxInt64-info.Size {
		return nil, errors.New("host rows: the table lies outside a file's range")
	}
	for range hostRowReads {
		handle, err := os.Open(paths[info.Shard])
		if err != nil {
			return nil, errors.Join(fmt.Errorf("host rows: %w", err), rows.Close())
		}
		rows.handles = append(rows.handles, handle)
	}
	return rows, nil
}

// Close closes the handles the rows are read through.
func (h *HostRows) Close() error {
	if h == nil {
		return nil
	}
	var errs []error
	for _, handle := range h.handles {
		errs = append(errs, handle.Close())
	}
	h.handles = nil
	return errors.Join(errs...)
}

// newHostRows reads info's rows through file, keeping at most cacheRows.
func newHostRows(file *gguf.File, info gguf.TensorInfo, cacheRows uint64) (*HostRows, error) {
	if file == nil {
		return nil, errors.New("host rows: GGUF file is nil")
	}
	layout, err := info.RowLayout()
	if err != nil {
		return nil, fmt.Errorf("host rows: %w", err)
	}
	if layout.BytesPerRow > uint64(math.MaxInt) || layout.ElementsPerRow > uint64(math.MaxInt) || layout.Count > hostRowEmpty {
		return nil, errors.New("host rows: rows exceed addressable memory")
	}
	rows := &HostRows{file: file, info: info, layout: layout, sets: cacheRows / hostRowWays}
	if rows.sets > uint64(math.MaxInt)/hostRowWays/layout.BytesPerRow {
		return nil, errors.New("host rows: cache exceeds addressable memory")
	}
	if rows.sets > 0 {
		rows.keys = make([]uint32, rows.sets*hostRowWays)
		for index := range rows.keys {
			rows.keys[index] = hostRowEmpty
		}
		rows.cached = make([]byte, rows.sets*hostRowWays*layout.BytesPerRow)
		rows.next = make([]uint8, rows.sets)
	}
	return rows, nil
}

// LoadHostRows dequantizes selected table entries, read once each.
func LoadHostRows(
	ctx context.Context,
	file *gguf.File,
	info gguf.TensorInfo,
	indices []uint32,
) (reference.Value, error) {
	rows, err := newHostRows(file, info, 0)
	if err != nil {
		return reference.Value{}, err
	}
	return rows.Load(ctx, indices)
}

// Load dequantizes the rows indices names, in order.
func (h *HostRows) Load(ctx context.Context, indices []uint32) (reference.Value, error) {
	if err := ctx.Err(); err != nil {
		return reference.Value{}, err
	}
	if len(indices) == tensor.FirstOffset {
		return reference.Value{}, errors.New("host rows: row list is empty")
	}
	layout := h.layout
	if uint64(len(indices)) > math.MaxUint64/layout.ElementsPerRow || uint64(len(indices))*layout.ElementsPerRow > uint64(math.MaxInt) ||
		uint64(len(indices)) > uint64(math.MaxInt)/layout.BytesPerRow {
		return reference.Value{}, errors.New("host rows: result exceeds addressable memory")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	// Each distinct row takes a slot of storage, from the cache or the file.
	slots := make(map[uint32]int, len(indices))
	var missing []uint32
	for _, index := range indices {
		if uint64(index) >= layout.Count {
			return reference.Value{}, fmt.Errorf("host rows: row %d exceeds tensor row count %d", index, layout.Count)
		}
		if _, seen := slots[index]; !seen {
			slots[index] = len(slots)
		}
	}
	rowBytes := int(layout.BytesPerRow)
	storage := make([]byte, len(slots)*rowBytes)
	for index, slot := range slots {
		if cached := h.find(index); cached != nil {
			copy(storage[slot*rowBytes:], cached)
		} else {
			missing = append(missing, index)
		}
	}
	if err := h.read(ctx, missing, slots, storage); err != nil {
		return reference.Value{}, err
	}
	for _, index := range missing {
		h.keep(index, storage[slots[index]*rowBytes:(slots[index]+1)*rowBytes])
	}
	width := int(layout.ElementsPerRow)
	output := make([]float32, width*len(indices))
	for position, index := range indices {
		slot := slots[index]
		if err := quant.DequantizeInto(h.info.Type, storage[slot*rowBytes:(slot+1)*rowBytes], output[position*width:(position+1)*width]); err != nil {
			return reference.Value{}, fmt.Errorf("dequantize host row %d from %q: %w", index, h.info.Name, err)
		}
	}
	shape, err := tensor.NewShape(layout.ElementsPerRow, uint64(len(indices)))
	if err != nil {
		return reference.Value{}, err
	}
	return reference.Value{Shape: shape, Data: output}, nil
}

// read reads rows into their slots of storage, a reader a handle, or
// hostRowReads readers through the file without handles; a cancelled
// context starts no further read.
func (h *HostRows) read(ctx context.Context, rows []uint32, slots map[uint32]int, storage []byte) error {
	rowBytes := h.layout.BytesPerRow
	var taken atomic.Int64
	errs := make([]error, min(hostRowReads, len(rows)))
	var group sync.WaitGroup
	for reader := range errs {
		group.Go(func() {
			for {
				next := int(taken.Add(1)) - 1
				if next >= len(rows) {
					return
				}
				if err := ctx.Err(); err != nil {
					errs[reader] = err
					return
				}
				index := rows[next]
				at := uint64(slots[index]) * rowBytes
				destination := storage[at : at+rowBytes]
				var err error
				if len(h.handles) > 0 {
					var read int
					read, err = h.handles[reader].ReadAt(destination, int64(h.base+uint64(index)*rowBytes))
					if read == len(destination) {
						err = nil
					} else if err == nil || errors.Is(err, io.EOF) {
						err = io.ErrUnexpectedEOF
					}
				} else {
					err = h.file.ReadTensorRange(h.info, uint64(index)*rowBytes, destination)
				}
				if err != nil {
					errs[reader] = fmt.Errorf("read host row %d from %q: %w", index, h.info.Name, err)
					return
				}
			}
		})
	}
	group.Wait()
	if err := errors.Join(errs...); err != nil {
		return err
	}
	return ctx.Err()
}

// set is the cache set row falls in.
func (h *HostRows) set(row uint32) uint64 {
	mixed := uint64(row) * hostRowMix
	return (mixed ^ mixed>>hostRowShift) % h.sets
}

// find returns row's cached bytes, nil where the cache lacks it.
func (h *HostRows) find(row uint32) []byte {
	if h.sets == 0 {
		return nil
	}
	set := h.set(row)
	for way := set * hostRowWays; way < (set+1)*hostRowWays; way++ {
		if h.keys[way] == row {
			return h.cached[way*h.layout.BytesPerRow : (way+1)*h.layout.BytesPerRow]
		}
	}
	return nil
}

// keep caches row's bytes in the next way of its set.
func (h *HostRows) keep(row uint32, bytes []byte) {
	if h.sets == 0 {
		return
	}
	set := h.set(row)
	way := set*hostRowWays + uint64(h.next[set])
	h.next[set] = uint8((uint64(h.next[set]) + 1) % hostRowWays)
	h.keys[way] = row
	copy(h.cached[way*h.layout.BytesPerRow:], bytes)
}
