package dataset

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sync"
)

// ParquetRows reads selected byte-array columns by physical row number. It
// retains at most one decoded row group. Concurrent callers share that group;
// returned strings are immutable and remain valid after the next read.
// Read and Close are serialized; closing does not invalidate returned strings.
type ParquetRows struct {
	mu                  sync.Mutex
	file                *os.File
	metadata            parquetMetadata
	columns             []string
	leaves, definitions []int
	ends                []uint64
	maximumBytes        int64
	group               int
	values              [][]parquetValue
}

// OpenParquetRows opens a bounded, row-addressed source. maximumBytes bounds
// footer bytes and, separately, cumulative requested decode storage per group.
// The latter includes chunks, decompression, dictionaries, levels and retained
// values, but is not a process-memory measurement. Caller-retained output and
// parsed metadata are outside the decode workspace budget.
func OpenParquetRows(ctx context.Context, path string, columns []string, maximumBytes uint64) (*ParquetRows, error) {
	if ctx == nil || len(columns) == 0 || maximumBytes == 0 || maximumBytes > uint64(^uint(0)>>1) {
		return nil, errors.New("dataset: incomplete bounded parquet source")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fail := func(err error) (*ParquetRows, error) { _ = file.Close(); return nil, err }
	info, err := file.Stat()
	if err != nil {
		return fail(err)
	}
	var head [len(parquetMagic)]byte
	var tail [len(parquetMagic) + 4]byte
	if !info.Mode().IsRegular() || info.Size() < int64(len(head)+len(tail)) {
		return fail(errors.New("dataset: invalid parquet file size"))
	}
	if _, err := file.ReadAt(head[:], 0); err != nil {
		return fail(err)
	}
	if _, err := file.ReadAt(tail[:], info.Size()-int64(len(tail))); err != nil {
		return fail(err)
	}
	footerBytes := uint64(binary.LittleEndian.Uint32(tail[:]))
	if string(head[:]) != parquetMagic || string(tail[4:]) != parquetMagic || footerBytes > maximumBytes || footerBytes > uint64(info.Size()-int64(len(head)+len(tail))) {
		return fail(errors.New("dataset: parquet framing or footer budget differs"))
	}
	metadata, err := readParquetFooter(file, info.Size())
	if err != nil {
		return fail(err)
	}
	r := &ParquetRows{file: file, metadata: metadata, columns: slices.Clone(columns), maximumBytes: int64(maximumBytes), group: -1}
	for index, column := range columns {
		if column == "" || slices.Contains(columns[:index], column) {
			return fail(errors.New("dataset: empty or duplicate parquet column"))
		}
		leaf, definition, err := stringColumnLeaf(metadata, column)
		if err != nil {
			return fail(err)
		}
		r.leaves = append(r.leaves, leaf)
		r.definitions = append(r.definitions, definition)
	}
	var total uint64
	for _, group := range metadata.rowGroups {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		if group.rows <= 0 || uint64(group.rows) > ^uint64(0)-total {
			return fail(errors.New("dataset: invalid parquet row count"))
		}
		for _, leaf := range r.leaves {
			if leaf >= len(group.columns) || group.columns[leaf].numValues != group.rows {
				return fail(errors.New("dataset: parquet column and group row counts differ"))
			}
		}
		total += uint64(group.rows)
		r.ends = append(r.ends, total)
	}
	return r, nil
}

// Rows returns the physical row count, including rows with null columns.
func (r *ParquetRows) Rows() uint64 { return r.ends[len(r.ends)-1] }

// Read returns a row's selected columns. A nil value denotes null, distinct
// from a present empty byte array. Column names are the requested schema names.
// Reading the row count itself returns io.EOF; no cursor is advanced on errors.
func (r *ParquetRows) Read(ctx context.Context, row uint64) (map[string]*string, error) {
	if ctx == nil {
		return nil, errors.New("dataset: nil parquet context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r.file == nil {
		return nil, os.ErrClosed
	}
	if row >= r.Rows() {
		return nil, io.EOF
	}
	group, _ := slices.BinarySearch(r.ends, row+1)
	if group != r.group {
		// Drop the old group before starting the next allocation budget.
		r.values, r.group = nil, -1
		budget := &parquetBudget{remaining: r.maximumBytes}
		if err := budget.take(int64(len(r.columns)), 3*parquetWordBytes); err != nil {
			return nil, err
		}
		values := make([][]parquetValue, len(r.columns))
		count := r.metadata.rowGroups[group].rows
		for index, leaf := range r.leaves {
			if err := budget.take(count, parquetValueBytes); err != nil {
				return nil, err
			}
			values[index] = make([]parquetValue, int(count))
			err := readColumnChunk(ctx, r.file, r.metadata.rowGroups[group].columns[leaf], r.definitions[index], budget, func(ordinal uint64, value parquetValue) error {
				values[index][ordinal] = value
				return nil
			})
			if err != nil {
				return nil, fmt.Errorf("dataset: parquet column %s: %w", r.columns[index], err)
			}
		}
		r.values, r.group = values, group
	}
	start := uint64(0)
	if group != 0 {
		start = r.ends[group-1]
	}
	result := make(map[string]*string, len(r.columns))
	for index, column := range r.columns {
		value := r.values[index][row-start]
		if value.valid {
			result[column] = &value.text
		} else {
			result[column] = nil
		}
	}
	return result, nil
}

// Close releases the file and cached group. Repeated calls are harmless.
func (r *ParquetRows) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.file == nil {
		return nil
	}
	err := r.file.Close()
	r.file, r.values, r.group = nil, nil, -1
	return err
}
