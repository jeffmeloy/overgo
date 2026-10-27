package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"overgo/internal/gguf"
	"overgo/internal/tensor"
	"overgo/internal/tensor/dtype"
)

func TestLoadHostTensor(t *testing.T) {
	data := hostTensorFixture(t, "weight", 4)
	file, err := gguf.Parse(bytes.NewReader(data), uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	value, err := LoadHostTensor(t.Context(), file, file.Tensors[0])
	if err != nil {
		t.Fatal(err)
	}
	if !value.Shape.Equal(tensor.MustShape(4)) {
		t.Fatalf("shape = %v", value.Shape.Slice())
	}
	for index, want := range []float32{1, 2, 3, 4} {
		if value.Data[index] != want {
			t.Fatalf("value[%d] = %v, want %v", index, value.Data[index], want)
		}
	}
}

// TestLoadHostRowsAndDotRows reads a 300-row table's rows, reordered and
// repeated, each distinct row once and through the file together; a cached
// reader serves rows it read without the file and keeps at most its eight
// ways a set; a table in a file is read through handles of its own at its
// offset; DotRows scores every row; invalid rows are refused; and a read
// cancelled before or during its reads returns no values, starting no read
// past those in flight.
func TestLoadHostRowsAndDotRows(t *testing.T) {
	const tableRows = 300
	data := hostTensorFixture(t, "table", 2, tableRows)
	reader := &countedRowReader{Reader: bytes.NewReader(data)}
	file, err := gguf.Parse(reader, uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	// Row r holds 2r+1 and 2r+2.
	check := func(indices []uint32, data []float32) {
		t.Helper()
		for position, row := range indices {
			if data[2*position] != float32(2*row+1) || data[2*position+1] != float32(2*row+2) {
				t.Fatalf("row %d at %d = %v", row, position, data[2*position:2*position+2])
			}
		}
	}
	every := make([]uint32, 0, 2*tableRows)
	for row := range uint32(tableRows) {
		every = append(every, tableRows-1-row, row)
	}
	reader.calls.Store(0)
	value, err := LoadHostRows(t.Context(), file, file.Tensors[0], every)
	if err != nil {
		t.Fatal(err)
	}
	if reader.calls.Load() != tableRows {
		t.Fatalf("read %d rows, want %d", reader.calls.Load(), tableRows)
	}
	check(every, value.Data)

	// One set of eight ways: of nine rows read, eight stay.
	rows, err := NewHostRows(file, file.Tensors[0], hostRowWays)
	if err != nil {
		t.Fatal(err)
	}
	nine := []uint32{0, 1, 2, 3, 4, 5, 6, 7, 8}
	for _, want := range []int64{9, 1} {
		reader.calls.Store(0)
		if value, err = rows.Load(t.Context(), nine); err != nil {
			t.Fatal(err)
		}
		if reader.calls.Load() != want {
			t.Fatalf("read %d of nine rows, want %d", reader.calls.Load(), want)
		}
		check(nine, value.Data)
	}

	// A prefetch reads the rows the cache lacks, and the Load after it reads none of them again.
	if rows, err = NewHostRows(file, file.Tensors[0], hostRowWays); err != nil {
		t.Fatal(err)
	}
	four := []uint32{9, 10, 11, 9}
	reader.calls.Store(0)
	rows.Prefetch(t.Context(), four)
	if value, err = rows.Load(t.Context(), four); err != nil {
		t.Fatal(err)
	}
	if reader.calls.Load() != 3 {
		t.Fatalf("a prefetch and a load read %d rows, want the 3 distinct once", reader.calls.Load())
	}
	check(four, value.Data)

	// A table in a file is read through handles of its own, at the table's offset in the file.
	path := filepath.Join(t.TempDir(), "table.gguf")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	opened, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer opened.Close()
	if rows, err = NewHostRows(opened, opened.Tensors[0], 0); err != nil {
		t.Fatal(err)
	}
	if len(rows.handles) != hostRowReads {
		t.Fatalf("the reader holds %d handles, want %d", len(rows.handles), hostRowReads)
	}
	if value, err = rows.Load(t.Context(), every); err != nil {
		t.Fatal(err)
	}
	check(every, value.Data)
	if err := rows.Close(); err != nil || len(rows.handles) != 0 {
		t.Fatalf("closing the reader left %d handles: %v", len(rows.handles), err)
	}

	scores, err := DotRows(t.Context(), file, file.Tensors[0], []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	for row, score := range scores {
		if score != float32(4*row+3) {
			t.Fatalf("score[%d] = %v, want %v", row, score, 4*row+3)
		}
	}
	for _, indices := range [][]uint32{{tableRows}, {2, tableRows}, nil} {
		if _, err := LoadHostRows(t.Context(), file, file.Tensors[0], indices); err == nil {
			t.Fatalf("invalid rows accepted: %v", indices)
		}
	}
	// A cancellation during the reads starts none past those in flight: both distinct rows of three, or the readers of every row.
	for _, cancelled := range []struct {
		indices []uint32
		reads   int64
	}{{[]uint32{2, 2, 0}, 2}, {every, hostRowReads}} {
		ctx, cancel := context.WithCancel(t.Context())
		reader.calls.Store(0)
		reader.after = cancel
		value, err = LoadHostRows(ctx, file, file.Tensors[0], cancelled.indices)
		reader.after = nil
		if !errors.Is(err, context.Canceled) || reader.calls.Load() > cancelled.reads || len(value.Data) != 0 {
			t.Fatalf("interrupted rows = %d values, reads %d, error %v", len(value.Data), reader.calls.Load(), err)
		}
		reader.calls.Store(0)
		if _, err := LoadHostRows(ctx, file, file.Tensors[0], []uint32{0}); !errors.Is(err, context.Canceled) || reader.calls.Load() != 0 {
			t.Fatalf("cancelled read touched file %d times: %v", reader.calls.Load(), err)
		}
	}
}

// TestHostRowsPrefetchesReadTogether reads a prefetch's rows beside a batch still reading: a Load of the later
// prefetch's row returns while the earlier one's read is held, and a Load of the held row waits for that read rather
// than reading it again.
func TestHostRowsPrefetchesReadTogether(t *testing.T) {
	data := hostTensorFixture(t, "table", 2, 300)
	reader := &heldRowReader{Reader: bytes.NewReader(data), started: make(chan struct{}), release: make(chan struct{})}
	file, err := gguf.Parse(reader, uint64(len(data)), gguf.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	rows, err := NewHostRows(file, file.Tensors[0], hostRowWays)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	reader.hold.Store(true)
	rows.Prefetch(t.Context(), []uint32{1})
	<-reader.started
	reader.hold.Store(false)
	rows.Prefetch(t.Context(), []uint32{2})
	value, err := rows.Load(t.Context(), []uint32{2})
	if err != nil {
		t.Fatal(err)
	}
	if value.Data[0] != 5 || reader.finished.Load() != 1 {
		t.Fatalf("row 2 = %v with %d reads finished, want 5 beside the held read", value.Data, reader.finished.Load())
	}
	close(reader.release)
	if value, err = rows.Load(t.Context(), []uint32{1}); err != nil {
		t.Fatal(err)
	}
	if value.Data[0] != 3 || reader.calls.Load() != 2 {
		t.Fatalf("row 1 = %v after %d reads, want 3 read once by its prefetch", value.Data, reader.calls.Load())
	}
}

// heldRowReader holds the first read made while hold is set until release closes, saying so on started.
type heldRowReader struct {
	*bytes.Reader
	hold, held       atomic.Bool
	calls, finished  atomic.Int64
	started, release chan struct{}
}

func (r *heldRowReader) ReadAt(p []byte, off int64) (int, error) {
	if r.hold.Load() && r.held.CompareAndSwap(false, true) {
		r.calls.Add(1)
		close(r.started)
		<-r.release
		return r.Reader.ReadAt(p, off)
	}
	if r.held.Load() {
		r.calls.Add(1)
		defer r.finished.Add(1)
	}
	return r.Reader.ReadAt(p, off)
}

type countedRowReader struct {
	*bytes.Reader
	calls atomic.Int64
	after func()
}

func (r *countedRowReader) ReadAt(p []byte, off int64) (int, error) {
	r.calls.Add(1)
	n, err := r.Reader.ReadAt(p, off)
	if r.after != nil {
		r.after()
	}
	return n, err
}

func hostTensorFixture(t *testing.T, name string, dimensions ...uint64) []byte {
	t.Helper()
	var buffer bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(value string) {
		write(uint64(len(value)))
		_, _ = buffer.WriteString(value)
	}
	_, _ = buffer.WriteString(gguf.Magic)
	write(uint32(gguf.CurrentVersion))
	write(uint64(1))
	write(uint64(0))
	writeString(name)
	write(uint32(len(dimensions)))
	elements := uint64(1)
	for _, dimension := range dimensions {
		write(dimension)
		elements *= dimension
	}
	write(uint32(dtype.F32))
	write(uint64(0))
	for buffer.Len()%gguf.DefaultAlignment != 0 {
		_ = buffer.WriteByte(0)
	}
	for index := range elements {
		write(float32(index + 1))
	}
	for buffer.Len()%gguf.DefaultAlignment != 0 {
		_ = buffer.WriteByte(0)
	}
	return buffer.Bytes()
}
