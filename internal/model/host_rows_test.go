package model

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/gguf"
)

// TestHostRowsReadEachRowOnceAndKeepIt reads a 300-row table's rows
// together, each once, and holds them to the fixture's values; a cached
// reader serves rows it read without the file and keeps at most its bound,
// and a read cancelled in flight starts no read past those in flight.
func TestHostRowsReadEachRowOnceAndKeepIt(t *testing.T) {
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

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	reader.calls.Store(0)
	reader.after = cancel
	value, err = LoadHostRows(ctx, file, file.Tensors[0], every)
	if !errors.Is(err, context.Canceled) || reader.calls.Load() > hostRowReads || len(value.Data) != 0 {
		t.Fatalf("interrupted rows = %d values, reads %d, error %v", len(value.Data), reader.calls.Load(), err)
	}

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
}
