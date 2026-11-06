package dataset

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"overgo/internal/testcheck"
)

func nullableParquetFixture(t *testing.T) (string, uint64) {
	t.Helper()
	data := testcheck.Must(os.ReadFile("testdata/nullable_parquet.json"))(t)
	var fixture struct {
		Parquet string `json:"parquet_base64"`
	}
	testcheck.NoError(t, json.Unmarshal(data, &fixture))
	encoded := testcheck.Must(base64.StdEncoding.DecodeString(fixture.Parquet))(t)
	path := filepath.Join(t.TempDir(), "nullable.parquet")
	testcheck.NoError(t, os.WriteFile(path, encoded, 0600))
	// For this tiny oracle the complete encoded file exceeds the cumulative
	// requested decode storage of either group; no production default is implied.
	return path, uint64(len(encoded))
}

func TestParquetRowsPhysicalJoin(t *testing.T) {
	path, budget := nullableParquetFixture(t)
	r := testcheck.Must(OpenParquetRows(t.Context(), path, []string{"audio.bytes", "text", "id"}, budget))(t)
	defer r.Close()
	if r.Rows() != 6 {
		t.Fatalf("physical rows = %d", r.Rows())
	}
	wantAudio := []*string{new("a"), nil, nil, new(""), new("e"), new("f")}
	wantText := []*string{new("A"), new("B"), nil, new("D"), new("E"), new("F")}
	// Revisit a previous group and resume inside the next group: identities
	// and nulls must not depend on traversal order or cache state.
	for _, ordinal := range []uint64{0, 1, 2, 3, 4, 5, 1, 4} {
		row := testcheck.Must(r.Read(t.Context(), ordinal))(t)
		if !reflect.DeepEqual(row["audio.bytes"], wantAudio[ordinal]) || !reflect.DeepEqual(row["text"], wantText[ordinal]) || row["id"] == nil || *row["id"] != string(rune('0'+ordinal)) {
			t.Fatalf("physical row %d = %v", ordinal, row)
		}
	}
	if _, err := r.Read(t.Context(), r.Rows()); !errors.Is(err, io.EOF) {
		t.Fatalf("end: %v", err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, err := r.Read(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	testcheck.NoError(t, r.Close())
	if _, err := r.Read(t.Context(), 0); !errors.Is(err, os.ErrClosed) {
		t.Fatalf("closed: %v", err)
	}
}

func TestParquetWorkspaceAndMalformedPages(t *testing.T) {
	path, budget := nullableParquetFixture(t)
	r := testcheck.Must(OpenParquetRows(t.Context(), path, []string{"audio.bytes", "text"}, budget))(t)
	defer r.Close()
	// Exhaust the decode budget independently of the metadata budget.
	r.maximumBytes = 1
	if _, err := r.Read(t.Context(), 0); err == nil {
		t.Fatal("unbounded decode admitted")
	}
	if r.group != -1 || r.values != nil {
		t.Fatal("partial group retained after failure")
	}
	r.maximumBytes = int64(budget)
	if _, err := r.Read(t.Context(), 0); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenParquetRows(t.Context(), path, []string{"text"}, 1); err == nil {
		t.Fatal("unbounded footer admitted")
	}
	for _, tc := range []struct {
		name    string
		header  parquetPageHeader
		payload []byte
	}{
		{"negative-values", parquetPageHeader{numValues: -1}, nil},
		{"negative-levels", parquetPageHeader{v2: true, defLevelBytes: -1}, nil},
		{"repeated-levels", parquetPageHeader{v2: true, repLevelBytes: 1, uncompressedSize: 1}, []byte{0}},
		{"missing-levels", parquetPageHeader{numValues: 1}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := &parquetBudget{remaining: int64(budget)}
			if _, err := decodeDataPage(tc.payload, parquetCodecUncompressed, tc.header, 1, nil, decodePlainStrings, b); err == nil {
				t.Fatal("malformed page decoded")
			}
		})
	}
	declared := binary.AppendUvarint(nil, budget+1)
	if _, err := snappyDecode(declared, &parquetBudget{remaining: int64(budget)}); err == nil {
		t.Fatal("oversized Snappy allocation admitted")
	}
	if _, err := snappyDecode([]byte{1, 4, 'a', 'b'}, nil); err == nil {
		t.Fatal("Snappy grew beyond declared capacity")
	}
}
