package main

import (
	"slices"
	"testing"
)

func TestExtractTable(t *testing.T) {
	source := `
GGML_TABLE_BEGIN(uint64_t, sample_grid, 2)
    0x0102030405060708, 9
GGML_TABLE_END()
`
	got, err := extractTable(source, tableSpec{
		cType:  "uint64_t",
		cName:  "sample_grid",
		goType: "uint64",
		goName: "sampleGrid",
		size:   2,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"0x0102030405060708", "0x0000000000000009"}
	if !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
}

func TestExtractTableRejectsUnexpectedSize(t *testing.T) {
	source := `
GGML_TABLE_BEGIN(uint32_t, sample_grid, 1)
    0x01
GGML_TABLE_END()
`
	_, err := extractTable(source, tableSpec{
		cType:  "uint32_t",
		cName:  "sample_grid",
		goType: "uint32",
		goName: "sampleGrid",
		size:   2,
	})
	if err == nil {
		t.Fatal("unexpected table size was accepted")
	}
}
