package dataset

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestArrowStreamDecodesHuggingFaceCache pins the reader against real
// HuggingFace dataset-cache bytes (MMLU abstract_algebra dev and
// validation splits): row counts match the cache's dataset_info, the
// first dev row decodes to its exact question, choices list, and
// ClassLabel answer, and every row carries the full field set.
func TestArrowStreamDecodesHuggingFaceCache(t *testing.T) {
	decode := func(path string) []map[string]json.RawMessage {
		t.Helper()
		file, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		var rows []map[string]json.RawMessage
		err = DecodeArrowStream(file, func(ordinal uint64, record map[string]json.RawMessage) error {
			if ordinal != uint64(len(rows)) {
				t.Fatalf("ordinal %d out of order at row %d", ordinal, len(rows))
			}
			rows = append(rows, record)
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return rows
	}

	dev := decode(filepath.Join("testdata", "mmlu-dev.arrow"))
	if len(dev) != 5 {
		t.Fatalf("dev rows = %d, want the cache's 5", len(dev))
	}
	first := dev[0]
	if string(first["question"]) != `"Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field."` ||
		string(first["subject"]) != `"abstract_algebra"` ||
		string(first["choices"]) != `["0","1","2","3"]` ||
		string(first["answer"]) != "1" {
		t.Fatalf("first dev row = %v", first)
	}
	for index, row := range dev {
		for _, field := range []string{"question", "subject", "choices", "answer"} {
			if _, present := row[field]; !present {
				t.Fatalf("dev row %d misses %q", index, field)
			}
		}
	}

	validation := decode(filepath.Join("testdata", "mmlu-validation.arrow"))
	if len(validation) != 11 {
		t.Fatalf("validation rows = %d, want the cache's 11", len(validation))
	}
}

// TestArrowStreamRefusesOutsideSubset pins the refusal contract: a
// stream whose framing is broken names the defect instead of guessing.
func TestArrowStreamRefusesOutsideSubset(t *testing.T) {
	var broken bytes.Buffer
	_ = binary.Write(&broken, binary.LittleEndian, uint32(0x12345678))
	err := DecodeArrowStream(&broken, func(uint64, map[string]json.RawMessage) error { return nil })
	if err == nil {
		t.Fatal("broken framing decoded")
	}
}

// TestArrowBenchmarkImport pins the format end to end: the arrow file
// imports through ImportBenchmark like any JSONL benchmark, its cases
// land as records with the bound fields, and the import document
// counts every row.
func TestArrowBenchmarkImport(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	path := filepath.Join("testdata", "mmlu-dev.arrow")
	digest, err := HashFile(path)
	if err != nil {
		t.Fatal(err)
	}
	imported, err := ImportBenchmark(t.Context(), store, path, BenchmarkImportSpec{
		Source: "cais/mmlu:abstract_algebra", Revision: "0.0.0", SHA256: digest,
		Split: "dev", Format: BenchmarkFormatArrow, Conversion: "lm-eval/mmlu/v1",
		Fields: []FieldBinding{
			{Target: "question", Source: "question"},
			{Target: "choices", Source: "choices"},
			{Target: "answer", Source: "answer"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if imported.Count != 5 || len(imported.Records) != 5 {
		t.Fatalf("imported = %+v, want 5 records", imported)
	}
	record, found, err := benchmarkRecordCodec.Read(t.Context(), store, imported.Records[0])
	if err != nil || !found {
		t.Fatalf("record read = (%t, %v)", found, err)
	}
	fields := map[string]string{}
	for _, field := range record.Fields {
		fields[field.Name] = string(field.Value)
	}
	if fields["choices"] != `["0","1","2","3"]` || fields["answer"] != "1" {
		t.Fatalf("record fields = %v", fields)
	}
	if _, err := artifact.JSONID(artifact.KindProfile, imported.Spec); err != nil {
		t.Fatal(err)
	}
}
