package dataset

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"reflect"
	"testing"

	"overgo/internal/repodb"
)

func TestBenchmarkImportPreservesSourceAndRecordIdentity(t *testing.T) {
	const source = "benchmark/source"
	const revision = "source-revision"
	const split = "validation"
	const conversion = "source-schema-to-canonical/v1"
	data := []byte("{\"query\":\"first\",\"label\":\"one\"}\n{\"query\":\"second\",\"label\":\"two\"}\n")
	path := t.TempDir() + "/records.jsonl"
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	spec := BenchmarkImportSpec{
		Source: source, Revision: revision, SHA256: hex.EncodeToString(digest[:]), Split: split,
		Format: BenchmarkFormatJSONL, Conversion: conversion,
		Fields: []FieldBinding{{Target: "answer", Source: "label"}, {Target: "prompt", Source: "query"}},
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	imported, err := ImportBenchmark(context.Background(), store, path, spec)
	if err != nil {
		t.Fatal(err)
	}
	loaded, ok, err := benchmarkImportCodec.Read(context.Background(), store, imported.ID)
	if err != nil || !ok || !reflect.DeepEqual(loaded, imported) {
		t.Fatalf("loaded import = %+v, %v, %v", loaded, ok, err)
	}
	if imported.Count != uint64(len(imported.Records)) || imported.Spec.Source != source ||
		imported.Spec.Revision != revision || imported.Spec.Split != split || imported.Spec.Conversion != conversion {
		t.Fatalf("import provenance = %+v", imported)
	}
	record, ok, err := benchmarkRecordCodec.Read(context.Background(), store, imported.Records[0])
	if err != nil || !ok || record.Profile != imported.Profile || record.Ordinal != 0 ||
		!reflect.DeepEqual(record.Fields, []BenchmarkField{
			{Name: "answer", Value: []byte(`"one"`)}, {Name: "prompt", Value: []byte(`"first"`)},
		}) {
		t.Fatalf("loaded record = %+v, %v, %v", record, ok, err)
	}
	changed := spec
	changed.Revision += "/changed"
	changedImport, err := ImportBenchmark(context.Background(), store, path, changed)
	if err != nil {
		t.Fatal(err)
	}
	if changedImport.Profile == imported.Profile || changedImport.ID == imported.ID ||
		changedImport.Records[0] == imported.Records[0] {
		t.Fatal("source revision did not change import identities")
	}
	changed.SHA256 = hex.EncodeToString(make([]byte, sha256.Size))
	if _, err := ImportBenchmark(context.Background(), store, path, changed); err == nil {
		t.Fatal("source hash mismatch accepted")
	}
}
