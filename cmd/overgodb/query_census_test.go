package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestRecordCensus builds a store with a templated schema -- records of one
// size that differ in one byte -- and a varied one, releases one templated
// record, and holds the census to what the catalog says: schemas ordered by
// record count; every record counted and sized, the released one included,
// because its descriptor still taxes a walk; the records whose bytes the
// release left on disk; and the modal size with its population, the whole
// schema for the template and a single record where no two share a size.
func TestRecordCensus(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	root := filepath.Join(t.TempDir(), "store")
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	mint := func(schema, body string) artifact.Content {
		t.Helper()
		contract := artifact.DocumentContract{Kind: artifact.KindOutput, MediaType: "application/json", Schema: schema}
		content, err := contract.ContentBytes([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		return content
	}
	var templated []artifact.Content
	for index := range 4 {
		templated = append(templated, mint("test/templated/v1", fmt.Sprintf(`{"template":"the same forty bytes of boilerplate","n":%d}`, index)))
	}
	varied := []artifact.Content{mint("test/varied/v1", `{"a":1}`), mint("test/varied/v1", `{"a":1,"b":"longer"}`)}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "census/fixture", Contents: append(templated, varied...)}); err != nil {
		t.Fatal(err)
	}
	released := templated[0].Descriptor.ID
	if _, err := overgodb.Release(ctx, store, func(string, artifact.ID) error { return nil }, overgodb.RetentionPolicy{},
		func(descriptor artifact.Descriptor) bool { return descriptor.ID == released }); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	var output bytes.Buffer
	if err := writeCensus(&output, root, len(varied)); err != nil {
		t.Fatal(err)
	}
	var census []schemaCensus
	if err := json.Unmarshal(output.Bytes(), &census); err != nil {
		t.Fatalf("census output %s: %v", output.Bytes(), err)
	}
	if len(census) != len(varied) || census[0].Schema != "test/templated/v1" || census[1].Schema != "test/varied/v1" {
		t.Fatalf("census order = %+v", census)
	}
	template, size := census[0], templated[0].Descriptor.Size
	if template.Records != len(templated) || template.Bytes != size*uint64(len(templated)) || template.Small != len(templated) ||
		template.ModalSize != size || template.ModalRecords != len(templated) {
		t.Fatalf("templated census = %+v, want %d records of %d bytes", template, len(templated), size)
	}
	if held := len(templated) - 1; template.Held != held || template.HeldBytes != size*uint64(held) {
		t.Fatalf("templated census holds %d records / %d bytes, want the %d the release left", template.Held, template.HeldBytes, held)
	}
	if other := census[1]; other.Records != len(varied) || other.ModalRecords != 1 || other.Held != len(varied) {
		t.Fatalf("varied census = %+v", other)
	}
}
