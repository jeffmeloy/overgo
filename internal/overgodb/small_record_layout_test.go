package overgodb

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// TestSmallRecordLayoutDecision measures the retained segmented layout against a
// representative distribution of many small immutable records and a few large
// payloads, then records the decision: the segmented journal already packs small
// records into a handful of segment files, far below one file per record, so an
// append-only pack layer would add no win over the retained layout plus snapshot
// reuse. Random lookup after a reopen stays exact. The reopen trigger is the
// physical file count approaching the record count, which would mean per-record
// files and would justify revisiting packing.
func TestSmallRecordLayoutDecision(t *testing.T) {
	root := t.TempDir()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	contract := artifact.DocumentContract{
		Kind: artifact.KindEvidence,
		// A representative small immutable record: the evidence descriptor census
		// averages ~158 bytes per text record, so the padded body lands nearby.
		MediaType: "application/vnd.overgo.small-record-fixture+json",
		Schema:    "overgo/small-record-fixture/v1",
	}
	record := func(index, pad int) (artifact.Descriptor, artifact.Content, artifact.AliasBinding) {
		t.Helper()
		data, err := json.Marshal(struct {
			Index int    `json:"index"`
			Pad   string `json:"pad"`
		}{index, strings.Repeat("x", pad)})
		if err != nil {
			t.Fatal(err)
		}
		id, err := contract.Identify(data)
		if err != nil {
			t.Fatal(err)
		}
		content, err := contract.Content(id, data)
		if err != nil {
			t.Fatal(err)
		}
		return content.Descriptor, content, artifact.AliasBinding{Name: fmt.Sprintf("small-record/%d", index), Target: id}
	}

	const (
		smallRecords = 256
		perBatch     = 32
		smallPad     = 120
		largeRecords = 3
		largePad     = 256 * 1024
	)
	var firstSmall, lastSmall artifact.ID
	for start := 0; start < smallRecords; start += perBatch {
		batch := artifact.Batch{Key: fmt.Sprintf("small-record/batch/%d", start)}
		for index := start; index < start+perBatch; index++ {
			descriptor, content, alias := record(index, smallPad)
			batch.Artifacts = append(batch.Artifacts, descriptor)
			batch.Contents = append(batch.Contents, content)
			batch.Aliases = append(batch.Aliases, alias)
			if index == 0 {
				firstSmall = alias.Target
			}
			lastSmall = alias.Target
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
	}
	largeBatch := artifact.Batch{Key: "small-record/large"}
	for index := range largeRecords {
		descriptor, content, alias := record(smallRecords+index, largePad)
		largeBatch.Artifacts = append(largeBatch.Artifacts, descriptor)
		largeBatch.Contents = append(largeBatch.Contents, content)
		largeBatch.Aliases = append(largeBatch.Aliases, alias)
	}
	if _, err := store.Commit(t.Context(), largeBatch); err != nil {
		t.Fatal(err)
	}

	destination := filepath.Join(t.TempDir(), "backup")
	report, err := store.Backup(t.Context(), destination)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retained layout backup: files=%d bytes=%d (small=%d large=%d)", report.FilesCopied, report.BytesCopied, smallRecords, largeRecords)
	// The decision is byte-based, not file-count-based: the store keys each
	// immutable content by hash, so packing small records would cut the file
	// count but not the bytes, while adding torn-write, crash-recovery, index
	// rebuild, and migration obligations. The retained layout is justified when
	// its per-record physical overhead over the logical payload stays small, so
	// packing buys no byte saving. Reopen trigger: per-record overhead beyond the
	// bound would make packing worth revisiting.
	logical := int64(smallRecords*smallPad + largeRecords*largePad)
	overhead := report.BytesCopied - logical
	if overhead < 0 {
		overhead = 0
	}
	const perRecordOverheadBound = 2048
	if perRecord := overhead / int64(smallRecords+largeRecords); perRecord > perRecordOverheadBound {
		t.Fatalf("retained layout adds %d bytes/record overhead (copied %d for %d logical); packing may be justified -- reopen the decision", perRecord, report.BytesCopied, logical)
	}

	// Random lookup after a reopen stays exact for both a small and a large record.
	replica, err := OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer replica.Close()
	for _, id := range []artifact.ID{firstSmall, lastSmall} {
		if _, found, err := artifact.ReadContent(t.Context(), replica, id); err != nil || !found {
			t.Fatalf("reopened small record %s: found=%v err=%v", id, found, err)
		}
	}
}
