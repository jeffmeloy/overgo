package overgodb

import (
	"bytes"
	"runtime"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

// TestMemoryMovementTraceReduction pins the delta-serving copy trace: a
// head-bound delta drops payload bytes by contract, so serving a window over
// a large committed content allocates less than half that payload — the
// former path materialized and re-validated every payload it then
// discarded, and this measured bound refuses its return. The full
// commit-delta read still materializes and validates payloads for the
// consumers that need the bytes.
func TestMemoryMovementTraceReduction(t *testing.T) {
	store, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := t.Context()
	contract := ProjectionContractVersion()
	seed := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "movement-seed"), Size: 1}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "movement/seed", Artifacts: []artifact.Descriptor{seed},
	}); err != nil {
		t.Fatal(err)
	}
	anchorHead, anchorSequence := store.Head()

	payload := bytes.Repeat([]byte{0x6d}, 1<<20)
	content, err := artifact.DocumentContract{
		Kind: artifact.KindEvidence, MediaType: "application/octet-stream", Schema: "overgo/movement-blob/v1",
	}.ContentBytes(payload)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "movement/blob", Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}

	var allocated uint64 = ^uint64(0)
	for range 3 {
		var before, after runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&before)
		delta, resync, err := store.DeltasSince(ctx, anchorHead, anchorSequence, contract, 8)
		runtime.ReadMemStats(&after)
		if err != nil || resync || len(delta.Contents) != 1 || delta.Contents[0] != content.Descriptor.ID {
			t.Fatalf("delta window = (%+v, resync=%t, %v)", delta, resync, err)
		}
		if moved := after.TotalAlloc - before.TotalAlloc; moved < allocated {
			allocated = moved
		}
	}
	if bound := uint64(len(payload) / 2); allocated >= bound {
		t.Fatalf("delta serving moved %d bytes for a dropped %d-byte payload (bound %d)", allocated, len(payload), bound)
	}

	full, found, err := store.CommitDeltaAt(ctx, anchorSequence+1)
	if err != nil || !found || len(full.Delta.Contents) != 1 ||
		!bytes.Equal(full.Delta.Contents[0].Data, payload) {
		t.Fatalf("full commit delta no longer materializes payloads: (found=%t, %v)", found, err)
	}
}
