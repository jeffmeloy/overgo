package overgodb_test

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// Baseline corpus extents: small enough for the fast lane, exact
// enough that every work counter is derived from construction.
const (
	baselineCommits    = 32
	baselineReadSample = 8
)

// TestInteractionEfficiencyBaseline records the representative storage
// interaction trace: a store session whose work counters are counted
// at the call sites and must equal the counts derived from the
// corpus's construction. The trace binds the session's head artifact
// and the journal's content identity as result and evidence, so a
// later interaction reduction is compared against completed work, not
// a smaller task.
func TestInteractionEfficiencyBaseline(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	var work runrecord.InteractionWork
	var lastID artifact.ID
	for ordinal := range baselineCommits {
		payload := []byte(fmt.Sprintf("interaction-baseline/%d", ordinal))
		id, err := artifact.IdentifyBytes(artifact.KindRun, payload)
		if err != nil {
			t.Fatal(err)
		}
		descriptor := artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "text/plain"}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:       fmt.Sprintf("baseline/%d", ordinal),
			Artifacts: []artifact.Descriptor{descriptor},
			Contents:  []artifact.Content{{Descriptor: descriptor, Data: payload}},
		}); err != nil {
			t.Fatal(err)
		}
		work.Commits++
		work.SemanticTransitions++
		lastID = id
	}
	for ordinal := 0; ordinal < baselineReadSample; ordinal++ {
		payload := []byte(fmt.Sprintf("interaction-baseline/%d", ordinal))
		id, err := artifact.IdentifyBytes(artifact.KindRun, payload)
		if err != nil {
			t.Fatal(err)
		}
		if _, found, err := store.Artifact(ctx, id); err != nil || !found {
			t.Fatalf("artifact %d: found=%v err=%v", ordinal, found, err)
		}
		work.ArtifactReads++
		work.ReturnedFacts++
	}
	_, reader, found, err := store.OpenContent(ctx, lastID)
	if err != nil || !found {
		t.Fatalf("content: found=%v err=%v", found, err)
	}
	if _, err := reader.Read(make([]byte, 1)); err != nil {
		t.Fatal(err)
	}
	work.BlobReads++

	journal, err := os.ReadFile(filepath.Join(root, "overgodb.log"))
	if err != nil {
		t.Fatal(err)
	}
	work.Bytes = uint64(len(journal))

	if work.Commits != baselineCommits || work.SemanticTransitions != baselineCommits ||
		work.ArtifactReads != baselineReadSample || work.ReturnedFacts != baselineReadSample ||
		work.BlobReads != 1 || work.Bytes == 0 {
		t.Fatalf("counted work does not match construction: %+v", work)
	}

	digest := sha256.Sum256(journal)
	evidence, err := artifact.IdentifyBytes(artifact.KindEvidence, digest[:])
	if err != nil {
		t.Fatal(err)
	}
	trace, err := runrecord.NewEfficiencyTrace(runrecord.EfficiencyTrace{
		Surface:  runrecord.SurfaceStorage,
		Task:     "storage baseline session",
		Work:     work,
		Result:   lastID,
		Evidence: evidence,
	})
	if err != nil {
		t.Fatal(err)
	}
	content, err := trace.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := runrecord.ParseEfficiencyTrace(content.Data)
	if err != nil {
		t.Fatal(err)
	}
	if parsed.Work != work {
		t.Fatalf("trace work drifted: %+v vs %+v", parsed.Work, work)
	}
}
