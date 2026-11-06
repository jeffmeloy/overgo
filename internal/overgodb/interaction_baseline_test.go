package overgodb_test

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
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
	t.Parallel()
	root := t.TempDir()
	store := testcheck.Must(overgodb.Create(root))(t)
	defer store.Close()
	ctx := t.Context()
	var work runrecord.InteractionWork
	var lastID artifact.ID
	for ordinal := range baselineCommits {
		payload := fmt.Appendf(nil, "interaction-baseline/%d", ordinal)
		id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, payload))(t)
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
	for ordinal := range baselineReadSample {
		payload := fmt.Appendf(nil, "interaction-baseline/%d", ordinal)
		id := testcheck.Must(artifact.IdentifyBytes(artifact.KindRun, payload))(t)
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
	if _, err := io.ReadAll(reader); err != nil {
		t.Fatal(err)
	}
	work.BlobReads++

	journal := testcheck.Must(os.ReadFile(filepath.Join(root, "overgodb.log")))(t)
	work.Bytes = uint64(len(journal))

	if work.Commits != baselineCommits || work.SemanticTransitions != baselineCommits ||
		work.ArtifactReads != baselineReadSample || work.ReturnedFacts != baselineReadSample ||
		work.BlobReads != 1 || work.Bytes == 0 {
		t.Fatalf("counted work does not match construction: %+v", work)
	}

	digest := sha256.Sum256(journal)
	evidence := testcheck.Must(artifact.IdentifyBytes(artifact.KindEvidence, digest[:]))(t)
	trace := testcheck.Must(runrecord.NewEfficiencyTrace(runrecord.EfficiencyTrace{
		Surface:  runrecord.SurfaceStorage,
		Task:     "storage baseline session",
		Work:     work,
		Result:   lastID,
		Evidence: evidence,
	}))(t)
	content := testcheck.Must(trace.Content())(t)
	parsed := testcheck.Must(runrecord.ParseEfficiencyTrace(content.Data))(t)
	if parsed.Work != work {
		t.Fatalf("trace work drifted: %+v vs %+v", parsed.Work, work)
	}
}
