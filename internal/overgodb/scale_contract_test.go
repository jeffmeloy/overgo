package overgodb

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
)

// Scale fixture extents. The corpus is exact and re-derivable: content
// bytes come from a hash of the commit ordinal, so two builds of the
// corpus are bitwise-identical stores. The extents bound the fixture's
// wall time; the measurements they produce are reported, never
// asserted, so a measured value can never become a hidden threshold.
const (
	// scaleCorpusCommits is the fixture's transaction extent: large
	// enough that replay, snapshot, and journal measurements dominate
	// per-call noise, small enough for the fast-correctness lane.
	scaleCorpusCommits = 512
	// scaleContentBytes is each artifact's content extent; with the
	// commit count it fixes the corpus's exact content volume.
	scaleContentBytes = 4096
	// scaleAliasStride binds an alias on every stride-th commit so the
	// corpus exercises alias state without alias count dominating.
	scaleAliasStride = 8
)

// buildScaleCorpus commits the deterministic corpus into root and
// returns the head, snapshot path, per-commit wall times, and total
// content bytes.
func buildScaleCorpus(t *testing.T, root string) (artifact.CommitID, string, []time.Duration, uint64) {
	t.Helper()
	store, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	ctx := context.Background()
	latencies := make([]time.Duration, 0, scaleCorpusCommits)
	var contentBytes uint64
	var previous artifact.ID
	for ordinal := range scaleCorpusCommits {
		content := scaleContent(ordinal)
		id, err := artifact.IdentifyBytes(artifact.KindRun, content)
		if err != nil {
			t.Fatal(err)
		}
		descriptor := artifact.Descriptor{
			ID: id, Size: uint64(len(content)), MediaType: "application/octet-stream",
		}
		batch := artifact.Batch{
			Key:       fmt.Sprintf("scale/%d", ordinal),
			Artifacts: []artifact.Descriptor{descriptor},
			Contents:  []artifact.Content{{Descriptor: descriptor, Data: content}},
		}
		if previous.Valid() {
			batch.Lineage = []artifact.Lineage{{Child: id, Parent: previous, Relation: artifact.RelationDerivedFrom}}
		}
		if ordinal%scaleAliasStride == 0 {
			batch.Aliases = []artifact.AliasBinding{{Name: fmt.Sprintf("scale/alias/%d", ordinal), Target: id}}
		}
		start := time.Now()
		if _, err := store.Commit(ctx, batch); err != nil {
			t.Fatalf("commit %d: %v", ordinal, err)
		}
		latencies = append(latencies, time.Since(start))
		contentBytes += uint64(len(content))
		previous = id
	}
	snapshot, err := store.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	head, _ := store.Head()
	return head, snapshot.Path, latencies, contentBytes
}

func scaleContent(ordinal int) []byte {
	content := make([]byte, 0, scaleContentBytes)
	seed := sha256.Sum256([]byte(fmt.Sprintf("scale-corpus/%d", ordinal)))
	for len(content) < scaleContentBytes {
		seed = sha256.Sum256(seed[:])
		content = append(content, seed[:]...)
	}
	return content[:scaleContentBytes]
}

// TestStoreScaleContract pins the store's scale envelope before the
// storage campaign changes it. Assertions are structural only: the
// corpus is bitwise-deterministic across roots, replay reproduces the
// head from the snapshot without fallback, and the catalog answers for
// its population. Replay wall, commit latency, snapshot and journal
// size, resident population, and content-to-metadata ratio are
// measurements, reported in the log for the campaign to compare
// against, never thresholds.
func TestStoreScaleContract(t *testing.T) {
	rootA, rootB := t.TempDir(), t.TempDir()
	headA, snapshotPath, latencies, contentBytes := buildScaleCorpus(t, rootA)
	headB, _, _, _ := buildScaleCorpus(t, rootB)
	if headA != headB {
		t.Fatalf("corpus is not deterministic: head %s != %s", headA, headB)
	}
	journalA := fileDigestAt(t, filepath.Join(rootA, "overgodb.log"))
	journalB := fileDigestAt(t, filepath.Join(rootB, "overgodb.log"))
	if journalA != journalB {
		t.Fatal("corpus journals differ bitwise across roots")
	}

	journalInfo, err := os.Stat(filepath.Join(rootA, "overgodb.log"))
	if err != nil {
		t.Fatal(err)
	}
	snapshotInfo, err := os.Stat(snapshotPath)
	if err != nil {
		t.Fatal(err)
	}
	snapshotSize := snapshotInfo.Size()

	replayStart := time.Now()
	store, err := OpenReadOnly(rootA)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	replayWall := time.Since(replayStart)
	replay := store.SnapshotReplay()
	if !replay.Loaded || replay.Fallback != "" {
		t.Fatalf("snapshot replay not used: loaded=%v fallback=%q", replay.Loaded, replay.Fallback)
	}
	head, sequence := store.Head()
	if head != headA || sequence != scaleCorpusCommits {
		t.Fatalf("replayed head %s seq %d, built %s seq %d", head, sequence, headA, scaleCorpusCommits)
	}

	ctx := context.Background()
	for ordinal := 0; ordinal < scaleCorpusCommits; ordinal += scaleAliasStride {
		content := scaleContent(ordinal)
		id, err := artifact.IdentifyBytes(artifact.KindRun, content)
		if err != nil {
			t.Fatal(err)
		}
		target, found, err := store.ResolveAlias(ctx, fmt.Sprintf("scale/alias/%d", ordinal))
		if err != nil || !found || target != id {
			t.Fatalf("alias %d: found=%v err=%v", ordinal, found, err)
		}
		if ordinal == 0 {
			descriptor, reader, found, err := store.OpenContent(ctx, id)
			if err != nil || !found || descriptor.Size != scaleContentBytes {
				t.Fatalf("content %d: found=%v size=%d err=%v", ordinal, found, descriptor.Size, err)
			}
			replayed := make([]byte, scaleContentBytes)
			if _, err := io.ReadFull(reader, replayed); err != nil {
				t.Fatal(err)
			}
			if sha256.Sum256(replayed) != sha256.Sum256(content) {
				t.Fatalf("content %d does not round-trip", ordinal)
			}
		}
	}

	var totalCommit time.Duration
	worst := latencies[0]
	for _, latency := range latencies {
		totalCommit += latency
		worst = max(worst, latency)
	}
	metadataBytes := uint64(journalInfo.Size()) - contentBytes
	t.Logf("scale envelope: commits=%d content_bytes=%d journal_bytes=%d metadata_bytes=%d content_to_metadata=%.3f",
		scaleCorpusCommits, contentBytes, journalInfo.Size(), metadataBytes,
		float64(contentBytes)/float64(metadataBytes))
	t.Logf("scale envelope: snapshot_bytes=%d replay_wall=%s commit_mean=%s commit_worst=%s",
		snapshotSize, replayWall, totalCommit/scaleCorpusCommits, worst)
}

func fileDigestAt(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}
