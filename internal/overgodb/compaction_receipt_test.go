package overgodb_test

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gitauthority"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// compactionFootprint is one measured on-disk footprint of a store.
type compactionFootprint struct {
	Bytes           int64 `json:"bytes"`
	Files           int64 `json:"files"`
	JournalBytes    int64 `json:"journal_bytes"`
	BlobBytes       int64 `json:"blob_bytes"`
	BlobFiles       int64 `json:"blob_files"`
	SnapshotBytes   int64 `json:"snapshot_bytes"`
	CheckpointBytes int64 `json:"checkpoint_bytes"`
	SegmentBytes    int64 `json:"segment_bytes"`
	ObjectBytes     int64 `json:"object_bytes"`
}

// chainBytes is the canonical-fact footprint: the active journal plus the
// sealed segments. A compaction that keeps the chain never shrinks it.
func (footprint compactionFootprint) chainBytes() int64 {
	return footprint.JournalBytes + footprint.SegmentBytes
}

// compactionReceipt records one in-place compaction of the live store: the
// sealed backup taken first, the footprint before and after, what the
// release kept and let go, the chain coordinates it preserved, one
// git-bound authority probed across it, and the rewrite it superseded.
type compactionReceipt struct {
	Version int    `json:"version"`
	Commit  string `json:"commit"`
	Store   string `json:"store"`
	Backup  struct {
		Destination string `json:"destination"`
		Head        string `json:"head"`
		Sequence    uint64 `json:"sequence"`
		Extent      int64  `json:"extent"`
		BytesCopied int64  `json:"bytes_copied"`
		FilesCopied int64  `json:"files_copied"`
	} `json:"backup"`
	Before  compactionFootprint `json:"before"`
	Release struct {
		// Classes are the re-derivable schemas the release admitted.
		Classes          []string `json:"classes"`
		Retained         int      `json:"retained"`
		PinnedRoots      int      `json:"pinned_roots"`
		PinnedMissing    int      `json:"pinned_missing"`
		Unreachable      int      `json:"unreachable"`
		ReleasedContents int      `json:"released_contents"`
		ReleasedBytes    int64    `json:"released_bytes"`
		Commits          int      `json:"commits"`
		// Inline counts released contents whose bytes sit inline in a
		// legacy journal frame and so free no file.
		Inline       int `json:"inline"`
		BlobsRemoved int `json:"blobs_removed"`
		BlobsAbsent  int `json:"blobs_absent"`
	} `json:"release"`
	// Chain records store commit identities in their hex rendering.
	Chain struct {
		HeadBefore     string `json:"head_before"`
		SequenceBefore uint64 `json:"sequence_before"`
		HeadAfter      string `json:"head_after"`
		SequenceAfter  uint64 `json:"sequence_after"`
	} `json:"chain"`
	// AuthorityProbe is the gate preparation the commit under test names in
	// its Overgo-Gate-Preparation trailers: the artifact and the store commit
	// that introduced it, which the release must leave exact.
	AuthorityProbe struct {
		Artifact artifact.ID `json:"artifact"`
		Commit   string      `json:"commit"`
		Sequence uint64      `json:"sequence"`
	} `json:"authority_probe"`
	After       compactionFootprint `json:"after"`
	OpenSeconds struct {
		Before float64 `json:"before"`
		After  float64 `json:"after"`
	} `json:"open_seconds"`
	StoreCheck string `json:"store_check"`
	// SupersededAttempts are the operations taken out of service before
	// this one, each with the reason; the receipt keeps them because each
	// names a property the store's compaction must have.
	SupersededAttempts []struct {
		Operation   string `json:"operation"`
		Destination string `json:"destination"`
		Outcome     string `json:"outcome"`
	} `json:"superseded_attempts"`
	Commands []string `json:"commands"`
}

// validCommitHex reports the hex rendering of a store commit identity.
func validCommitHex(text string) bool {
	decoded, err := hex.DecodeString(text)
	return err == nil && len(decoded) == sha256.Size
}

// TestCompactionReceipt proves the recorded compaction is internally sound --
// a sealed backup names the pre-release chain, the release kept the chain and
// shrank the blob tree, the store passed store-check afterwards -- and, when
// a data root is present, that the live store still carries that chain: the
// pre-release head at its sequence and the probed gate preparation at its
// recorded introduction commit.
func TestCompactionReceipt(t *testing.T) {
	root := testutil.RepoRoot(t)
	var receipt compactionReceipt
	if err := jsonfile.DecodeStrict(filepath.Join(root, "docs/verification/overgodb_compaction.json"), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Version != 2 || !gitauthority.ValidObjectID(receipt.Commit) || receipt.Store == "" || len(receipt.Commands) == 0 {
		t.Fatalf("incomplete compaction receipt: %+v", receipt)
	}
	backup, chain := receipt.Backup, receipt.Chain
	if backup.Destination == "" || backup.Extent <= 0 || backup.BytesCopied <= 0 || backup.FilesCopied <= 0 ||
		backup.Head != chain.HeadBefore || backup.Sequence != chain.SequenceBefore {
		t.Fatalf("backup seal does not cover the pre-release chain: %+v vs %+v", backup, chain)
	}
	release := receipt.Release
	if !validCommitHex(chain.HeadBefore) || !validCommitHex(chain.HeadAfter) || chain.HeadBefore == chain.HeadAfter ||
		chain.SequenceAfter != chain.SequenceBefore+uint64(release.Commits) || release.Commits <= 0 {
		t.Fatalf("release did not extend the chain by its commits: %+v", chain)
	}
	// A release admits classes, never everything the roots miss: evidence
	// records are discovered through lineage children and memo keys too.
	if len(release.Classes) == 0 || release.Retained <= 0 || release.PinnedRoots <= 0 ||
		release.ReleasedContents <= 0 || release.ReleasedContents >= release.Unreachable ||
		release.ReleasedBytes <= 0 || release.ReleasedContents != release.Inline+release.BlobsRemoved+release.BlobsAbsent {
		t.Fatalf("release counts are inconsistent: %+v", release)
	}
	before, after := receipt.Before, receipt.After
	if before.Bytes <= 0 || before.Files <= 0 || before.BlobFiles <= 0 || before.chainBytes() <= 0 {
		t.Fatalf("missing pre-release footprint: %+v", before)
	}
	if after.Bytes >= before.Bytes || after.BlobFiles >= before.BlobFiles || after.BlobBytes >= before.BlobBytes {
		t.Fatalf("release did not shrink the blob tree: before %+v after %+v", before, after)
	}
	if after.chainBytes() < before.chainBytes() {
		t.Fatalf("release shrank the commit chain: before %d after %d", before.chainBytes(), after.chainBytes())
	}
	if int64(before.BlobFiles-after.BlobFiles) != int64(release.BlobsRemoved) {
		t.Fatalf("blob files fell by %d, release removed %d", before.BlobFiles-after.BlobFiles, release.BlobsRemoved)
	}
	probe := receipt.AuthorityProbe
	if !probe.Artifact.Valid() || !validCommitHex(probe.Commit) || probe.Sequence == 0 || probe.Sequence > chain.SequenceBefore {
		t.Fatalf("authority probe does not name a pre-release introduction: %+v", probe)
	}
	if receipt.StoreCheck != "passed" || len(receipt.SupersededAttempts) == 0 ||
		receipt.OpenSeconds.Before <= 0 || receipt.OpenSeconds.After <= 0 {
		t.Fatalf("receipt lacks the store check, the superseded attempts or the open timings: %+v", receipt)
	}
	for _, attempt := range receipt.SupersededAttempts {
		if attempt.Operation == "" || attempt.Outcome == "" {
			t.Fatalf("superseded attempt lacks its operation or outcome: %+v", attempt)
		}
	}
	// Live check: the store in service must carry the pre-release chain
	// exactly. Later commits extend the chain, so its bounds are the recorded
	// coordinates, never the head; the store's size is no bound at all -- it
	// regrows with every gate, and the growth ratchet owns that measure.
	if os.Getenv(dataroot.Env) == "" {
		return
	}
	roots, err := dataroot.Resolve(root)
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatalf("data root names a store that does not open: %v", err)
	}
	defer store.Close()
	ctx := t.Context()
	if _, sequence := store.Head(); sequence < chain.SequenceAfter {
		t.Fatalf("live store at sequence %d predates the release at %d", sequence, chain.SequenceAfter)
	}
	commit, found, err := store.CommitAt(ctx, chain.SequenceBefore)
	if err != nil || !found || commit.ID.String() != chain.HeadBefore {
		t.Fatalf("live commit %d = (%+v, %v, %v), want %s", chain.SequenceBefore, commit, found, err, chain.HeadBefore)
	}
	introduction, found, err := store.ArtifactIntroduction(ctx, probe.Artifact)
	if err != nil || !found || introduction.Commit.String() != probe.Commit || introduction.Sequence != probe.Sequence {
		t.Fatalf("live introduction of %s = (%+v, %v, %v), want %s at %d",
			probe.Artifact, introduction, found, err, probe.Commit, probe.Sequence)
	}
}
