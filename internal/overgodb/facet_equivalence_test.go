package overgodb

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"path/filepath"
	"sort"
	"testing"

	"overgo/internal/artifact"
)

// catalogDigest reduces every facet to one deterministic digest:
// artifacts in sequence order with manifests and content locators,
// sorted aliases, sorted lineage and causality, sorted locations, and the
// ordered commit record. Two states with equal digests answer every
// catalog read identically.
func catalogDigest(t *testing.T, state catalogState) string {
	t.Helper()
	hash := sha256.New()
	write := func(format string, values ...any) { fmt.Fprintf(hash, format, values...) }
	for _, id := range state.artifacts.bySequence {
		record, ok := state.artifacts.record(id)
		if !ok {
			t.Fatalf("sequence index names unknown artifact %s", id)
		}
		write("artifact/%s/%d/%s/%s/%d\n", id, record.descriptor.Size,
			record.descriptor.MediaType, record.descriptor.Schema, record.sequence)
		if record.hasManifest {
			write("manifest/%s/%d\n", record.manifest.ID, len(record.manifest.Components))
		}
		if locator, ok := state.contents.locator(id); ok {
			write("content/%s/%d/%d/%d\n", id, locator.offset, locator.size, locator.sequence)
		}
		for _, key := range state.lineage.parentsOf(id) {
			write("lineage/%s/%s/%s\n", key.child, key.parent, key.relation)
		}
		if link, found := state.causality.records[id]; found {
			write("causality/%s/%s/%s/%s/%v\n", link.Execution, link.Root, link.Trigger, link.Subject, link.Motivation)
		}
		for _, location := range state.locations.of(id) {
			write("location/%s/%s/%s\n", id, location.Kind, location.Value)
		}
	}
	names := make([]string, 0, state.aliases.count())
	state.aliases.each(func(name string, _ artifact.ID) { names = append(names, name) })
	sort.Strings(names)
	for _, name := range names {
		target, _ := state.aliases.resolve(name)
		write("alias/%s/%s\n", name, target)
	}
	for _, commit := range state.commits.all() {
		write("commit/%s/%s/%d\n", commit.key, commit.id, commit.sequence)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// TestCatalogFacetReplayEquivalence proves the facet-composed catalog
// is path-independent on the deterministic corpus: the state built by
// live commits, the state replayed from the journal alone, and the
// state loaded through the snapshot answer every facet identically.
func TestCatalogFacetReplayEquivalence(t *testing.T) {
	root := t.TempDir()
	head, _, _, _ := buildScaleCorpus(t, root)

	live, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer live.Close()
	snapshotReplay := live.SnapshotReplay()
	if !snapshotReplay.Loaded {
		t.Fatalf("snapshot-loaded path unavailable: %+v", snapshotReplay)
	}
	snapshotDigest := catalogDigest(t, live.state)
	liveHead, _ := live.Head()
	if liveHead != head {
		t.Fatalf("snapshot path head %s, corpus %s", liveHead, head)
	}

	logRoot := t.TempDir()
	copyCorpusFile(t, live.log.file.Name(), logRoot+"/"+storeFilename)
	copyCorpusTree(t, filepath.Join(root, blobDirectory), filepath.Join(logRoot, blobDirectory))
	copyCorpusTree(t, filepath.Join(root, segmentDirectory), filepath.Join(logRoot, segmentDirectory))
	logOnly, err := OpenReadOnly(logRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer logOnly.Close()
	if logOnly.SnapshotReplay().Loaded {
		t.Fatal("journal-only path unexpectedly loaded a snapshot")
	}
	logDigest := catalogDigest(t, logOnly.state)

	if snapshotDigest != logDigest {
		t.Fatalf("facet state diverges by construction path:\nsnapshot %s\njournal  %s", snapshotDigest, logDigest)
	}
}
