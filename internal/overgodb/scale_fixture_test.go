package overgodb

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
)

var scaleCorpusBuilds, scaleCorpusCopies atomic.Int64
var scaleFixtureRoot string
var scaleFixture = sync.OnceValues(func() (scaleCorpus, error) {
	var err error
	scaleFixtureRoot, err = os.MkdirTemp("", "overgo-scale-corpus-")
	if err != nil {
		return scaleCorpus{}, err
	}
	return buildScaleCorpus(context.Background(), scaleFixtureRoot)
})

func TestMain(m *testing.M) {
	code := m.Run()
	if scaleFixtureRoot != "" {
		if err := os.RemoveAll(scaleFixtureRoot); err != nil {
			fmt.Fprintln(os.Stderr, err)
			code = 1
		}
	}
	fmt.Printf("scale corpus: builds=%d copies=%d\n", scaleCorpusBuilds.Load(), scaleCorpusCopies.Load())
	os.Exit(code)
}

// The process owns one closed source; each test owns a byte copy. Fresh builds
// remain in TestStoreScaleContract to measure commits and test determinism.
func copyScaleCorpus(t *testing.T, root string) (artifact.CommitID, string, uint64) {
	t.Helper()
	fixture, err := scaleFixture()
	if err != nil {
		t.Fatal(err)
	}
	copyCorpusTree(t, fixture.root, root)
	scaleCorpusCopies.Add(1)
	return fixture.head, filepath.Join(root, snapshotDirectory, filepath.Base(fixture.snapshot)), fixture.contentBytes
}

func TestSharedScaleCorpusIsolation(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()
	head, _, size := copyScaleCorpus(t, first)
	otherHead, _, otherSize := copyScaleCorpus(t, second)
	before := drillTreeHash(t, scaleFixtureRoot)
	if head != otherHead || size != otherSize || drillTreeHash(t, first) != before || drillTreeHash(t, second) != before {
		t.Fatal("fixture copies changed the deterministic corpus")
	}
	checkpoint := filepath.Join(first, checkpointDirectory, "aliases"+checkpointExtension)
	data, err := os.ReadFile(checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)/2] ^= 0xff
	if err := os.WriteFile(checkpoint, data, storeFileMode); err != nil {
		t.Fatal(err)
	}
	writer, err := Open(first)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = writer.Close() })
	if writer.SnapshotReplay().Fallback == "" {
		t.Fatal("damaged private checkpoint did not exercise recovery")
	}
	id, err := artifact.IdentifyBytes(artifact.KindRun, []byte("private fixture mutation"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := writer.Commit(t.Context(), artifact.Batch{Key: "fixture/private", Artifacts: []artifact.Descriptor{{ID: id}}}); err != nil {
		t.Fatal(err)
	}
	if current, _ := writer.Head(); current == head {
		t.Fatal("private write did not change its own head")
	}
	reader, err := OpenReadOnly(second)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reader.Close() })
	if current, _ := reader.Head(); current != head || !reader.SnapshotReplay().Loaded {
		t.Fatal("private corruption or write reached a sibling")
	}
	if drillTreeHash(t, scaleFixtureRoot) != before || drillTreeHash(t, second) != before {
		t.Fatal("private mutation changed the source or sibling bytes")
	}
}
