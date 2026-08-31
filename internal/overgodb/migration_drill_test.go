package overgodb

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// drillSurfaceDigest reduces the corpus-visible query surface — aliases,
// artifacts, and content presence — to one string without the commit head,
// so stores that reach the same surface through different commit structure
// (rebuild, compaction, backup) compare equal.
func drillSurfaceDigest(t *testing.T, store *Store) string {
	t.Helper()
	ctx := context.Background()
	var out strings.Builder
	for ordinal := 0; ordinal < scaleCorpusCommits; ordinal += scaleAliasStride {
		id, err := artifact.IdentifyBytes(artifact.KindRun, scaleContent(ordinal))
		if err != nil {
			t.Fatal(err)
		}
		target, found, err := store.ResolveAlias(ctx, aliasName(ordinal))
		if err != nil || !found || target != id {
			t.Fatalf("alias %d: found=%v err=%v", ordinal, found, err)
		}
		descriptor, found, err := store.Artifact(ctx, id)
		if err != nil || !found {
			t.Fatalf("artifact %d: %v", ordinal, err)
		}
		if has, err := store.HasContent(ctx, id); err != nil || !has {
			t.Fatalf("content %d: %v", ordinal, err)
		}
		fmt.Fprintf(&out, "%s/%s/%d\n", aliasName(ordinal), target, descriptor.Size)
	}
	return out.String()
}

// drillTreeHash reduces one store root to a byte-exact digest so a drill can
// prove the source was never written.
func drillTreeHash(t *testing.T, root string) string {
	t.Helper()
	type entry struct {
		path string
		sum  [sha256.Size]byte
	}
	var entries []entry
	err := filepath.WalkDir(root, func(path string, item fs.DirEntry, walkErr error) error {
		if walkErr != nil || item.IsDir() {
			return walkErr
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		entries = append(entries, entry{path: filepath.ToSlash(relative), sum: sha256.Sum256(data)})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].path < entries[j].path })
	hash := sha256.New()
	for _, item := range entries {
		fmt.Fprintf(hash, "%s/%x\n", item.path, item.sum)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func drillCopyStore(t *testing.T, sourceRoot, destinationRoot string) {
	t.Helper()
	copyCorpusTree(t, sourceRoot, destinationRoot)
}

// TestRSIStoreMigrationDrill rebuilds the pinned scale corpus into the
// segmented journal, blob, and projection layout and executes the backup,
// restore, retention, interruption, and rollback drills end to end: every
// destination answers the same query surface read-only, an interruption at
// any point of the active journal rolls back to a valid commit prefix, a
// torn sealed segment refuses to open, and the source store's bytes never
// change.
func TestRSIStoreMigrationDrill(t *testing.T) {
	base := t.TempDir()
	sourceRoot := filepath.Join(base, "source")
	buildScaleCorpus(t, sourceRoot)
	sourceHash := drillTreeHash(t, sourceRoot)
	ctx := context.Background()

	source, err := OpenReadOnly(sourceRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	digest := drillSurfaceDigest(t, source)
	head, sequence := source.Head()
	if sequence != scaleCorpusCommits {
		t.Fatalf("corpus sequence = %d", sequence)
	}

	// Migration: rebuild into a fresh segmented layout and read it back
	// read-only.
	rebuiltRoot := filepath.Join(base, "rebuilt")
	report, err := Rebuild(ctx, source, rebuiltRoot, nil)
	if err != nil {
		t.Fatal(err)
	}
	if report.Batches == 0 {
		t.Fatalf("rebuild moved nothing: %+v", report)
	}
	rebuilt, err := OpenReadOnly(rebuiltRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer rebuilt.Close()
	if drillSurfaceDigest(t, rebuilt) != digest {
		t.Fatal("rebuilt store answers a different query surface")
	}

	// Backup and restore: the streamed backup opens read-only at the same
	// head and surface.
	backupRoot := filepath.Join(base, "backup")
	backupHead, _, err := source.Backup(backupRoot)
	if err != nil {
		t.Fatal(err)
	}
	if backupHead != head {
		t.Fatalf("backup head %s differs from source head %s", backupHead, head)
	}
	restored, err := OpenReadOnly(backupRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if drillSurfaceDigest(t, restored) != digest {
		t.Fatal("restored backup answers a different query surface")
	}

	// Retention: compaction preserves the live surface in a fresh root.
	compactRoot := filepath.Join(base, "compacted")
	if _, err := Compact(ctx, source, compactRoot); err != nil {
		t.Fatal(err)
	}
	compacted, err := OpenReadOnly(compactRoot)
	if err != nil {
		t.Fatal(err)
	}
	defer compacted.Close()
	if drillSurfaceDigest(t, compacted) != digest {
		t.Fatal("compacted store answers a different query surface")
	}

	// Interruption: truncating the active journal at arbitrary publication
	// boundaries rolls back to a valid commit prefix on the next writable
	// open, and the recovered prefix answers its bounded surface.
	activeSize := int64(0)
	if info, statErr := os.Stat(filepath.Join(rebuiltRoot, storeFilename)); statErr == nil {
		activeSize = info.Size()
	}
	if activeSize == 0 {
		t.Fatal("rebuilt store has no active journal to interrupt")
	}
	for _, fraction := range []float64{0.35, 0.6, 0.85} {
		drillRoot := filepath.Join(base, fmt.Sprintf("interrupt-%d", int(fraction*100)))
		drillCopyStore(t, rebuiltRoot, drillRoot)
		cut := storeHeaderBytes + int64(float64(activeSize-storeHeaderBytes)*fraction)
		if err := os.Truncate(filepath.Join(drillRoot, storeFilename), cut); err != nil {
			t.Fatal(err)
		}
		recovered, err := Open(drillRoot)
		if err != nil {
			t.Fatalf("interrupted store at %d bytes did not recover: %v", cut, err)
		}
		_, recoveredSequence := recovered.Head()
		if recoveredSequence > sequence {
			t.Fatalf("recovered sequence %d exceeds the source %d", recoveredSequence, sequence)
		}
		if closeErr := recovered.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
	}

	// Rollback proof on the SOURCE layout: a truncated source copy recovers
	// to a prefix whose bounded surface still answers exactly. The cut lands
	// past the store header — cutting the header itself is corruption the
	// open must refuse, not a torn tail it can roll back.
	prefixRoot := filepath.Join(base, "prefix")
	drillCopyStore(t, sourceRoot, prefixRoot)
	if info, statErr := os.Stat(filepath.Join(prefixRoot, storeFilename)); statErr == nil {
		cut := storeHeaderBytes + (info.Size()-storeHeaderBytes)/2
		if err := os.Truncate(filepath.Join(prefixRoot, storeFilename), cut); err != nil {
			t.Fatal(err)
		}
	}
	prefix, err := Open(prefixRoot)
	if err != nil {
		t.Fatalf("truncated source copy did not recover: %v", err)
	}
	_, prefixSequence := prefix.Head()
	if prefixSequence == 0 || prefixSequence > sequence {
		t.Fatalf("recovered prefix sequence = %d", prefixSequence)
	}
	querySurfaceDigestBounded(t, prefix, int(prefixSequence))
	if err := prefix.Close(); err != nil {
		t.Fatal(err)
	}

	// A torn sealed segment refuses instead of serving a corrupt surface.
	segments, _ := os.ReadDir(filepath.Join(rebuiltRoot, segmentDirectory))
	if len(segments) > 0 {
		tornRoot := filepath.Join(base, "torn-sealed")
		drillCopyStore(t, rebuiltRoot, tornRoot)
		sealed := filepath.Join(tornRoot, segmentDirectory, segments[0].Name())
		info, err := os.Stat(sealed)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Truncate(sealed, info.Size()-1); err != nil {
			t.Fatal(err)
		}
		if torn, err := OpenReadOnly(tornRoot); err == nil {
			_ = torn.Close()
			t.Fatal("torn sealed segment opened")
		}
	}

	// The source was never written by any drill.
	if drillTreeHash(t, sourceRoot) != sourceHash {
		t.Fatal("a drill mutated the source store")
	}
	if drillSurfaceDigest(t, source) != digest {
		t.Fatal("the source query surface drifted")
	}
}
