package overgodb

import (
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

const benchmarkReplayCommits = 256

func BenchmarkSnapshotReplay(b *testing.B) {
	root := b.TempDir()
	store, err := Open(root)
	if err != nil {
		b.Fatal(err)
	}
	for sequence := range benchmarkReplayCommits {
		payload := []byte(fmt.Sprintf("benchmark-artifact-%d", sequence))
		id := testutil.ArtifactBytesID(b, artifact.KindRun, payload)
		_, err = store.Commit(b.Context(), artifact.Batch{
			Key: fmt.Sprintf("benchmark/%d", sequence),
			Artifacts: []artifact.Descriptor{{
				ID: id, Size: uint64(len(payload)), MediaType: "application/octet-stream",
			}},
		})
		if err != nil {
			b.Fatal(err)
		}
	}
	if _, err := store.Snapshot(b.Context()); err != nil {
		b.Fatal(err)
	}
	if err := store.Close(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		opened, openErr := OpenReadOnly(root)
		if openErr != nil {
			b.Fatal(openErr)
		}
		if closeErr := opened.Close(); closeErr != nil {
			b.Fatal(closeErr)
		}
	}
}

func BenchmarkCompaction(b *testing.B) {
	ctx := b.Context()
	source, err := Open(b.TempDir())
	if err != nil {
		b.Fatal(err)
	}
	defer source.Close()
	var previous artifact.ID
	for sequence := range benchmarkReplayCommits {
		data := []byte(fmt.Sprintf("benchmark-content-%d", sequence))
		id := testutil.ArtifactBytesID(b, artifact.KindEvidence, data)
		batch := artifact.Batch{
			Key: fmt.Sprintf("benchmark/compact/%d", sequence),
			Contents: []artifact.Content{{
				Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: "application/octet-stream"},
				Data:       data,
			}},
		}
		if previous.Valid() {
			batch.Lineage = []artifact.Lineage{{Child: id, Parent: previous, Relation: artifact.RelationDependsOn}}
		}
		if _, err := source.Commit(ctx, batch); err != nil {
			b.Fatal(err)
		}
		previous = id
	}
	if _, err := source.Commit(ctx, artifact.Batch{
		Key: "benchmark/compact/alias", Aliases: []artifact.AliasBinding{{Name: "benchmark/active", Target: previous}},
	}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := Compact(ctx, source, b.TempDir()); err != nil {
			b.Fatal(err)
		}
	}
}
