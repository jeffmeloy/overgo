package repodb

import (
	"context"
	"fmt"
	"testing"

	"overgo/internal/artifact"
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
		id, identifyErr := artifact.IdentifyBytes(artifact.KindRun, payload)
		if identifyErr != nil {
			b.Fatal(identifyErr)
		}
		_, err = store.Commit(context.Background(), artifact.Batch{
			Key: fmt.Sprintf("benchmark/%d", sequence),
			Artifacts: []artifact.Descriptor{{
				ID: id, Size: uint64(len(payload)), MediaType: "application/octet-stream",
			}},
		})
		if err != nil {
			b.Fatal(err)
		}
	}
	if _, err := store.Snapshot(context.Background()); err != nil {
		b.Fatal(err)
	}
	if err := store.Close(); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		opened, openErr := OpenReadOnly(root)
		if openErr != nil {
			b.Fatal(openErr)
		}
		if closeErr := opened.Close(); closeErr != nil {
			b.Fatal(closeErr)
		}
	}
}
