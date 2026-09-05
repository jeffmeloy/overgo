package modelartifact

import (
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

func TestRawTextPublicationPreservesImmutableFacts(t *testing.T) {
	for _, tc := range []struct {
		name, priorType, nextType, schema string
		kind                              artifact.Kind
		sizeMismatch, reject              bool
	}{
		{name: "plain then markdown", priorType: "text/plain", nextType: "text/markdown", kind: artifact.KindFile},
		{name: "markdown then plain", priorType: "text/markdown", nextType: "text/plain", kind: artifact.KindFile},
		{name: "binary conflict", priorType: "application/octet-stream", nextType: "text/markdown", kind: artifact.KindFile, reject: true},
		{name: "typed document", priorType: "text/plain", nextType: "text/markdown", schema: "fixture/v1", kind: artifact.KindFile, reject: true},
		{name: "size conflict", priorType: "text/plain", nextType: "text/markdown", kind: artifact.KindFile, sizeMismatch: true, reject: true},
		{name: "tensor container", priorType: "text/plain", nextType: "text/markdown", kind: artifact.KindTensorSet, reject: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			data := []byte("# A license declaration\n")
			id, err := artifact.IdentifyBytes(tc.kind, data)
			if err != nil {
				t.Fatal(err)
			}
			prior := artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: tc.priorType, Schema: tc.schema}
			if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "prior", Artifacts: []artifact.Descriptor{prior}}); err != nil {
				t.Fatal(err)
			}
			next := prior
			next.MediaType = tc.nextType
			if tc.sizeMismatch {
				next.Size++
			}
			for _, inline := range []bool{false, true} {
				batch := artifact.Batch{Key: "next"}
				if inline {
					batch.Contents = []artifact.Content{{Descriptor: next, Data: data}}
				} else {
					batch.Artifacts = []artifact.Descriptor{next}
				}
				err := PreserveRawTextDescriptors(t.Context(), store, &batch)
				if err == nil {
					_, err = artifact.CommitBatch(t.Context(), store, batch)
				}
				if tc.reject {
					if err == nil || errors.Is(err, artifact.ErrNoChange) {
						t.Fatal("conflicting facts were accepted")
					}
					continue
				}
				if err != nil && !errors.Is(err, artifact.ErrNoChange) {
					t.Fatal(err)
				}
				existing, found, err := store.Artifact(t.Context(), id)
				if err != nil || !found || existing != prior {
					t.Fatalf("immutable descriptor changed: %+v, %v", existing, err)
				}
			}
		})
	}
}
