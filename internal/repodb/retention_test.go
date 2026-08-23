package repodb

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
)

func retentionContent(t *testing.T, kind artifact.Kind, body any) artifact.Content {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	id, err := artifact.IdentifyBytes(kind, data)
	if err != nil {
		t.Fatal(err)
	}
	return artifact.Content{
		Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data)), MediaType: "application/json", Schema: "test/retention/v1"},
		Data:       data,
	}
}

// TestSupersededCollection pins retention's whole contract: a compacted store
// keeps everything reachable from aliases -- including artifacts referenced
// only inside retained document content -- and leaves superseded documents
// behind, while the source store survives untouched.
func TestSupersededCollection(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	source, err := Open(filepath.Join(root, "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()

	evidence := retentionContent(t, artifact.KindEvidence, map[string]any{"run": "gate evidence body"})
	superseded := retentionContent(t, artifact.KindEvidence, map[string]any{"decision": "old"})
	current := retentionContent(t, artifact.KindEvidence, map[string]any{
		"decision": "new", "verified": evidence.Descriptor.ID.String(),
	})
	if _, err := source.Commit(ctx, artifact.Batch{
		Key:      "test/retention/seed",
		Contents: []artifact.Content{evidence, superseded, current},
		Aliases:  []artifact.AliasBinding{{Name: "test/active", Target: superseded.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := source.Commit(ctx, artifact.Batch{
		Key: "test/retention/supersede",
		Aliases: []artifact.AliasBinding{{
			Name: "test/active", Target: current.Descriptor.ID,
			Previous: artifact.IDPointer(superseded.Descriptor.ID),
		}},
	}); err != nil {
		t.Fatal(err)
	}

	report, err := Compact(ctx, source, filepath.Join(root, "compact"))
	if err != nil {
		t.Fatal(err)
	}
	if report.DroppedArtifacts != 1 {
		t.Fatalf("report = %+v", report)
	}
	compacted, err := OpenReadOnly(filepath.Join(root, "compact"))
	if err != nil {
		t.Fatal(err)
	}
	defer compacted.Close()
	if target, ok, err := compacted.ResolveAlias(ctx, "test/active"); err != nil || !ok || target != current.Descriptor.ID {
		t.Fatalf("alias = %v %v %v", target, ok, err)
	}
	if ok, err := compacted.HasContent(ctx, evidence.Descriptor.ID); err != nil || !ok {
		t.Fatalf("content-referenced evidence dropped: %v %v", ok, err)
	}
	if ok, err := compacted.HasContent(ctx, superseded.Descriptor.ID); err != nil || ok {
		t.Fatalf("superseded document survived compaction: %v %v", ok, err)
	}
	if ok, err := source.HasContent(ctx, superseded.Descriptor.ID); err != nil || !ok {
		t.Fatalf("source archive lost the superseded document: %v %v", ok, err)
	}
}

// TestRetentionRefusesExistingDestination keeps compaction from writing into
// a store that already holds anything.
func TestRetentionRefusesExistingDestination(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	source, err := Open(filepath.Join(root, "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	seeded := retentionContent(t, artifact.KindEvidence, map[string]any{"seed": true})
	if _, err := source.Commit(ctx, artifact.Batch{
		Key: "test/seed", Contents: []artifact.Content{seeded},
		Aliases: []artifact.AliasBinding{{Name: "seed", Target: seeded.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	occupied, err := Open(filepath.Join(root, "occupied"))
	if err != nil {
		t.Fatal(err)
	}
	other := retentionContent(t, artifact.KindEvidence, map[string]any{"other": true})
	if _, err := occupied.Commit(ctx, artifact.Batch{Key: "test/other", Contents: []artifact.Content{other}}); err != nil {
		t.Fatal(err)
	}
	if err := occupied.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Compact(ctx, source, filepath.Join(root, "occupied")); err == nil {
		t.Fatal("compaction wrote into an occupied store")
	}
}
