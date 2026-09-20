package overgodb

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
)

// retentionContent is a document large enough to be a blob: retention,
// release and cloning are about blob files, and content under
// inlineContentLimit rides in its frame and has none.
func retentionContent(t *testing.T, kind artifact.Kind, body any) artifact.Content {
	t.Helper()
	data, err := json.Marshal(map[string]any{"body": body, "padding": strings.Repeat("-", inlineContentLimit)})
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

// releaseEverything admits every class: a test that releases with it proves
// its consumer needs nothing outside the live set.
func releaseEverything(artifact.Descriptor) bool { return true }

// TestRetentionTypedLineage keeps what typed lineage reaches and releases
// what only an untyped reference inside a document names.
func TestRetentionTypedLineage(t *testing.T) {
	ctx := t.Context()
	source, err := Open(filepath.Join(t.TempDir(), "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()

	evidence := retentionContent(t, artifact.KindEvidence, map[string]any{"run": "gate evidence body"})
	superseded := retentionContent(t, artifact.KindEvidence, map[string]any{"decision": "old"})
	current := retentionContent(t, artifact.KindEvidence, map[string]any{
		"decision": "new", "untyped_reference": superseded.Descriptor.ID.String(),
	})
	if _, err := source.Commit(ctx, artifact.Batch{
		Key:      "test/retention/seed",
		Contents: []artifact.Content{evidence, superseded, current},
		Lineage: []artifact.Lineage{{
			Child: current.Descriptor.ID, Parent: evidence.Descriptor.ID, Relation: artifact.RelationDependsOn,
		}},
		Aliases: []artifact.AliasBinding{{Name: "test/active", Target: superseded.Descriptor.ID}},
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

	report, err := Release(ctx, source, nil, RetentionPolicy{}, releaseEverything)
	if err != nil {
		t.Fatal(err)
	}
	if report.Released != 1 || report.Retained != 2 {
		t.Fatalf("report = %+v", report)
	}
	if target, ok, err := source.ResolveAlias(ctx, "test/active"); err != nil || !ok || target != current.Descriptor.ID {
		t.Fatalf("alias = %v %v %v", target, ok, err)
	}
	if ok, err := source.HasContent(ctx, evidence.Descriptor.ID); err != nil || !ok {
		t.Fatalf("typed parent released: %v %v", ok, err)
	}
	if ok, err := source.HasContent(ctx, superseded.Descriptor.ID); err != nil || ok {
		t.Fatalf("superseded document survived the release: %v %v", ok, err)
	}
}

// TestRetentionPinnedRootsSurvive keeps alias-less content the repository's
// committed documents pin: retained when pinned with a missing pin counted,
// never invented, and released once nothing pins it.
func TestRetentionPinnedRootsSurvive(t *testing.T) {
	ctx := t.Context()
	source, err := Open(filepath.Join(t.TempDir(), "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	aliased := retentionContent(t, artifact.KindEvidence, map[string]any{"stage": "aliased"})
	pinned := retentionContent(t, artifact.KindEvidence, map[string]any{"stage": "pinned"})
	if _, err := source.Commit(ctx, artifact.Batch{
		Key: "test/retention/pinned", Contents: []artifact.Content{aliased, pinned},
		Aliases: []artifact.AliasBinding{{Name: "test/aliased", Target: aliased.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}
	missing := retentionContent(t, artifact.KindEvidence, map[string]any{"stage": "absent"}).Descriptor.ID
	report, err := Release(ctx, source, nil, RetentionPolicy{Roots: []artifact.ID{pinned.Descriptor.ID, missing}}, releaseEverything)
	if err != nil {
		t.Fatal(err)
	}
	if report.PinnedRoots != 2 || report.PinnedMissing != 1 || report.Released != 0 {
		t.Fatalf("pinned accounting = %+v", report)
	}
	if ok, err := source.HasContent(ctx, pinned.Descriptor.ID); err != nil || !ok {
		t.Fatalf("pinned content was released: %v %v", ok, err)
	}
	if report, err = Release(ctx, source, nil, RetentionPolicy{}, releaseEverything); err != nil || report.Released != 1 {
		t.Fatalf("unpinned release = (%+v, %v)", report, err)
	}
	if ok, err := source.HasContent(ctx, pinned.Descriptor.ID); err != nil || ok {
		t.Fatalf("alias-less content survived a release without a root: %v %v", ok, err)
	}
}

// TestRetentionRefusesLiveStoreLocalAlias refuses a release while a
// store-local authority is live, and such an alias can never be retired.
func TestRetentionRefusesLiveStoreLocalAlias(t *testing.T) {
	ctx := t.Context()
	source, err := Open(filepath.Join(t.TempDir(), "source"))
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	content := retentionContent(t, artifact.KindEvidence, map[string]any{"receipt": "physical"})
	localAlias := StoreLocalAliasPrefix + "fixture/current"
	if _, err := source.Commit(ctx, artifact.Batch{
		Key: "test/retention/store-local", Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{
			{Name: localAlias, Target: content.Descriptor.ID},
			{Name: "test/portable", Target: content.Descriptor.ID},
		},
	}); err != nil {
		t.Fatal(err)
	}
	_, sequence := source.Head()
	if _, err := Release(ctx, source, nil, RetentionPolicy{}, releaseEverything); err == nil ||
		!strings.Contains(err.Error(), localAlias) {
		t.Fatalf("store-local release error = %v", err)
	}
	if _, err := source.Commit(ctx, artifact.Batch{
		Key: "test/retention/retire-store-local",
		Aliases: []artifact.AliasBinding{{
			Name: localAlias, Target: content.Descriptor.ID,
			Previous: artifact.IDPointer(content.Descriptor.ID), Remove: true,
		}},
	}); err == nil || !strings.Contains(err.Error(), "cannot be retired") {
		t.Fatalf("store-local retirement error = %v", err)
	}
	if _, after := source.Head(); after != sequence {
		t.Fatalf("refused operations advanced the chain %d -> %d", sequence, after)
	}
}
