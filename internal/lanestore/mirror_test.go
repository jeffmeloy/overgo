package lanestore

import (
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

// TestMirrorActivationsCopiesClosureAndReleasesStale pins the mirror: the
// source's activation, its definition, the record the definition names and
// the alias that embeds a closure id reach the target; an activation the
// target alone holds is released; a repeated mirror copies and commits
// nothing, since every batch is named by its own records.
func TestMirrorActivationsCopiesClosureAndReleasesStale(t *testing.T) {
	ctx := t.Context()
	open := func(name string) *overgodb.Store {
		store, err := overgodb.Open(filepath.Join(t.TempDir(), name))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = store.Close() })
		return store
	}
	source, target := open("source"), open("target")
	content := func(kind artifact.Kind, value string) artifact.Content {
		t.Helper()
		payload := []byte(value)
		id, err := artifact.IdentifyBytes(kind, payload)
		if err != nil {
			t.Fatal(err)
		}
		return artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(payload)), MediaType: "text/plain"}, Data: payload}
	}
	model := content(artifact.KindModel, "model")
	child := content(artifact.KindEvidence, "child evidence")
	viaAlias := content(artifact.KindEvidence, "reached through the status alias")
	definition := content(artifact.KindEvidence, "definition names "+child.Descriptor.ID.String())
	active := modelrecipe.ActiveAlias(model.Descriptor.ID, recipe.TaskInference)
	if _, err := source.Commit(ctx, artifact.Batch{
		Key: "fixture/source", Contents: []artifact.Content{model, child, viaAlias, definition},
		Aliases: []artifact.AliasBinding{
			{Name: active, Target: definition.Descriptor.ID},
			{Name: "recipe.status." + definition.Descriptor.ID.String(), Target: viaAlias.Descriptor.ID},
		},
	}); err != nil {
		t.Fatal(err)
	}
	stale := content(artifact.KindModel, "stale model")
	staleDefinition := content(artifact.KindEvidence, "stale definition")
	staleAlias := modelrecipe.ActiveAlias(stale.Descriptor.ID, recipe.TaskInference)
	if _, err := target.Commit(ctx, artifact.Batch{
		Key: "fixture/target", Contents: []artifact.Content{stale, staleDefinition},
		Aliases: []artifact.AliasBinding{{Name: staleAlias, Target: staleDefinition.Descriptor.ID}},
	}); err != nil {
		t.Fatal(err)
	}

	report, err := MirrorActivations(ctx, source, target, MirrorDepth)
	if err != nil {
		t.Fatal(err)
	}
	if report.SourceActive != 1 || report.Copied["content"] != 4 || !slices.Equal(report.Released, []string{staleAlias}) || len(report.Commits) != 2 {
		t.Fatalf("report = %+v", report)
	}
	for _, id := range []artifact.ID{model.Descriptor.ID, child.Descriptor.ID, viaAlias.Descriptor.ID, definition.Descriptor.ID} {
		if present, err := target.HasContent(ctx, id); err != nil || !present {
			t.Fatalf("target lacks %s: %v", id, err)
		}
	}
	if current, bound, err := target.ResolveAlias(ctx, active); err != nil || !bound || current != definition.Descriptor.ID {
		t.Fatalf("target activation = %s bound=%t, %v", current, bound, err)
	}
	if _, bound, err := target.ResolveAlias(ctx, staleAlias); err != nil || bound {
		t.Fatalf("stale activation still bound=%t, %v", bound, err)
	}

	again, err := MirrorActivations(ctx, source, target, MirrorDepth)
	if err != nil {
		t.Fatal(err)
	}
	if len(again.Copied) != 0 || again.Bindings != 0 || len(again.Commits) != 0 {
		t.Fatalf("repeated mirror = %+v, want nothing copied or committed", again)
	}
}
