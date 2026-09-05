package discovery

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestSelectedServableScope(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	paths := make([]string, 0, 2)
	models := make([]artifact.ID, 0, 2)
	for _, name := range []string{"selected", "unrelated"} {
		payload := []byte(name)
		id := testutil.ArtifactBytesID(t, artifact.KindModel, payload)
		path := filepath.Join(t.TempDir(), name+".gguf")
		if err := os.WriteFile(path, payload, 0o600); err != nil {
			t.Fatal(err)
		}
		manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: id}})
		if err != nil {
			t.Fatal(err)
		}
		_, err = store.Commit(t.Context(), artifact.Batch{Key: "selection/" + name,
			Artifacts: []artifact.Descriptor{{ID: id, Size: uint64(len(payload))}}, Manifests: []artifact.Manifest{manifest},
			Locations: []artifact.LocationEvent{{Action: artifact.LocationAdd, Location: artifact.Location{Artifact: id, Kind: artifact.LocationFile, Value: path}}},
		})
		if err != nil {
			t.Fatal(err)
		}
		publishVerifiedActivation(t, store, manifest.ID, name)
		paths, models = append(paths, path), append(models, manifest.ID)
	}
	before, sequence := store.Head()
	memo := NewMemo()
	entries, err := ServableWithMemo(t.Context(), store, 2*len(models), memo, strings.ToUpper(filepath.ToSlash(paths[0])))
	if err != nil || len(entries) != 1 || entries[0].Model != models[0] {
		t.Errorf("selected catalog = %+v, %v", entries, err)
	}
	if len(memo.entries) != 1 {
		t.Errorf("selected admission hashed %d files, want only its one component", len(memo.entries))
	}
	if _, hashed := memo.entries[paths[1]+"\x00"+artifact.KindModel.String()]; hashed {
		t.Error("unrelated model bytes were hashed")
	}
	all, err := ServableWithMemo(t.Context(), store, 2*len(models), NewMemo())
	if err != nil || len(all) != len(paths) {
		t.Fatalf("full catalog changed: %+v, %v", all, err)
	}
	if _, err := ServableWithMemo(t.Context(), store, 1, NewMemo()); err == nil {
		t.Fatal("truncated catalog reported complete discovery")
	}
	if after, current := store.Head(); after != before || current != sequence {
		t.Fatal("catalog read published state")
	}
	// An unrelated corrupt activation must not even be resolved in selected mode.
	alias := "recipe.active.inference." + models[1].String()
	previous, _, err := store.ResolveAlias(t.Context(), alias)
	if err != nil {
		t.Fatal(err)
	}
	broken := testutil.ArtifactID(t, artifact.KindRecipe, "missing-definition")
	_, err = store.Commit(t.Context(), artifact.Batch{Key: "selection/corrupt-unrelated",
		Artifacts: []artifact.Descriptor{{ID: broken}}, Aliases: []artifact.AliasBinding{{Name: alias, Target: broken, Previous: &previous}},
	})
	if err != nil {
		t.Fatal(err)
	}
	entries, err = ServableWithMemo(t.Context(), store, 2*len(models), NewMemo(), paths[0])
	if err != nil || len(entries) != 1 || entries[0].Stale != "" {
		t.Fatalf("unrelated activation affected selected model: %+v, %v", entries, err)
	}
	if err := os.WriteFile(paths[0], []byte("replaced selected bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	entries, err = ServableWithMemo(t.Context(), store, 2*len(models), memo, paths[0])
	if err != nil || len(entries) != 0 {
		t.Fatalf("changed selected bytes accepted: %+v, %v", entries, err)
	}
}
