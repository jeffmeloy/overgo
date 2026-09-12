package server

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// Seed a disposable journey store with exact local activations and file references.
// Canonical history and model weights stay in place; browser writes stay private.
func prepareBrowserJourneyStore(ctx context.Context, source, destination string) error {
	store, err := overgodb.OpenReadOnly(source)
	if err != nil {
		return err
	}
	defer store.Close()
	entries, truncated, err := discovery.CapabilityCatalog(ctx, store, laneCatalogLimit, discovery.LoadMemo(ctx, store))
	if err != nil || truncated {
		return errors.Join(errors.New("browser fixture: incomplete source catalog"), err)
	}
	var pending []artifact.ID
	for _, entry := range entries {
		if !entry.Present || strings.HasPrefix(entry.Location, "remote://") {
			continue
		}
		if _, err := os.Stat(entry.Location); err != nil {
			return fmt.Errorf("browser fixture: local weights %q: %w", entry.Location, err)
		}
		pending = append(pending, entry.Model)
		for _, capability := range entry.Capabilities {
			if capability.Stale != "" {
				continue
			}
			active, found, err := modelrecipe.ActiveRecord(ctx, store, entry.Model, capability.Task)
			if err != nil || !found {
				return errors.Join(fmt.Errorf("browser fixture: activation %s/%s is unavailable", entry.Model, capability.Task), err)
			}
			pending = append(pending, active.Definition.ID, active.Event.ID)
		}
	}
	if len(pending) == 0 {
		return errors.New("browser fixture: no local activated models")
	}
	if memo, found, err := artifact.ResolveAlias(ctx, store, discovery.IdentityEvidenceAlias); err != nil {
		return err
	} else if found {
		pending = append(pending, memo)
	}
	batch := artifact.Batch{Key: "browser-journey/inputs", ExpectedHead: new(artifact.CommitID)}
	seen := map[artifact.ID]bool{}
	for len(pending) > 0 {
		id := pending[len(pending)-1]
		pending = pending[:len(pending)-1]
		if seen[id] {
			continue
		}
		seen[id] = true
		descriptor, found, err := store.Artifact(ctx, id)
		if err != nil || !found {
			return errors.Join(fmt.Errorf("browser fixture: missing dependency %s", id), err)
		}
		batch.Artifacts = append(batch.Artifacts, descriptor)
		if content, found, err := artifact.ReadContent(ctx, store, id); err != nil {
			return err
		} else if found {
			batch.Contents = append(batch.Contents, content)
		}
		if manifest, found, err := store.Manifest(ctx, id); err != nil {
			return err
		} else if found {
			batch.Manifests = append(batch.Manifests, manifest)
			for _, component := range manifest.Components {
				pending = append(pending, component.Artifact)
			}
		}
		parents, err := store.Parents(ctx, id)
		if err != nil {
			return err
		}
		batch.Lineage = append(batch.Lineage, parents...)
		for _, parent := range parents {
			pending = append(pending, parent.Parent)
		}
		locations, err := store.Locations(ctx, id)
		if err != nil {
			return err
		}
		for _, location := range locations {
			batch.Locations = append(batch.Locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
		}
	}
	// Alias counts are independent of the model catalog size. Walk them once
	// after selecting the closure instead of truncating each artifact's aliases.
	if err := store.VisitAliases(ctx, "", func(alias overgodb.AliasView) error {
		if seen[alias.Target] {
			batch.Aliases = append(batch.Aliases, artifact.AliasBinding{Name: alias.Name, Target: alias.Target})
		}
		return nil
	}); err != nil {
		return err
	}
	private, err := overgodb.OpenContext(ctx, destination)
	if err != nil {
		return err
	}
	defer private.Close()
	if head, _ := private.Head(); head.Valid() {
		return errors.New("browser fixture: destination already contains history")
	}
	_, err = artifact.CommitBatch(ctx, private, batch)
	return err
}

func TestBrowserJourneyStoreIsolation(t *testing.T) {
	ctx := t.Context()
	source := t.TempDir()
	store, err := overgodb.Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	payload := []byte("browser fixture shared weights")
	weights := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
	manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: weights}})
	if err != nil {
		t.Fatal(err)
	}
	location := filepath.Join(t.TempDir(), "weights.bin")
	if err := os.WriteFile(location, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "browser-fixture/model", Artifacts: []artifact.Descriptor{{ID: weights, Size: uint64(len(payload))}},
		Manifests: []artifact.Manifest{manifest},
		Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: weights, Kind: artifact.LocationFile, Value: location}, Action: artifact.LocationAdd}},
	}); err != nil {
		t.Fatal(err)
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskForecast, manifest.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipetest.PublishActivation(ctx, store, "browser-fixture/active", definition); err != nil {
		t.Fatal(err)
	}
	unrelated := testutil.ArtifactID(t, artifact.KindEvidence, "unrelated historical result")
	testutil.PublishArtifact(t, store, unrelated)
	aliasBatch := artifact.Batch{Key: "browser-fixture/many-aliases"}
	for index := range laneCatalogLimit + 1 {
		aliasBatch.Aliases = append(aliasBatch.Aliases, artifact.AliasBinding{Name: fmt.Sprintf("browser-fixture/alias/%d", index), Target: manifest.ID})
	}
	if _, err := artifact.CommitBatch(ctx, store, aliasBatch); err != nil {
		t.Fatal(err)
	}
	memo := discovery.NewMemo()
	if entries, truncated, err := discovery.CapabilityCatalog(ctx, store, laneCatalogLimit, memo); err != nil || truncated || len(entries) != 1 || !entries[0].Present {
		t.Fatalf("source fixture: %v %v", entries, err)
	}
	if err := discovery.PublishMemo(ctx, store, memo); err != nil {
		t.Fatal(err)
	}
	before, sequence := store.Head()
	memoID, found, err := artifact.ResolveAlias(ctx, store, discovery.IdentityEvidenceAlias)
	if err != nil || !found {
		t.Fatalf("source identity memo: %v", err)
	}
	destination := filepath.Join(t.TempDir(), "store")
	if err := prepareBrowserJourneyStore(ctx, source, destination); err != nil {
		t.Fatal(err)
	}
	private, err := overgodb.Open(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer private.Close()
	aliases := 0
	if err := private.VisitAliases(ctx, "browser-fixture/alias/", func(alias overgodb.AliasView) error {
		if alias.Target != manifest.ID {
			return errors.New("copied alias changed target")
		}
		aliases++
		return nil
	}); err != nil || aliases != laneCatalogLimit+1 {
		t.Fatalf("aliases truncated or changed: %d %v", aliases, err)
	}
	active, found, err := modelrecipe.ActiveRecord(ctx, private, manifest.ID, recipe.TaskForecast)
	if err != nil || !found || active.Definition.ID != definition.ID {
		t.Fatalf("exact activation did not survive: %+v %v", active, err)
	}
	if _, found, err := private.Artifact(ctx, unrelated); err != nil || found {
		t.Fatalf("unrelated history copied: %t %v", found, err)
	}
	if copied, found, err := artifact.ResolveAlias(ctx, private, discovery.IdentityEvidenceAlias); err != nil || !found || copied != memoID {
		t.Fatalf("exact digest memo was not reused: %s %v", copied, err)
	}
	locations, err := private.Locations(ctx, weights)
	if err != nil || len(locations) != 1 || locations[0].Value != location {
		t.Fatalf("weights were relocated: %v %v", locations, err)
	}
	if stored, err := private.HasContent(ctx, weights); err != nil || stored {
		t.Fatalf("weights were copied into store content: %t %v", stored, err)
	}
	mutation := testutil.ArtifactID(t, artifact.KindModel, "private browser declaration")
	testutil.PublishArtifact(t, private, mutation)
	if err := private.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.OpenReadOnly(destination)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	if _, found, err := reopened.Artifact(ctx, mutation); err != nil || !found {
		t.Fatalf("private result did not survive reopen: %t %v", found, err)
	}
	if err := prepareBrowserJourneyStore(ctx, source, destination); err == nil {
		t.Fatal("populated destination accepted")
	}
	if err := prepareBrowserJourneyStore(ctx, source, source); err == nil {
		t.Fatal("source accepted as destination")
	}
	freshSource, err := overgodb.OpenReadOnly(source)
	if err != nil {
		t.Fatal(err)
	}
	defer freshSource.Close()
	if after, afterSequence := freshSource.Head(); after != before || afterSequence != sequence {
		t.Fatal("browser preparation or mutation changed source history")
	}
	if _, found, err := freshSource.Artifact(ctx, mutation); err != nil || found {
		t.Fatalf("browser fixture leaked into source: %t %v", found, err)
	}
}
