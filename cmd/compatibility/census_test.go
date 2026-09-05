package main

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestCapabilityCensusContract(t *testing.T) {
	ctx := t.Context()
	storePath := t.TempDir()
	store, err := overgodb.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Error(err)
		}
	})
	register := func(name string, present, active bool) (artifact.ID, string) {
		t.Helper()
		payload := []byte(name)
		component := testutil.ArtifactBytesID(t, artifact.KindTensorSet, payload)
		manifest, err := artifact.NewManifest(artifact.KindModel, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: component}})
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), name+".bin")
		if present {
			if err := os.WriteFile(path, payload, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key:       "census-fixture/" + name,
			Artifacts: []artifact.Descriptor{{ID: component, Size: uint64(len(payload))}},
			Manifests: []artifact.Manifest{manifest},
			Locations: []artifact.LocationEvent{{Location: artifact.Location{Artifact: component, Kind: artifact.LocationFile, Value: path}, Action: artifact.LocationAdd}},
		}); err != nil {
			t.Fatal(err)
		}
		if active {
			definition, err := modelrecipe.CapabilityDefinition(recipe.TaskForecast, manifest.ID)
			if err != nil {
				t.Fatal(err)
			}
			if err := modelrecipetest.PublishActivation(ctx, store, "census-fixture/"+name, definition); err != nil {
				t.Fatal(err)
			}
		}
		return manifest.ID, path
	}
	presentID, presentPath := register("present", true, true)
	missingID, _ := register("missing", false, true)
	inactiveID, _ := register("inactive", true, false)
	staleID, _ := register("stale", true, true)
	aliases, err := store.Query(ctx, overgodb.Query{Kind: artifact.KindRecipe, Projection: overgodb.ProjectAliases, MaxResults: mediaCatalogLimit})
	if err != nil || aliases.Truncated {
		t.Fatalf("aliases = %+v, %v", aliases, err)
	}
	bogus := testutil.ArtifactID(t, artifact.KindRecipe, "broken active recipe")
	for _, alias := range aliases.Aliases {
		model, _, active := modelrecipe.ParseActiveAlias(alias.Name)
		if active && model == staleID {
			if _, err := store.Commit(ctx, artifact.Batch{Key: "census-fixture/stale-alias", Artifacts: []artifact.Descriptor{{ID: bogus}}, Aliases: []artifact.AliasBinding{{Name: alias.Name, Target: bogus, Previous: &alias.Target}}}); err != nil {
				t.Fatal(err)
			}
		}
	}
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "historical observation")
	testutil.PublishArtifact(t, store, evidence)
	verification, err := runrecord.NewModelVerification(presentID, "present", []runrecord.CapabilityClaim{{Capability: "forecast", Tier: runrecord.TierRealArtifactSmoke, Commit: strings.Repeat("a", 40), Evidence: []artifact.ID{evidence}}})
	if err != nil {
		t.Fatal(err)
	}
	verificationBatch, err := verification.Batch("census-fixture/verification")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, verificationBatch); err != nil {
		t.Fatal(err)
	}
	later, err := runrecord.NewModelVerification(presentID, "later", []runrecord.CapabilityClaim{{Capability: "forecast", Tier: runrecord.TierRealArtifactSmoke, Commit: strings.Repeat("c", 40), Evidence: []artifact.ID{evidence}}})
	if err != nil {
		t.Fatal(err)
	}
	laterContent, err := later.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: "census-fixture/later-declaration", Artifacts: []artifact.Descriptor{laterContent.Descriptor}}); err != nil {
		t.Fatal(err)
	}

	memo := discovery.NewMemo()
	value, err := buildCapabilityCensus(ctx, store, strings.Repeat("b", 40), mediaCatalogLimit, memo)
	if err != nil {
		t.Fatal(err)
	}
	if err := discovery.PublishMemo(ctx, store, memo); err != nil {
		t.Fatal(err)
	}
	if len(value.Models) != 4 || !slices.Equal(value.Verifications, []artifact.ID{verification.ID}) {
		t.Fatalf("incomplete census: %+v", value)
	}
	byID := map[artifact.ID]discovery.CatalogEntry{}
	for _, model := range value.Models {
		byID[model.Model] = model
	}
	if !byID[presentID].Present || byID[missingID].Present || len(byID[missingID].Capabilities) != 1 || len(byID[inactiveID].Capabilities) != 0 || !byID[inactiveID].Present {
		t.Fatalf("availability/inactive denominator lost: %+v", byID)
	}
	if stale := byID[staleID].Capabilities; len(stale) != 1 || stale[0].Recipe != bogus || stale[0].Stale == "" {
		t.Fatalf("stale recipe identity lost: %+v", stale)
	}
	activeOnly, truncated, err := discovery.CapabilityCatalog(ctx, store, mediaCatalogLimit, nil)
	if err != nil || truncated || len(activeOnly) != 3 {
		t.Fatalf("active-only catalog changed: %d %t %v", len(activeOnly), truncated, err)
	}
	if err := checkCapabilityCensus(ctx, store, value, mediaCatalogLimit); err != nil {
		t.Fatal(err)
	}
	laterBatch, err := later.Batch("census-fixture/later-publication")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, laterBatch); err != nil {
		t.Fatal(err)
	}
	if err := checkCapabilityCensus(ctx, store, value, mediaCatalogLimit); err != nil {
		t.Fatalf("later evidence invalidated the frozen before picture: %v", err)
	}
	batch, err := capabilityCensusBatch(value)
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Lineage) != 8 {
		t.Fatalf("lineage = %d, want four models, three recipes and one verification", len(batch.Lineage))
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	if err := runCapabilityCensus(storePath, false, value.ID.String(), &output); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output.String(), "registered identities=4 activated task/recipe pairs=3 unavailable identities=1 stale activations=1 historical verification records=1") || !strings.Contains(output.String(), strings.Repeat("a", 40)) || !strings.Contains(output.String(), "do not establish current task quality") {
		t.Fatalf("census audit/provenance = %s", output.String())
	}

	for _, test := range []struct {
		name   string
		change func(*capabilityCensus)
	}{
		{"omitted model", func(value *capabilityCensus) { value.Models = value.Models[1:] }},
		{"omitted verification", func(value *capabilityCensus) { value.Verifications = nil }},
		{"wrong store checkpoint", func(value *capabilityCensus) { value.StoreSequence++ }},
		{"omitted task", func(value *capabilityCensus) {
			for index := range value.Models {
				if value.Models[index].Model == presentID {
					value.Models[index].Capabilities = nil
				}
			}
		}},
		{"wrong recipe", func(value *capabilityCensus) {
			for index := range value.Models {
				if value.Models[index].Model == presentID {
					value.Models[index].Capabilities[0].Recipe = bogus
				}
			}
		}},
		{"duplicate model", func(value *capabilityCensus) { value.Models = append(value.Models, value.Models[0]) }},
		{"duplicate task", func(value *capabilityCensus) {
			for index := range value.Models {
				if value.Models[index].Model == presentID {
					value.Models[index].Capabilities = append(value.Models[index].Capabilities, value.Models[index].Capabilities[0])
				}
			}
		}},
		{"wrong verification type", func(value *capabilityCensus) { value.Verifications = []artifact.ID{value.ID} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			changed := capabilityCensusCodec.Clone(value)
			test.change(&changed)
			changed, err := capabilityCensusCodec.New(changed)
			if err != nil {
				return
			}
			if err := checkCapabilityCensus(ctx, store, changed, mediaCatalogLimit); err == nil {
				t.Fatal("invalid census accepted")
			}
		})
	}
	if _, err := buildCapabilityCensus(ctx, store, value.ProducerCommit, 1, nil); err == nil {
		t.Fatal("truncated census published")
	}
	if err := checkCapabilityCensus(ctx, store, value, 1); err == nil {
		t.Fatal("truncated live denominator accepted")
	}
	if err := os.Remove(presentPath); err != nil {
		t.Fatal(err)
	}
	if err := checkCapabilityCensus(ctx, store, value, mediaCatalogLimit); err == nil {
		t.Fatal("availability drift accepted")
	}
	if err := os.WriteFile(presentPath, []byte("present"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := checkCapabilityCensus(ctx, store, value, mediaCatalogLimit); err != nil {
		t.Fatal(err)
	}
	register("added", true, false)
	if err := checkCapabilityCensus(ctx, store, value, mediaCatalogLimit); err == nil {
		t.Fatal("new registered identity accepted without disposition")
	}
}
