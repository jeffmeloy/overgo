package modelintake

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestProjectionFactsPublicationBindsCompleteCandidate(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	inventory := func(kind artifact.Kind, name string) modelartifact.Inventory {
		t.Helper()
		weights := testutil.ArtifactID(t, artifact.KindTensorSet, name)
		manifest, err := artifact.NewManifest(kind, []artifact.Component{{Role: artifact.ComponentWeights, Name: "weights", Artifact: weights}})
		if err != nil {
			t.Fatal(err)
		}
		tensors, err := modelartifact.NewTensorInventoryDocument(manifest.ID, modelartifact.TensorFormatGGUF, []modelartifact.TensorFact{{Name: "weight", Shape: []uint64{1}, Storage: "f32", Bytes: 4}})
		if err != nil {
			t.Fatal(err)
		}
		return modelartifact.Inventory{Manifest: manifest, TensorInventory: tensors, Components: []artifact.Descriptor{{ID: weights}}}
	}
	model, media := inventory(artifact.KindModel, "language"), inventory(artifact.KindProjector, "media")
	makeCandidate := func(policy, source string) ProjectionCandidate {
		t.Helper()
		config, err := modelartifact.NewModelConfigDocument(model.Manifest.ID, nil, &modelartifact.GenerationEssentials{ImageAttention: policy}, []modelartifact.ConfigSource{{Name: "config.json", SHA256: strings.Repeat(source, 64)}})
		if err != nil {
			t.Fatal(err)
		}
		processor, err := (projector.MediaPreprocessProfile{Version: artifact.InitialDocumentVersion}).BindModelConfig(config)
		if err != nil {
			t.Fatal(err)
		}
		definition, err := modelrecipe.ProjectionDefinition(model.Manifest.ID, media.Manifest.ID, processor.ID, recipe.DataImage)
		if err != nil {
			t.Fatal(err)
		}
		return ProjectionCandidate{Model: model, Projector: media, Processor: &processor, Config: &config, Definition: definition, Media: []recipe.DataKind{recipe.DataImage}}
	}
	first, changed := makeCandidate("causal", "a"), makeCandidate("vision", "b")
	for _, candidate := range []ProjectionCandidate{first, changed} {
		if err := RegisterProjectionCandidate(t.Context(), store, candidate); err != nil {
			t.Fatal(err)
		}
	}
	before, sequence := store.Head()
	for _, candidate := range []ProjectionCandidate{first, changed} {
		if err := RegisterProjectionCandidate(t.Context(), store, candidate); err != nil {
			t.Fatal(err)
		}
		if _, err := recipe.RequireDefinition(t.Context(), store, candidate.Definition.ID); err != nil {
			t.Fatal(err)
		}
	}
	if after, n := store.Head(); after != before || n != sequence {
		t.Fatal("exact candidate replay mutated store")
	}
	// A location is a fact too. Relocation must not reuse a different batch's key.
	moved := changed
	moved.Model.Locations = []artifact.Location{{Artifact: model.Manifest.ID, Kind: artifact.LocationDirectory, Value: t.TempDir()}}
	if err := RegisterProjectionCandidate(t.Context(), store, moved); err != nil {
		t.Fatal(err)
	}
}
