package projector

import (
	"math"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestMediaPreprocessProfileCatalog(t *testing.T) {
	first, found, err := catalogMediaPreprocessProfile(qwen2VLProjectorType)
	if err != nil || !found {
		t.Fatalf("first profile = (%+v, %t, %v)", first, found, err)
	}
	second, found, err := catalogMediaPreprocessProfile(qwen3VLProjectorType)
	if err != nil || !found || second.ID != first.ID {
		t.Fatalf("second profile = (%+v, %t, %v), want %s", second, found, err, first.ID)
	}
	if _, err := first.Content(); err != nil {
		t.Fatal(err)
	}
	if _, found, err := catalogMediaPreprocessProfile("unbound"); err != nil || found {
		t.Fatalf("unbound profile = (%t, %v)", found, err)
	}
}

func TestImageAttentionRecipePinsConfig(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "attention-model")
	projectorID := testutil.ArtifactID(t, artifact.KindProjector, "attention-projector")
	base, found, err := catalogMediaPreprocessProfile(gemma4VisionTowerProjectorType)
	if err != nil || !found {
		t.Fatal(err)
	}
	declare := func(policy, digest string) modelartifact.ModelConfigDocument {
		t.Helper()
		config, err := modelartifact.NewModelConfigDocument(model, nil,
			&modelartifact.GenerationEssentials{ImageAttention: policy},
			[]modelartifact.ConfigSource{{Name: "config.json", SHA256: strings.Repeat(digest, 64)}})
		if err != nil {
			t.Fatal(err)
		}
		content, err := config.Content()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: config.ID.String(), Contents: []artifact.Content{content}}); err != nil {
			t.Fatal(err)
		}
		return config
	}
	causal := declare("causal", "a")
	profile, err := base.BindModelConfig(causal)
	if err != nil {
		t.Fatal(err)
	}
	definitionFor := func(profile MediaPreprocessProfile, modelID artifact.ID) recipe.Definition {
		t.Helper()
		content, err := profile.Content()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: profile.ID.String(), Contents: []artifact.Content{content}}); err != nil && err != artifact.ErrNoChange {
			t.Fatal(err)
		}
		definition, err := modelrecipe.ProjectionDefinition(modelID, projectorID, profile.ID, recipe.DataImage)
		if err != nil {
			t.Fatal(err)
		}
		return definition
	}
	definition := definitionFor(profile, model)
	// A later valid declaration changes future candidates, never this recipe.
	declare("vision", "b")
	resolved, err := ResolvePreprocessProfile(t.Context(), store, definition, &base)
	if err != nil || resolved == nil || resolved.ImageAttention != "causal" || resolved.ModelConfig != causal.ID {
		t.Fatalf("pinned replay changed: %+v %v", resolved, err)
	}
	for _, test := range []struct {
		name   string
		change func(*MediaPreprocessProfile)
	}{
		{"contradictory policy", func(p *MediaPreprocessProfile) { p.ImageAttention = "vision" }},
		{"missing source", func(p *MediaPreprocessProfile) {
			p.ModelConfig = testutil.ArtifactID(t, artifact.KindProfile, "missing-config")
		}},
		{"changed encoder policy", func(p *MediaPreprocessProfile) { p.AudioAttentionRopeFreqBase *= 2 }},
	} {
		t.Run(test.name, func(t *testing.T) {
			changed := profile
			changed.ID = artifact.ID{}
			test.change(&changed)
			changed, err := mediaPreprocessProfileCodec.New(changed)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := ResolvePreprocessProfile(t.Context(), store, definitionFor(changed, model), &base); err == nil {
				t.Fatal("contradictory binding accepted")
			}
		})
	}
	other := testutil.ArtifactID(t, artifact.KindModel, "other-model")
	if _, err := ResolvePreprocessProfile(t.Context(), store, definitionFor(profile, other), &base); err == nil {
		t.Fatal("another model's config accepted")
	}
}

func TestAudioProcessorProfileIdentity(t *testing.T) {
	profile, found, err := catalogMediaPreprocessProfile(gemma4VisionTowerProjectorType)
	if err != nil || !found {
		t.Fatalf("tower profile: found=%t err=%v", found, err)
	}
	if _, err := NewAudioProjectionProfile(profile.AudioAttentionRopeFreqBase); err != nil {
		t.Fatal(err)
	}
	changed := profile
	changed.AudioAttentionRopeFreqBase *= 2
	if _, err := changed.Content(); err == nil {
		t.Fatal("changed audio policy retained the accepted identity")
	}
	for _, invalid := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		changed.AudioAttentionRopeFreqBase = invalid
		if _, err := mediaPreprocessProfileCodec.New(changed); err == nil {
			t.Fatalf("invalid audio-only policy %v accepted", invalid)
		}
	}
	legacy, found, err := catalogMediaPreprocessProfile(qwen2VLProjectorType)
	if err != nil || !found {
		t.Fatal("legacy raster profile missing")
	}
	content, err := legacy.Content()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(content.Data), "audio_attention") {
		t.Fatal("audio extension rewrote legacy raster profile bytes")
	}
}
