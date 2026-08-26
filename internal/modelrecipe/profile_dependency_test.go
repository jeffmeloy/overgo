package modelrecipe

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

const (
	componentProfileMediaType = "application/vnd.overgo.test-component-profile+json"
	componentProfileSchema    = "overgo/test-component-profile/v1"
)

type componentProfileFixture struct {
	Name string      `json:"name"`
	ID   artifact.ID `json:"-"`
}

var componentProfileFixtureCodec = artifact.JSONDocumentCodec(
	"component profile fixture", artifact.KindProfile, componentProfileMediaType, componentProfileSchema,
	func(value *componentProfileFixture) error {
		if value.Name == "" {
			return errors.New("empty fixture name")
		}
		return nil
	},
	func(value componentProfileFixture) artifact.ID { return value.ID },
	func(value *componentProfileFixture, id artifact.ID) { value.ID = id }, nil,
)

func TestResolvedComponentProfilesUseTypedRecipeDependencies(t *testing.T) {
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	roles := []recipe.DependencyRole{
		recipe.DependencyProcessorProfile,
		recipe.DependencyFlowProfile,
		recipe.DependencyDerivationProfile,
	}
	dependencies := make([]recipe.Dependency, 0, len(roles)+1)
	modelID, err := artifact.JSONID(artifact.KindModel, "fixture model")
	if err != nil {
		t.Fatal(err)
	}
	dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyModel, Artifact: modelID})
	for _, role := range roles {
		profile, err := componentProfileFixtureCodec.New(componentProfileFixture{Name: string(role)})
		if err != nil {
			t.Fatal(err)
		}
		batch, err := componentProfileFixtureCodec.Batch("fixture/"+string(role), profile, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		dependencies = append(dependencies, recipe.Dependency{Role: role, Artifact: profile.ID})
	}
	definition, err := inference(dependencies, recipe.PlacementHost, DecodeSessionRequest, recipe.ResidencyHostCache)
	if err != nil {
		t.Fatal(err)
	}
	for _, role := range roles {
		profile, err := ResolveProfileDependency(ctx, store, definition, role, componentProfileFixtureCodec)
		if err != nil || profile.Name != string(role) {
			t.Fatalf("resolve %s = (%+v, %v)", role, profile, err)
		}
	}
	if _, err := ResolveProfileDependency(ctx, store, definition, recipe.DependencyMemory, componentProfileFixtureCodec); err == nil {
		t.Fatal("non-profile dependency role accepted")
	}
}
