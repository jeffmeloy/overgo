package modelrecipe

import (
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
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
	func(value *componentProfileFixture, id artifact.ID) { value.ID = id },
)

func TestResolvedComponentProfilesUseTypedRecipeDependencies(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())

	roles := []recipe.DependencyRole{
		recipe.DependencyProcessorProfile,
		recipe.DependencyFlowProfile,
		recipe.DependencyDerivationProfile,
	}
	dependencies := make([]recipe.Dependency, 0, len(roles)+1)
	modelID, err := artifact.JSONID(artifact.KindModel, "fixture model")
	testcheck.NoError(t, err)
	dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyModel, Artifact: modelID})
	for _, role := range roles {
		profile, err := componentProfileFixtureCodec.New(componentProfileFixture{Name: string(role)})
		testcheck.NoError(t, err)
		batch, err := componentProfileFixtureCodec.Batch("fixture/"+string(role), profile, nil, nil)
		testcheck.NoError(t, err)
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		dependencies = append(dependencies, recipe.Dependency{Role: role, Artifact: profile.ID})
	}
	definition, err := inference(dependencies, recipe.PlacementHost, DecodeSessionRequest, recipe.ResidencyHostCache)
	testcheck.NoError(t, err)
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

func TestResolvedProfileDocumentDependencyRequiresStoredDerivationLineage(t *testing.T) {
	ctx := t.Context()
	store := overgodbtest.Open(t, t.TempDir())
	profile, _ := model.LookupArchitecture("llama")
	document, err := NewProfileDocument(profile)
	testcheck.NoError(t, err)
	content, err := profileContent(document)
	testcheck.NoError(t, err)
	batch, err := artifact.NewDocumentBatch("fixture/profile/unbound-dependency", []artifact.Content{content}, nil, nil)
	testcheck.NoError(t, err)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	modelID, err := artifact.JSONID(artifact.KindModel, "profile dependency model")
	testcheck.NoError(t, err)
	definition, err := inference([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: document.ID},
	}, recipe.PlacementHost, DecodeSessionRequest, recipe.ResidencyHostCache)
	testcheck.NoError(t, err)
	if _, err := resolveProfileDocumentDependency(ctx, store, definition, recipe.DependencyProfile); err == nil {
		t.Fatal("profile dependency without derivation lineage loaded")
	}
}
