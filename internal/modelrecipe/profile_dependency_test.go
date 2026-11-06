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
	modelID := testcheck.Must(artifact.JSONID(artifact.KindModel, "fixture model"))(t)
	dependencies = append(dependencies, recipe.Dependency{Role: recipe.DependencyModel, Artifact: modelID})
	for _, role := range roles {
		profile := testcheck.Must(componentProfileFixtureCodec.New(componentProfileFixture{Name: string(role)}))(t)
		batch := testcheck.Must(componentProfileFixtureCodec.Batch("fixture/"+string(role), profile, nil, nil))(t)
		if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
			t.Fatal(err)
		}
		dependencies = append(dependencies, recipe.Dependency{Role: role, Artifact: profile.ID})
	}
	definition := testcheck.Must(inference(dependencies, recipe.PlacementHost, DecodeSessionRequest, recipe.ResidencyHostCache))(t)
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
	document := testcheck.Must(NewProfileDocument(profile))(t)
	content := testcheck.Must(profileContent(document))(t)
	batch := testcheck.Must(artifact.NewDocumentBatch("fixture/profile/unbound-dependency", []artifact.Content{content}, nil, nil))(t)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		t.Fatal(err)
	}
	modelID := testcheck.Must(artifact.JSONID(artifact.KindModel, "profile dependency model"))(t)
	definition := testcheck.Must(inference([]recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: modelID},
		{Role: recipe.DependencyProfile, Artifact: document.ID},
	}, recipe.PlacementHost, DecodeSessionRequest, recipe.ResidencyHostCache))(t)
	if _, err := resolveProfileDocumentDependency(ctx, store, definition, recipe.DependencyProfile); err == nil {
		t.Fatal("profile dependency without derivation lineage loaded")
	}
}
