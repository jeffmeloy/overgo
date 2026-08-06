package modelrecipe

import (
	"context"
	"crypto/sha256"
	"fmt"
	"testing"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/recipe"
	"llamacpp2go/internal/repodb"
)

const profileCatalogSemanticDigest = "185cfaf9ee287f95508b3557f4751770cc3fcfab75b91547474a354a0143136a"

func TestRegisteredProfileDocumentsRoundTrip(t *testing.T) {
	documents, err := SeedProfileDocuments()
	if err != nil {
		t.Fatal(err)
	}
	if len(documents) != len(model.SupportedArchitectures()) {
		t.Fatalf("documents = %d", len(documents))
	}
	for _, document := range documents {
		content, err := document.Content()
		if err != nil {
			t.Fatalf("%s: %v", document.Architecture, err)
		}
		parsed, err := ParseProfileDocument(content)
		if err != nil {
			t.Fatalf("%s: %v", document.Architecture, err)
		}
		if parsed.ID != document.ID || parsed.Policy != document.Policy {
			t.Fatalf("%s: profile round trip drifted", document.Architecture)
		}
	}
}

func TestRegisteredProfileSemanticDigest(t *testing.T) {
	documents, err := SeedProfileDocuments()
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.New()
	for _, document := range documents {
		fmt.Fprintln(digest, document.ID)
	}
	if got := fmt.Sprintf("%x", digest.Sum(nil)); got != profileCatalogSemanticDigest {
		t.Fatalf("profile catalog digest = %s", got)
	}
}

func TestPublishProfilesSeedsRegisteredAliases(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	_, documents, err := PublishProfiles(ctx, store, "fixture/profiles")
	if err != nil {
		t.Fatal(err)
	}
	document, ok, err := RegisteredProfile(ctx, store, "llama")
	if err != nil || !ok || document.Architecture != "llama" {
		t.Fatalf("registered profile = (%s, %v, %v)", document.Architecture, ok, err)
	}
	if len(documents) != len(model.SupportedArchitectures()) {
		t.Fatalf("documents = %d", len(documents))
	}
}

func TestPublishProfileCatalogReplacesBootstrapAuthority(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	bootstrap, _ := model.LookupArchitecture("llama")
	bootstrapDocument, err := NewProfileDocument(bootstrap)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishProfileCatalog(ctx, store, "fixture/catalog/bootstrap", []ProfileDocument{bootstrapDocument}); err != nil {
		t.Fatal(err)
	}
	candidate := bootstrap
	candidate.DeciSparse = true
	candidateDocument, err := NewProfileDocument(candidate)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishProfileCatalog(ctx, store, "fixture/catalog/candidate", []ProfileDocument{candidateDocument}); err != nil {
		t.Fatal(err)
	}
	stored, ok, err := RegisteredProfile(ctx, store, "llama")
	if err != nil || !ok || stored.ID != candidateDocument.ID || !stored.Policy.DeciSparse {
		t.Fatalf("stored profile = (%+v, %v, %v)", stored, ok, err)
	}
	registered, _ := model.LookupArchitecture("llama")
	if registered != bootstrap {
		t.Fatal("RepoDB profile publication mutated bootstrap fallback")
	}
}

func TestPublishProfileCatalogRejectsDuplicateArchitecture(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	profile, _ := model.LookupArchitecture("llama")
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := PublishProfileCatalog(
		ctx, store, "fixture/catalog/duplicate", []ProfileDocument{document, document},
	); err == nil {
		t.Fatal("duplicate profile architecture accepted")
	}
}

func TestProfileCandidateParityPromotion(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := commitFixtureModel(t, ctx, store, "fixture/profile/model", "parity-model")
	profile, _ := model.LookupArchitecture("llama")
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := InferenceWithProfile(modelID, document.ID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishProfileCandidate(
		ctx, store, "fixture/profile/candidate", definition, document,
	); err != nil {
		t.Fatal(err)
	}
	if _, bound, err := store.ResolveAlias(ctx, legacyRecipeProfileAlias(definition.ID)); err != nil || bound {
		t.Fatalf("legacy profile alias = (%v, %v)", bound, err)
	}
	parents, err := store.Parents(ctx, definition.ID)
	if err != nil || len(parents) != len(definition.Dependencies) {
		t.Fatalf("dependency lineage = (%+v, %v)", parents, err)
	}
	spec := model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", BlockCount: 1}}
	_, _, evidence, err := ValidateProfileCandidate(
		ctx, store, "fixture/profile/validated", definition, spec, model.Weights{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if evidence.Profile != document.ID || evidence.Recipe != definition.ID {
		t.Fatalf("evidence = %+v", evidence)
	}
	if _, _, err := ActivateProfileCandidate(
		ctx, store, "fixture/profile/active", definition, nil,
	); err != nil {
		t.Fatal(err)
	}
	plan, ok, err := CompileActive(ctx, store, modelID, spec, model.Weights{})
	if err != nil || !ok || plan.Model.Profile != profile {
		t.Fatalf("active profile plan = (%+v, %v, %v)", plan.Model, ok, err)
	}
	activeProfile, ok, err := ActiveProfile(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || activeProfile.ID != document.ID {
		t.Fatalf("active profile = (%s, %v, %v)", activeProfile.ID, ok, err)
	}
}

func TestProfileCandidateRejectsParityDrift(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID := commitFixtureModel(t, ctx, store, "fixture/drift/model", "drift-model")
	profile, _ := model.LookupArchitecture("llama")
	profile.Attention = model.AttentionLFM2
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := InferenceWithProfile(modelID, document.ID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishProfileCandidate(
		ctx, store, "fixture/drift/candidate", definition, document,
	); err != nil {
		t.Fatal(err)
	}
	_, _, _, err = ValidateProfileCandidate(
		ctx, store, "fixture/drift/validated", definition,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", BlockCount: 1}}, model.Weights{},
	)
	if err == nil {
		t.Fatal("drifted profile passed parity validation")
	}
}

func TestCompileWithProfileMatchesRegistry(t *testing.T) {
	modelID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("profile-model"))
	if err != nil {
		t.Fatal(err)
	}
	profile, ok := model.LookupArchitecture("llama")
	if !ok {
		t.Fatal("llama profile is absent")
	}
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	definition, err := InferenceWithProfile(modelID, document.ID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := VerifyProfileParity(
		definition, document,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama", BlockCount: 1}}, model.Weights{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Model.Profile.Name != "llama" || len(plan.Model.Layers) != 1 {
		t.Fatalf("plan = %+v", plan.Model)
	}
}

func TestProfileDocumentRejectsNonCanonicalContent(t *testing.T) {
	profile, _ := model.LookupArchitecture("llama")
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	content, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	content = append(content[:len(content)-1], []byte(",\"extra\":true}")...)
	if _, err := ParseProfileDocument(content); err == nil {
		t.Fatal("unknown profile field accepted")
	}
}

func commitFixtureModel(
	t *testing.T,
	ctx context.Context,
	store artifact.Repository,
	key string,
	payload string,
) artifact.ID {
	t.Helper()
	data := []byte(payload)
	id, err := artifact.IdentifyBytes(artifact.KindModel, data)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{Key: key, Contents: []artifact.Content{{
		Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data))}, Data: data,
	}}}); err != nil {
		t.Fatal(err)
	}
	return id
}
