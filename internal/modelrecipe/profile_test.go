package modelrecipe

import (
	"context"
	"testing"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/recipe"
	"llamacpp2go/internal/repodb"
)

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

func TestProfileCandidateParityPromotion(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("parity-model"))
	definition, err := Inference(modelID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	profile, _ := model.LookupArchitecture("llama")
	document, err := NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishProfileCandidate(
		ctx, store, "fixture/profile/candidate", definition, document,
	); err != nil {
		t.Fatal(err)
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
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("drift-model"))
	definition, err := Inference(modelID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	profile, _ := model.LookupArchitecture("llama")
	profile.Attention = model.AttentionLFM2
	document, err := NewProfileDocument(profile)
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
	definition, err := Inference(modelID, recipe.PlacementHost)
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
