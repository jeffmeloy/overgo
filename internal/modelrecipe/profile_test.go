package modelrecipe

import (
	"testing"

	"llamacpp2go/internal/artifact"
	"llamacpp2go/internal/model"
	"llamacpp2go/internal/recipe"
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
