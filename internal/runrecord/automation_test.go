package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

func TestAutomationActivation(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	id := func(kind artifact.Kind, name string) artifact.ID {
		data := []byte(name)
		identified, identifyErr := artifact.IdentifyBytes(kind, data)
		if identifyErr != nil {
			t.Fatal(identifyErr)
		}
		content := artifact.Content{Descriptor: artifact.Descriptor{
			ID: identified, Size: uint64(len(data)), MediaType: "application/octet-stream",
		}, Data: data}
		if _, commitErr := artifact.CommitBatch(t.Context(), store, artifact.Batch{
			Key: "automation/test-authority/" + identified.String(), Contents: []artifact.Content{content},
		}); commitErr != nil {
			t.Fatal(commitErr)
		}
		return identified
	}
	definition, err := recipe.NewAutomationDefinition(recipe.AutomationDefinition{
		Name: "daily-report", Recipe: id(artifact.KindRecipe, "active-automation-recipe"),
		TriggerPolicy:  id(artifact.KindProfile, "active-automation-trigger"),
		DeliveryPolicy: id(artifact.KindProfile, "active-automation-delivery"),
	})
	if err != nil {
		t.Fatal(err)
	}
	authority := id(artifact.KindEvidence, "active-automation-authority")
	ctx := t.Context()
	automations := AutomationAuthority{Repository: store}
	firstActive, err := automations.Activate(ctx, "automation/activate/first", definition, authority)
	if err != nil {
		t.Fatal(err)
	}
	first := firstActive.Activation
	active, found, err := automations.Resolve(ctx, definition.Name)
	if err != nil || !found || active.Definition.ID != definition.ID || active.Activation.ID != first.ID {
		t.Fatalf("active automation = (%+v, %v, %v)", active, found, err)
	}

	stale, err := newAutomationActivation(definition, authority, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := publishAutomationActivation(ctx, store, "automation/activate/stale", definition, stale); err == nil {
		t.Fatal("stale automation activation passed")
	}
	secondAuthority := id(artifact.KindEvidence, "active-automation-authority-2")
	secondActive, err := automations.Activate(ctx, "automation/activate/second", definition, secondAuthority)
	if err != nil {
		t.Fatal(err)
	}
	second := secondActive.Activation
	active, found, err = automations.Resolve(ctx, definition.Name)
	if err != nil || !found || active.Activation.ID != second.ID || active.Activation.Prior != first.ID {
		t.Fatalf("advanced automation = (%+v, %v, %v)", active, found, err)
	}
	if _, found, err := automations.Resolve(ctx, "missing"); err != nil || found {
		t.Fatalf("missing automation = (%v, %v)", found, err)
	}
}
