package recipe

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestAutomationDefinition(t *testing.T) {
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, name)
	}
	firstDataset := id(artifact.KindDataset, "automation-dataset-a")
	secondDataset := id(artifact.KindDataset, "automation-dataset-b")
	firstBundle := id(artifact.KindProfile, "automation-bundle-a")
	secondBundle := id(artifact.KindProfile, "automation-bundle-b")
	definition, err := NewAutomationDefinition(AutomationDefinition{
		Name: "daily-report", Recipe: id(artifact.KindRecipe, "automation-recipe"),
		TriggerPolicy:     id(artifact.KindProfile, "automation-trigger"),
		DeliveryPolicy:    id(artifact.KindProfile, "automation-delivery"),
		Datasets:          []artifact.ID{secondDataset, firstDataset},
		CapabilityBundles: []artifact.ID{secondBundle, firstBundle},
	})
	if err != nil || definition.ID.Kind() != artifact.KindRecipe ||
		definition.Datasets[0].String() >= definition.Datasets[1].String() ||
		definition.CapabilityBundles[0].String() >= definition.CapabilityBundles[1].String() {
		t.Fatalf("automation definition = (%+v, %v)", definition, err)
	}
	if _, err := definition.ArtifactContent(); err != nil {
		t.Fatal(err)
	}
	if len(definition.Lineage()) != 3+len(definition.Datasets)+len(definition.CapabilityBundles) {
		t.Fatalf("automation lineage = %+v", definition.Lineage())
	}

	invalid := []AutomationDefinition{
		{Name: "Daily Report", Recipe: definition.Recipe, TriggerPolicy: definition.TriggerPolicy, DeliveryPolicy: definition.DeliveryPolicy},
		{Name: definition.Name, Recipe: id(artifact.KindModel, "wrong-recipe"), TriggerPolicy: definition.TriggerPolicy, DeliveryPolicy: definition.DeliveryPolicy},
		{Name: definition.Name, Recipe: definition.Recipe, TriggerPolicy: id(artifact.KindEvidence, "wrong-trigger"), DeliveryPolicy: definition.DeliveryPolicy},
		{Name: definition.Name, Recipe: definition.Recipe, TriggerPolicy: definition.TriggerPolicy, DeliveryPolicy: definition.DeliveryPolicy, Datasets: []artifact.ID{firstDataset, firstDataset}},
	}
	for index, candidate := range invalid {
		if _, err := NewAutomationDefinition(candidate); err == nil {
			t.Fatalf("invalid automation definition %d passed", index)
		}
	}
}
