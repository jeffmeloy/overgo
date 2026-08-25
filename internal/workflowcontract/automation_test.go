package workflowcontract

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowrecipe"
)

type automationFixture struct {
	compiler   AutomationCompiler
	definition recipe.AutomationDefinition
	store      artifact.Repository
}

func TestCompileAutomationExecutionPlan(t *testing.T) {
	fixture := prepareAutomationFixture(t, false)
	defer fixture.store.Close()

	first, err := fixture.compiler.Compile(context.Background(), fixture.definition.Name)
	if err != nil {
		t.Fatal(err)
	}
	second, err := fixture.compiler.Compile(context.Background(), fixture.definition.Name)
	if err != nil {
		t.Fatal(err)
	}
	if first.ID != second.ID || first.CacheIdentity != second.CacheIdentity ||
		first.Definition != fixture.definition.ID || first.Recipe != fixture.definition.Recipe ||
		first.Resources.Recipe != first.Recipe || len(first.Datasets) != 1 || len(first.CapabilityBundles) != 1 {
		t.Fatalf("compiled automation plan differs: first=%+v second=%+v", first, second)
	}
	if _, err := first.Content(); err != nil || first.ValidateIdentity() != nil || len(first.Lineage()) == 0 {
		t.Fatalf("compiled automation plan contract failed: %v", err)
	}
}

func TestAutomationExecutionPlanRefuses(t *testing.T) {
	fixture := prepareAutomationFixture(t, true)
	defer fixture.store.Close()
	if _, err := fixture.compiler.Compile(context.Background(), fixture.definition.Name); err == nil {
		t.Fatal("automation with recipe dependency mismatch compiled")
	}
	if _, err := fixture.compiler.Compile(context.Background(), "absent"); err == nil {
		t.Fatal("automation without active authority compiled")
	}
}

func prepareAutomationFixture(t *testing.T, mismatch bool) automationFixture {
	t.Helper()
	ctx := context.Background()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	model := commitAutomationBlob(t, store, artifact.KindModel, "automation-model")
	dataset := commitAutomationBlob(t, store, artifact.KindDataset, "automation-dataset")
	bundle := commitAutomationBlob(t, store, artifact.KindProfile, "automation-capability-bundle")
	authority := commitAutomationBlob(t, store, artifact.KindEvidence, "automation-authority")
	dependencies := []recipe.Dependency{
		{Role: recipe.DependencyModel, Artifact: model},
		{Role: recipe.DependencyDataset, Artifact: dataset},
		{Role: recipe.DependencyCapabilityBundle, Artifact: bundle},
	}
	definition, err := recipe.NewDefinitionWithDependencies(
		recipe.TaskGeneration, dependencies,
		[]recipe.Node{{
			ID: "generate", Module: workflowrecipe.ModuleGenerate,
			Placement: recipe.PlacementHost, Session: recipe.SessionCapacity,
		}}, nil,
		[]recipe.Input{{Name: "tokens", Data: recipe.DataTokens, Target: recipe.Endpoint{Node: "generate", Port: "tokens"}}},
		[]recipe.Output{{Name: "tokens", Data: recipe.DataTokens, Source: recipe.Endpoint{Node: "generate", Port: "tokens"}}},
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	trigger, err := (recipe.AutomationTriggerPolicy{Kind: recipe.AutomationTriggerManual}).Identify()
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	delivery, err := (recipe.AutomationDeliveryPolicy{Kind: recipe.AutomationDeliveryArtifact}).Identify()
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	recipeContent, _ := definition.ArtifactContent()
	triggerContent, _ := trigger.ArtifactContent()
	deliveryContent, _ := delivery.ArtifactContent()
	batch, err := artifact.NewDocumentBatch(
		"automation/fixture", []artifact.Content{recipeContent, triggerContent, deliveryContent}, nil, nil,
	)
	if err == nil {
		_, err = artifact.CommitBatch(ctx, store, batch)
	}
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if err := modelrecipe.EnsureRuntimePolicy(ctx, store, definition); err != nil {
		store.Close()
		t.Fatal(err)
	}
	automationDatasets := []artifact.ID{dataset}
	if mismatch {
		automationDatasets = nil
	}
	automation, err := recipe.NewAutomationDefinition(recipe.AutomationDefinition{
		Name: "daily-report", Recipe: definition.ID,
		TriggerPolicy: trigger.ID, DeliveryPolicy: delivery.ID,
		Datasets: automationDatasets, CapabilityBundles: []artifact.ID{bundle},
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if _, err := (runrecord.AutomationAuthority{Repository: store}).Activate(
		ctx, "automation/fixture/activate", automation, authority,
	); err != nil {
		store.Close()
		t.Fatal(err)
	}
	return automationFixture{
		compiler:   AutomationCompiler{Repository: store, Catalog: workflowrecipe.Catalog()},
		definition: automation, store: store,
	}
}

func commitAutomationBlob(t *testing.T, store artifact.Repository, kind artifact.Kind, label string) artifact.ID {
	t.Helper()
	data := []byte(label)
	id, err := artifact.IdentifyBytes(kind, data)
	if err != nil {
		t.Fatal(err)
	}
	content := artifact.Content{Descriptor: artifact.Descriptor{
		ID: id, Size: uint64(len(data)), MediaType: "application/octet-stream",
	}, Data: data}
	if _, err := artifact.CommitBatch(context.Background(), store, artifact.Batch{
		Key: "automation/blob/" + label, Contents: []artifact.Content{content},
	}); err != nil {
		t.Fatal(err)
	}
	return id
}
