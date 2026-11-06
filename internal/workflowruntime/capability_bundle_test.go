package workflowruntime

import (
	"context"
	"io"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

type countingRepository struct {
	artifact.Repository
	contentReads map[artifact.ID]int
}

func (repository *countingRepository) OpenContent(ctx context.Context, id artifact.ID) (artifact.Descriptor, io.Reader, bool, error) {
	repository.contentReads[id]++
	return repository.Repository.OpenContent(ctx, id)
}

func TestCapabilityResourceOnDemand(t *testing.T) {
	ctx := t.Context()
	store := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	repository := &countingRepository{Repository: store, contentReads: make(map[artifact.ID]int)}
	model := testutil.ArtifactID(t, artifact.KindModel, "bundle-tool-model")
	contract := artifact.DocumentContract{Kind: artifact.KindFile, MediaType: "text/plain", Schema: "overgo/capability-resource/v1"}
	instruction := testcheck.Must(contract.ContentBytes([]byte("instruction")))(t)
	resource := testcheck.Must(contract.ContentBytes([]byte("resource")))(t)
	bundle := testcheck.Must(artifact.NewManifest(artifact.KindProfile, []artifact.Component{
		{Role: artifact.ComponentInstruction, Name: "system", Artifact: instruction.Descriptor.ID},
		{Role: artifact.ComponentResource, Name: "schema", Artifact: resource.Descriptor.ID},
	}))(t)
	testcheck.NoError(t, artifact.ValidateCapabilityBundle(bundle))
	if _, err := repository.Commit(ctx, artifact.Batch{
		Key: "bundle-tool/artifacts", Artifacts: []artifact.Descriptor{{ID: model}},
		Contents: []artifact.Content{instruction, resource}, Manifests: []artifact.Manifest{bundle},
	}); err != nil {
		t.Fatal(err)
	}
	module := recipe.ToolModule(testToolModule, recipe.PlacementHost)
	node := recipe.Node{ID: "execute", Module: module.ID, Placement: recipe.PlacementHost}
	definition := testcheck.Must(recipe.NewDefinitionWithDependencies(
		recipe.TaskInference,
		[]recipe.Dependency{
			{Role: recipe.DependencyModel, Artifact: model},
			{Role: recipe.DependencyCapabilityBundle, Artifact: bundle.ID},
		},
		[]recipe.Node{node}, nil,
		[]recipe.Input{{Name: recipe.ToolCallPort, Data: recipe.DataToolCall, Target: recipe.Endpoint{Node: node.ID, Port: recipe.ToolCallPort}}},
		[]recipe.Output{{Name: recipe.ToolResultPort, Data: recipe.DataToolResult, Source: recipe.Endpoint{Node: node.ID, Port: recipe.ToolResultPort}}},
	))(t)
	catalog := testcheck.Must(recipe.NewCatalog(module))(t)
	program := testcheck.Must(recipe.CompileProgram(definition, catalog))(t)
	executor := testcheck.Must(NewToolExecutor(repository, program, testToolRetention, typedToolAdapter(
		func(context.Context, toolArguments) (toolOutput, error) { return toolOutput{}, nil },
	)))(t)
	t.Cleanup(func() { executor.Close(); _ = repository.Close() })
	loaded := testcheck.Must(executor.LoadCapabilityComponent(ctx, bundle.ID, artifact.ComponentResource, "schema"))(t)
	if loaded.Descriptor.ID != resource.Descriptor.ID || repository.contentReads[resource.Descriptor.ID] != 1 ||
		repository.contentReads[instruction.Descriptor.ID] != 0 {
		t.Fatalf("loaded=%s reads=%v", loaded.Descriptor.ID, repository.contentReads)
	}
}
