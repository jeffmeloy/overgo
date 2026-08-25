package recipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestAgentDefinitionAuthority(t *testing.T) {
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	definition := agentDefinitionFixture(t)
	content, err := definition.ArtifactContent()
	if err != nil {
		t.Fatal(err)
	}
	artifacts := agentDefinitionDependencies(definition)
	if _, err := artifact.CommitBatch(context.Background(), store, artifact.Batch{
		Key: "agent/definition", Artifacts: artifacts, Contents: []artifact.Content{content}, Lineage: definition.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	loaded, err := RequireAgentDefinition(context.Background(), store, definition.ID)
	if err != nil || loaded.ID != definition.ID || loaded.Prompt != definition.Prompt ||
		len(loaded.ToolManuals) != 1 || len(loaded.CapabilityBundles) != 1 || len(loaded.Datasets) != 1 ||
		len(loaded.Automations) != 1 || len(loaded.Policies) != 1 {
		t.Fatalf("agent definition=(%+v, %v)", loaded, err)
	}
}

func TestAgentCapabilityRefusal(t *testing.T) {
	definition := agentDefinitionFixture(t)
	definition.Prompt = testutil.ArtifactID(t, artifact.KindProfile, "agent-wrong-prompt")
	if _, err := NewAgentDefinition(definition); err == nil {
		t.Fatal("non-file prompt accepted")
	}
	definition = agentDefinitionFixture(t)
	definition.ToolManuals = append(definition.ToolManuals, definition.ToolManuals[0])
	if _, err := NewAgentDefinition(definition); err == nil {
		t.Fatal("duplicate tool authority accepted")
	}
	definition = agentDefinitionFixture(t)
	definition.Policies = nil
	if _, err := NewAgentDefinition(definition); err == nil {
		t.Fatal("policy-free agent accepted")
	}
}

func agentDefinitionFixture(t *testing.T) AgentDefinition {
	t.Helper()
	value, err := NewAgentDefinition(AgentDefinition{
		Name:              "research-agent",
		Prompt:            testutil.ArtifactID(t, artifact.KindFile, "agent-prompt"),
		ModelRecipe:       testutil.ArtifactID(t, artifact.KindRecipe, "agent-model-recipe"),
		ToolManuals:       []artifact.ID{testutil.ArtifactID(t, artifact.KindRecipe, "agent-tool")},
		CapabilityBundles: []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "agent-bundle")},
		Datasets:          []artifact.ID{testutil.ArtifactID(t, artifact.KindDataset, "agent-dataset")},
		Automations:       []artifact.ID{testutil.ArtifactID(t, artifact.KindRecipe, "agent-automation")},
		Policies:          []artifact.ID{testutil.ArtifactID(t, artifact.KindProfile, "agent-policy")},
	})
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func agentDefinitionDependencies(value AgentDefinition) []artifact.Descriptor {
	lineage := value.Lineage()
	result := make([]artifact.Descriptor, 0, len(lineage))
	for _, edge := range lineage {
		result = append(result, artifact.Descriptor{ID: edge.Parent})
	}
	return result
}
