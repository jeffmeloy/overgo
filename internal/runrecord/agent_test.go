package runrecord

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestAgentActivation(t *testing.T) {
	store, definition, decision := agentAuthorityFixture(t)
	defer store.Close()
	authority := AgentAuthority{Repository: store}
	active, err := authority.Activate(t.Context(), "agent/activate", definition, decision)
	if err != nil || active.Activation.State != AgentActive || active.Definition.ID != definition.ID {
		t.Fatalf("agent activation=(%+v, %v)", active, err)
	}
	resolved, found, err := authority.Resolve(t.Context(), definition.Name)
	if err != nil || !found || resolved.Activation.ID != active.Activation.ID {
		t.Fatalf("resolved agent=(%+v, %t, %v)", resolved, found, err)
	}
	if _, err := authority.RequireActive(t.Context(), definition.Name); err != nil {
		t.Fatal(err)
	}
}

func TestAgentPauseResumeAuthority(t *testing.T) {
	store, definition, decision := agentAuthorityFixture(t)
	defer store.Close()
	authority := AgentAuthority{Repository: store}
	active, err := authority.Activate(t.Context(), "agent/activate", definition, decision)
	if err != nil {
		t.Fatal(err)
	}
	pauseDecision := publishAgentAuthorityArtifact(t, store, artifact.KindEvidence, "agent-pause-decision")
	paused, err := authority.Pause(t.Context(), "agent/pause", definition.Name, pauseDecision)
	if err != nil || paused.Activation.State != AgentPaused || paused.Activation.Prior != active.Activation.ID {
		t.Fatalf("paused agent=(%+v, %v)", paused, err)
	}
	if _, err := authority.RequireActive(t.Context(), definition.Name); err == nil {
		t.Fatal("paused agent admitted")
	}
	resumeDecision := publishAgentAuthorityArtifact(t, store, artifact.KindEvidence, "agent-resume-decision")
	resumed, err := authority.Resume(t.Context(), "agent/resume", definition.Name, resumeDecision)
	if err != nil || resumed.Activation.State != AgentActive || resumed.Activation.Prior != paused.Activation.ID ||
		resumed.Definition.ID != definition.ID {
		t.Fatalf("resumed agent=(%+v, %v)", resumed, err)
	}
	if _, err := authority.Pause(t.Context(), "agent/stale-pause", definition.Name, resumeDecision); err == nil {
		t.Fatal("reused lifecycle authority accepted")
	}
}

func agentAuthorityFixture(t *testing.T) (*overgodb.Store, recipe.AgentDefinition, artifact.ID) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	definition, err := recipe.NewAgentDefinition(recipe.AgentDefinition{
		Name:              "durable-agent",
		Prompt:            publishAgentAuthorityArtifact(t, store, artifact.KindFile, "agent-prompt"),
		ModelRecipe:       publishAgentAuthorityArtifact(t, store, artifact.KindRecipe, "agent-recipe"),
		ToolManuals:       []artifact.ID{publishAgentAuthorityArtifact(t, store, artifact.KindRecipe, "agent-tool")},
		CapabilityBundles: []artifact.ID{publishAgentAuthorityArtifact(t, store, artifact.KindProfile, "agent-bundle")},
		Datasets:          []artifact.ID{publishAgentAuthorityArtifact(t, store, artifact.KindDataset, "agent-dataset")},
		Automations:       []artifact.ID{publishAgentAuthorityArtifact(t, store, artifact.KindRecipe, "agent-automation")},
		Policies:          []artifact.ID{publishAgentAuthorityArtifact(t, store, artifact.KindProfile, "agent-policy")},
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	decision := publishAgentAuthorityArtifact(t, store, artifact.KindEvidence, "agent-activate-decision")
	return store, definition, decision
}

func publishAgentAuthorityArtifact(t *testing.T, store artifact.Repository, kind artifact.Kind, name string) artifact.ID {
	t.Helper()
	id := testutil.ArtifactID(t, kind, name)
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "agent/authority/" + name, Artifacts: []artifact.Descriptor{{ID: id}},
	}); err != nil {
		t.Fatal(err)
	}
	return id
}
