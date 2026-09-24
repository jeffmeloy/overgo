package server

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestActiveAgentNamesReadOnlyTheActivationRoot lists the active agent from
// the activation documents its root binds, and nothing else the store binds:
// a neighbouring alias under a longer root and an alias of another document
// under the agent root stay out. The manifest asks on every page load.
func TestActiveAgentNamesReadOnlyTheActivationRoot(t *testing.T) {
	t.Parallel()
	fixture := newAgentWorkspaceFixture(t, nil, nil, nil)
	defer fixture.store.Close()
	activateDefinitionFromAPI(t, fixture.handler, publishAgentFromAPI(t, fixture, nil, nil), "/agents/activate")
	stray := testutil.ArtifactID(t, artifact.KindEvidence, "not-an-activation")
	if _, err := fixture.store.Commit(t.Context(), artifact.Batch{
		Key: "agent-names/stray", Artifacts: []artifact.Descriptor{{ID: stray}},
		Aliases: []artifact.AliasBinding{
			{Name: runrecord.AgentActiveAliasRoot + "stray", Target: stray},
			{Name: "agent/activeness/neighbour", Target: stray},
		},
	}); err != nil {
		t.Fatal(err)
	}
	names, err := fixture.handler.activeAgentNames(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(names, []string{"research-agent"}) {
		t.Fatalf("active agent names = %v, want [research-agent]", names)
	}
}
