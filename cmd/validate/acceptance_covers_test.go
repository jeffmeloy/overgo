package main

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

// TestAcceptanceCoversARecipeReplacedForItsSession holds what was accepted
// under a recipe to the recipe active now: the same recipe covers it, so does
// one that is the same computation under another session policy, and one
// whose placement moved does not, nor one that is not in the store.
func TestAcceptanceCoversARecipeReplacedForItsSession(t *testing.T) {
	t.Parallel()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := testutil.ArtifactID(t, artifact.KindModel, "acceptance-covers")
	testutil.PublishArtifact(t, store, model)
	accepted, err := modelrecipe.CapabilityDefinition(recipe.TaskForecast, model)
	if err != nil {
		t.Fatal(err)
	}
	variant := func(change func(*recipe.Node)) recipe.Definition {
		t.Helper()
		nodes := slices.Clone(accepted.Nodes)
		change(&nodes[0])
		value, err := recipe.NewDefinitionWithDependencies(accepted.Task, accepted.Dependencies, nodes, accepted.Edges, accepted.Inputs, accepted.Outputs)
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	session := variant(func(node *recipe.Node) { node.Session = recipe.SessionRequest })
	moved := variant(func(node *recipe.Node) {
		node.Placement = recipe.PlacementHost
		if accepted.Nodes[0].Placement == recipe.PlacementHost {
			node.Placement = recipe.PlacementDevice
		}
	})
	for key, definition := range map[string]recipe.Definition{"accepted": accepted, "session": session, "moved": moved} {
		if _, _, err := modelrecipe.PublishCandidate(ctx, store, "acceptance-covers/"+key, definition); err != nil {
			t.Fatal(err)
		}
	}
	for name, test := range map[string]struct {
		active  artifact.ID
		covered bool
	}{
		"the same recipe":          {accepted.ID, true},
		"another session policy":   {session.ID, true},
		"another placement":        {moved.ID, false},
		"a recipe the store lacks": {testutil.ArtifactID(t, artifact.KindRecipe, "acceptance-covers-absent"), false},
	} {
		covered, err := acceptanceCovers(ctx, store, accepted.ID, test.active)
		if test.covered && (err != nil || !covered) || !test.covered && covered {
			t.Fatalf("%s: covered %v, error %v", name, covered, err)
		}
	}
}
