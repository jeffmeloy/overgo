package runrecord

import (
	"testing"

	"overgo/internal/recipe"
)

// TestRecipeDeviceFollowsPlacement holds a recipe's recorded device to where
// its nodes are placed: wholly on the host is the host, and any node on the
// device, or split between host and device, is the CUDA device.
func TestRecipeDeviceFollowsPlacement(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		placements      []recipe.Placement
		device, backend string
	}{
		{[]recipe.Placement{recipe.PlacementHost}, "host", "go"},
		{[]recipe.Placement{recipe.PlacementHost, recipe.PlacementHost}, "host", "go"},
		{[]recipe.Placement{recipe.PlacementHybrid}, "cuda:0", "cuda"},
		{[]recipe.Placement{recipe.PlacementHost, recipe.PlacementDevice}, "cuda:0", "cuda"},
	} {
		var definition recipe.Definition
		for _, placement := range test.placements {
			definition.Nodes = append(definition.Nodes, recipe.Node{Placement: placement})
		}
		if device, backend := RecipeDevice(definition); device != test.device || backend != test.backend {
			t.Errorf("placements %v record %s/%s, want %s/%s", test.placements, device, backend, test.device, test.backend)
		}
	}
}
