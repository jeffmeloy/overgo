package main

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// TestTrainRoutesFoldedProbes holds each route, one subtest by its name, to
// the probe command it replaced: the command is gone and the route is
// reached by name through cmd/train. An unknown name is refused with the
// routes listed, and the real Un-0 checkpoint trains through the
// oscillatorimage route and descends.
func TestTrainRoutesFoldedProbes(t *testing.T) {
	for name := range trainRoutes {
		t.Run(name, func(t *testing.T) {
			if _, err := os.Stat(filepath.Join("..", name+"-train-probe")); !os.IsNotExist(err) {
				t.Errorf("cmd/%s-train-probe still exists beside its cmd/train route (%v)", name, err)
			}
			if err := runRoute([]string{name}); err == nil || !strings.Contains(err.Error(), routeFlag+" "+name) {
				t.Errorf("route %s without its flags = %v, want a refusal naming the route", name, err)
			}
		})
	}
	t.Run("unknown", func(t *testing.T) {
		if err := runRoute([]string{"absent"}); err == nil || !strings.Contains(err.Error(), strings.Join(slices.Sorted(maps.Keys(trainRoutes)), ", ")) {
			t.Errorf("unknown route = %v, want a refusal listing the routes", err)
		}
	})
	t.Run("oscillatorimage-un0", trainsUn0)
}

// trainsUn0 trains the real Un-0 checkpoint through the oscillatorimage
// route and holds it to descending.
func trainsUn0(t *testing.T) {
	testskip.Short(t, "trains the real Un-0 checkpoint")
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	model := roots.ResolveModelPath("Un-0")
	if _, err := os.Stat(filepath.Join(model, "config.json")); err != nil {
		testskip.NotApplicable(t, "the Un-0 checkpoint is absent at "+model)
	}
	if err := runRoute([]string{"oscillatorimage", "-model", model, "-steps", "2"}); err != nil {
		t.Fatalf("oscillatorimage route on Un-0: %v", err)
	}
}
