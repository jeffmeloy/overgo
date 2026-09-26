package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// TestTrainRoutesFoldedProbes holds each route to the probe command it
// replaced: the command is gone, the route is reached by name through
// cmd/train, an unknown name is refused with the routes listed, and the real
// Un-0 checkpoint trains through the oscillatorimage route and descends.
func TestTrainRoutesFoldedProbes(t *testing.T) {
	for name := range trainRoutes {
		if _, err := os.Stat(filepath.Join("..", name+"-train-probe")); !os.IsNotExist(err) {
			t.Errorf("cmd/%s-train-probe still exists beside its cmd/train route (%v)", name, err)
		}
		if err := runRoute([]string{name}); err == nil || !strings.Contains(err.Error(), routeFlag+" "+name) {
			t.Errorf("route %s without its flags = %v, want a refusal naming the route", name, err)
		}
	}
	if err := runRoute([]string{"absent"}); err == nil || !strings.Contains(err.Error(), "oscillatorimage, tabular") {
		t.Errorf("unknown route = %v, want a refusal listing the routes", err)
	}
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
