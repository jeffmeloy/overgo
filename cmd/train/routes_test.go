package main

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
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
			former := name + "-train-probe"
			if name == "graft" {
				former = "graft-probe"
			}
			if _, err := os.Stat(filepath.Join("..", former)); !os.IsNotExist(err) {
				t.Errorf("cmd/%s still exists beside its cmd/train route (%v)", former, err)
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

// formerProbes are the training commands the route table replaced, each by
// the route that trains its model now.
var formerProbes = map[string]string{
	"adapter-train-probe": "adapter", "dit-train-probe": "dit", "flow-organ-train-probe": "flow-organ",
	"graft-probe": "graft", "hybrid-stack-train-probe": "hybrid-stack", "mixture-train-probe": "mixture",
	"mot-train-probe": "mot", "oscillatorimage-train-probe": "oscillatorimage",
	"seriesforecast-train-probe": "seriesforecast", "tabular-train-probe": "tabular",
}

// TestTrainRoutesEveryProbe holds the fold complete: every training probe
// command that existed is gone and its trainer is a cmd/train route, and
// every route the train lane dispatches runs cmd/train, never a probe.
func TestTrainRoutesEveryProbe(t *testing.T) {
	t.Parallel()
	for command, route := range formerProbes {
		if _, err := os.Stat(filepath.Join("..", command)); !os.IsNotExist(err) {
			t.Errorf("cmd/%s still exists (%v)", command, err)
		}
		if _, routed := trainRoutes[route]; !routed {
			t.Errorf("cmd/%s's trainer is not the %s route of cmd/train", command, route)
		}
	}
	var catalog struct {
		Routes map[string]struct {
			Argv []string `json:"argv"`
		} `json:"routes"`
	}
	if err := jsonfile.Decode(filepath.Join("..", "..", "docs", "training_routes.json"), &catalog); err != nil {
		t.Fatal(err)
	}
	for key, route := range catalog.Routes {
		if !slices.Equal(route.Argv[:min(len(route.Argv), len(trainInvocation))], trainInvocation) {
			t.Errorf("training route %s runs %v, not cmd/train", key, route.Argv)
		}
	}
}

// trainInvocation is how the training route table starts cmd/train.
var trainInvocation = []string{"go", "run", "./cmd/train"}

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
