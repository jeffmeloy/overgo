package main

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestTrainRoutesFoldedProbes holds each route, one subtest by its name, to
// the probe command it replaced: the command is gone and the route is
// reached by name through cmd/train. An unknown name is refused with the
// routes listed.
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
			if err := runRoute(t.Context(), []string{name}); err == nil || !strings.Contains(err.Error(), routeFlag+" "+name) {
				t.Errorf("route %s without its flags = %v, want a refusal naming the route", name, err)
			}
		})
	}
	t.Run("unknown", func(t *testing.T) {
		if err := runRoute(t.Context(), []string{"absent"}); err == nil || !strings.Contains(err.Error(), strings.Join(slices.Sorted(maps.Keys(trainRoutes)), ", ")) {
			t.Errorf("unknown route = %v, want a refusal listing the routes", err)
		}
	})
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
// command that existed is gone and its trainer is a cmd/train route.
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
}
