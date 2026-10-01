package main

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// TestTrainRoutesFoldedProbes holds each route, one subtest by its name, to
// the probe command it replaced: the command is gone and the route is
// reached by name through cmd/train. An unknown name is refused with the
// routes listed, and the real Un-0 checkpoint's bootstrap trains through
// the oscillatorimage route, descends and publishes its checkpoint.
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

// trainsUn0 trains the real Un-0 checkpoint's scratch bootstrap through the
// oscillatorimage route and holds it to descending and publishing a
// checkpoint that trained from the bootstrap, not from Un-0's weights.
func trainsUn0(t *testing.T) {
	testskip.Short(t, "trains the real Un-0 checkpoint")
	repository := testutil.RepoRoot(t)
	roots, err := dataroot.Resolve(repository)
	if err != nil {
		t.Fatal(err)
	}
	model := roots.ResolveModelPath("Un-0")
	if _, err := os.Stat(filepath.Join(model, "config.json")); err != nil {
		testskip.NotApplicable(t, "the Un-0 checkpoint is absent at "+model)
	}
	out := filepath.Join(t.TempDir(), "checkpoint")
	if err := runRoute(t.Context(), []string{"oscillatorimage", "-model", model, "-steps", "2",
		"-stimulus", filepath.Join(repository, "docs", "verification", "un0-training-dataset.txt"),
		"-store", filepath.Join(t.TempDir(), "store"), "-out", out}); err != nil {
		t.Fatalf("oscillatorimage route on Un-0: %v", err)
	}
	checkpoint, err := trainingprogram.LoadCheckpoint(out)
	if err != nil {
		t.Fatalf("the trained Un-0 bootstrap's checkpoint: %v", err)
	}
	weights, err := trainingworkflow.IdentifyModelWeights(model)
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(weights)
	if err != nil {
		t.Fatal(err)
	}
	un0, _, err := artifact.Identify(artifact.KindModel, file)
	if closeErr := file.Close(); err != nil || closeErr != nil || checkpoint.Model == un0 {
		t.Fatalf("the trained bootstrap names %s as trained from (%v, %v), want its scratch construction, not Un-0's weights %s", checkpoint.Model, err, closeErr, un0)
	}
}
