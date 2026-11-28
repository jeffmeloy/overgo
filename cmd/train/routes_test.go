package main

import (
	"bytes"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
	"overgo/internal/testcheck"
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

// TestAProofThatCannotVerifyIsRefusedFirst holds a -proof run the validator awaits to the verification it exists for
// (SQA-522): the training command, run as validate runs it, whose build can name no code revision for its verification
// to bind, is refused with that reason under the validator's preflight, for the dense workflow and every route that
// takes -proof, before its admission says it is ready, so no preflight counts the proof ready and no card is used. The
// command's main reads and clears the preflight's variable before a route runs, and a check reading the variable had
// seen it empty and only warned (SQA-522 reopened); before that, a -proof run from a modified tree prepared and trained
// on the card, then printed that it verified nothing and exited 0. A -proof run by hand is told and goes on.
func TestAProofThatCannotVerifyIsRefusedFirst(t *testing.T) {
	// Serial: the training command runs as a child under the environment this sets.
	t.Setenv(commandChildEnvironment, childUnrevisioned)
	command := func(preflight bool, arguments ...string) (printed, refusal string) {
		t.Helper()
		var out bytes.Buffer
		written := filepath.Join(t.TempDir(), "error.txt")
		environment := append(os.Environ(), clioptions.ErrorFileEnvironment+"="+written)
		if preflight {
			environment = append(environment, clioptions.PreflightEnvironment+"=1")
		}
		testcheck.Must(processcontrol.Run(t.Context(), processcontrol.Command{Path: os.Args[0], Args: arguments, Stdout: &out, Env: environment}))(t)
		refused, _ := os.ReadFile(written)
		return out.String(), string(refused)
	}
	for _, arguments := range [][]string{{"-proof"}, {"-route", "dit", "-proof"}, {"-route", "mot", "-proof"}, {"-route", "codec", "-proof"}, {"-route", "tabular", "-proof"}} {
		if _, refusal := command(true, arguments...); !strings.Contains(refusal, unrevisionedReason) {
			t.Errorf("train %v under the validator's preflight, which can name no code revision, wrote %q; want it refused with the revision's reason", arguments, refusal)
		}
		if printed, refusal := command(false, arguments...); strings.Contains(refusal, unrevisionedReason) || !strings.Contains(printed, "will verify no recipe") {
			t.Errorf("train %v by hand printed %q and wrote %q; want it told it will verify no recipe and refused for nothing of that", arguments, printed, refusal)
		}
	}
}
