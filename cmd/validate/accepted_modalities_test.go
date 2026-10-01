package main

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestProtocolCaseProjection(t *testing.T) {
	t.Parallel()
	declared := []json.RawMessage{json.RawMessage(`{"name":"first"}`), json.RawMessage(`{"name":"second"}`)}
	for name, observed := range map[string][]json.RawMessage{
		"complete": {declared[1], declared[0]}, "omitted": {declared[0]}, "duplicate": {declared[0], declared[0]}, "foreign": {declared[0], json.RawMessage(`{"name":"third"}`)},
	} {
		t.Run(name, func(t *testing.T) {
			if (checkProtocolCaseNames(declared, observed) == nil) != (name == "complete") {
				t.Fatal("case membership differs")
			}
		})
	}
}

func TestResourceCaseProjection(t *testing.T) {
	t.Parallel()
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, "resource-recipe")
	observation := testutil.ArtifactID(t, artifact.KindEvidence, "resource-observation")
	gate := runrecord.GateResult{CodeCommit: "source", Recipe: recipeID, Outcome: runrecord.OutcomeSucceeded, Steps: []runrecord.GateStep{
		{Name: "media-resource-contract", Outcome: runrecord.StepSucceeded, Evidence: "reloads=1;cycles=1;actions=1"},
		{Name: "media-r0", Outcome: runrecord.StepSucceeded, Evidence: fmt.Sprintf(`{"Mode":"mixed","Observation":%q}`, observation)},
	}}
	for name, mutate := range map[string]func(*runrecord.GateResult){
		"complete":        func(*runrecord.GateResult) {},
		"omitted":         func(g *runrecord.GateResult) { g.Steps = g.Steps[:1] },
		"duplicate":       func(g *runrecord.GateResult) { g.Steps = append(g.Steps, g.Steps[1]) },
		"failed":          func(g *runrecord.GateResult) { g.Steps[1].Outcome = runrecord.StepFailed },
		"failed-contract": func(g *runrecord.GateResult) { g.Steps[0].Outcome = runrecord.StepFailed },
		"zero":            func(g *runrecord.GateResult) { g.Steps[0].Evidence = "reloads=0;cycles=1;actions=1" },
		"overflow":        func(g *runrecord.GateResult) { g.Steps[0].Evidence = "reloads=18446744073709551615;cycles=2;actions=1" },
		"duplicate-axis":  func(g *runrecord.GateResult) { g.Steps[0].Evidence += ";reloads=1" },
		"changed-source":  func(g *runrecord.GateResult) { g.CodeCommit = "other" },
	} {
		t.Run(name, func(t *testing.T) {
			changed := gate
			changed.Steps = slices.Clone(gate.Steps)
			mutate(&changed)
			value := modalityProtocolProjection{Source: gate.CodeCommit, Recipe: recipeID, InputMode: "mixed-history"}
			if (projectResourceCaseCounts(&value, changed) == nil) != (name == "complete") {
				t.Fatal("resource denominator verdict differs")
			}
		})
	}
}
