package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/discovery"
	"overgo/internal/longform"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestGuardCoverageInventory(t *testing.T) {
	result := guardResult(t)
	baseline := longform.Summary{Record: testutil.ArtifactID(t, artifact.KindEvidence, "coverage baseline"), Result: result}
	targets := []target{{weights: result.Inputs.Model, entry: discovery.Entry{
		Model: result.Program.Model, Recipe: result.Program.Recipe, Location: result.ModelPath,
	}}}
	baselines := map[artifact.ID]longform.Summary{result.Inputs.Model: baseline}
	definition := recipe.Definition{ID: result.Program.Recipe, Task: recipe.TaskInference,
		Dependencies: []recipe.Dependency{{Role: recipe.DependencyDefinition, Artifact: result.Program.Definition}},
		Nodes:        []recipe.Node{{Module: modelrecipe.ModuleForwardTokens, Session: recipe.SessionCapacity, Residency: result.Program.Residency}},
	}
	resolved := modelrecipe.ResolvedModelDefinition{
		Document: modelrecipe.ModelDefinitionDocument{ID: result.Program.Definition, Model: result.Program.Model, Profile: result.Program.Profile},
		Spec:     model.Spec{AttentionSpec: model.AttentionSpec{HeadCount: 8, HeadCountKV: 2, KeyLength: 64, ValueLength: 64, SlidingWindow: 2048}},
		Tensors:  modelartifact.TensorInventoryDocument{Tensors: []modelartifact.TensorFact{{Name: "q", Storage: "F16"}, {Name: "k", Storage: "F16"}, {Name: "norm", Storage: "F32"}}},
	}
	resolve := func(target) (recipe.Definition, modelrecipe.ResolvedModelDefinition, error) {
		return definition, resolved, nil
	}
	report, err := inspectGuardCoverage(targets, baselines, result.Surface, resolve)
	if err != nil || report.Selected != 1 || report.Covered != 1 || report.Uncovered != 0 || len(report.Entries) != 1 {
		t.Fatalf("coverage = %+v, %v", report, err)
	}
	entry := report.Entries[0]
	if entry.Spec.HeadCountKV != 2 || entry.Spec.KeyLength != 64 || entry.Spec.SlidingWindow != 2048 ||
		entry.TensorStorage["F16"] != 2 || entry.TensorStorage["F32"] != 1 || entry.Nodes[0].Session != recipe.SessionCapacity ||
		entry.Baseline != baseline.Record.String() || entry.MeasuredWallNS != result.WallNS {
		t.Fatalf("execution facts lost: %+v", entry)
	}
	t.Run("same storage does not substitute for another geometry or model", func(t *testing.T) {
		other := targets[0]
		other.weights = testutil.ArtifactID(t, artifact.KindModel, "other weights")
		other.entry.Model = testutil.ArtifactID(t, artifact.KindModel, "other serving model")
		other.entry.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "other recipe")
		otherResolve := func(selected target) (recipe.Definition, modelrecipe.ResolvedModelDefinition, error) {
			d, r := definition, resolved
			if selected.weights == other.weights {
				d.ID, r.Document.Model = other.entry.Recipe, other.entry.Model
				d.Nodes = slices.Clone(d.Nodes)
				d.Nodes[0].Session = recipe.SessionRequest
				r.Spec.KeyLength = 256
			}
			return d, r, nil
		}
		report, err := inspectGuardCoverage(append(slices.Clone(targets), other), baselines, result.Surface, otherResolve)
		if err == nil || report.Selected != 2 || report.Covered != 1 || report.Uncovered != 1 {
			t.Fatalf("unmeasured model accepted: %+v %v", report, err)
		}
		var encoded bytes.Buffer
		if err := clioptions.WritePrettyJSON(&encoded, report); err != nil {
			t.Fatal(err)
		}
		var roundtrip guardCoverageReport
		if err := json.Unmarshal(encoded.Bytes(), &roundtrip); err != nil || roundtrip.Uncovered != 1 || len(roundtrip.Entries) != 2 {
			t.Fatalf("refusal report lost denominator: %+v %v", roundtrip, err)
		}
	})
	for name, mutate := range map[string]func(*longform.Summary){
		"stale source": func(b *longform.Summary) { b.Result.Surface = "different" },
		"different recipe": func(b *longform.Summary) {
			b.Result.Program.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "different")
		},
		"missing device": func(b *longform.Summary) { b.Result.Device.Name = "" },
		"failed verdict": func(b *longform.Summary) { b.Result.Verdict.Passed = false },
		"missing output": func(b *longform.Summary) { b.Result.Shape.OutputIDs = nil },
	} {
		t.Run(name, func(t *testing.T) {
			bad := baseline
			mutate(&bad)
			report, err := inspectGuardCoverage(targets, map[artifact.ID]longform.Summary{result.Inputs.Model: bad}, result.Surface, resolve)
			if err == nil || report.Covered != 0 || report.Uncovered != 1 || report.Entries[0].Gap == "" {
				t.Fatalf("invalid coverage accepted: %+v %v", report, err)
			}
		})
	}
	t.Run("absent facts remain reportable", func(t *testing.T) {
		report, err := inspectGuardCoverage(targets, baselines, result.Surface, func(target) (recipe.Definition, modelrecipe.ResolvedModelDefinition, error) {
			return recipe.Definition{}, modelrecipe.ResolvedModelDefinition{}, errors.New("missing model definition")
		})
		if err == nil || report.Uncovered != 1 || report.Entries[0].Gap == "" {
			t.Fatalf("missing facts accepted: %+v %v", report, err)
		}
		if err := clioptions.WritePrettyJSON(new(bytes.Buffer), report); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("empty and unused are refusals", func(t *testing.T) {
		if _, err := inspectGuardCoverage(nil, baselines, result.Surface, resolve); err == nil {
			t.Fatal("empty inventory accepted")
		}
		if _, err := inspectGuardCoverage(targets, baselines, "", resolve); err == nil {
			t.Fatal("unbound surface accepted")
		}
		unused := map[artifact.ID]longform.Summary{result.Inputs.Model: baseline, testutil.ArtifactID(t, artifact.KindModel, "unused"): baseline}
		if report, err := inspectGuardCoverage(targets, unused, result.Surface, resolve); err == nil || len(report.Unused) != 1 {
			t.Fatalf("unused baseline accepted: %+v %v", report, err)
		}
	})
	t.Run("CLI cannot mix inventory and measurement", func(t *testing.T) {
		args := []string{"-guard-coverage", "-budget", "1m", "-corpus", "fixed", "-baseline", baseline.Record.String(), "model.gguf"}
		if options, err := parseOptions(args); err != nil || !options.GuardCoverage {
			t.Fatalf("coverage options: %+v %v", options, err)
		}
		for _, flag := range []string{"-check", "-publish", "-validate-baselines", "-guard"} {
			if _, err := parseOptions(append([]string{flag}, args...)); err == nil {
				t.Fatalf("mixed mode %s accepted", flag)
			}
		}
	})
}
