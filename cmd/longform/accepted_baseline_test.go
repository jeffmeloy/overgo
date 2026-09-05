package main

import (
	"context"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/discovery"
	"overgo/internal/longform"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
)

func guardResult(t *testing.T) longform.Result {
	t.Helper()
	r := contractResult(t, "guard")
	id := func(kind artifact.Kind) artifact.ID {
		value, _, err := artifact.Identify(kind, strings.NewReader("guard fixture"))
		if err != nil {
			t.Fatal(err)
		}
		return value
	}
	r.Program = modelrecipe.ProgramIdentity{Model: r.Inputs.Model, Profile: id(artifact.KindProfile), Definition: id(artifact.KindModelDefinition),
		Recipe: id(artifact.KindRecipe), RecipeVersion: 1, Placement: recipe.PlacementHybrid, Residency: recipe.ResidencyHybridNative, Runtime: modelrecipe.RuntimeInference}
	r.Device = driver.DeviceInfo{Name: "fixture device", TotalMemoryBytes: 1 << 30}
	r.ContextLength = uint32(r.Floors.CheckRungCeiling + max(r.Floors.OutputTokens, r.Floors.ScoreTokens))
	r.BudgetNS, r.WallNS = int64(time.Minute), int64(time.Second)
	r.Shape.Measure.PromptTokens = r.Floors.ShortPromptTokens
	r.Shape.Measure.OutputTokens = len(r.Shape.OutputIDs)
	r.Shape.Measure.Memory = driver.MemoryStats{CurrentBytes: 1024, PeakBytes: 2048, Allocations: 1}
	r.Rungs = nil
	for _, length := range longform.LadderRungs(r.ContextLength, int(r.ContextLength), r.Floors, r.Floors.CheckRungCeiling) {
		m := r.Measure
		m.PromptTokens = length
		m.Score.ContextGain = m.Score.ShortContextNLL - m.Score.LongContextNLL
		m.Memory = driver.MemoryStats{CurrentBytes: uint64(length), PeakBytes: uint64(length * 2), Allocations: 1}
		ids := make([]int32, m.OutputTokens)
		for i := range ids {
			ids[i] = int32(i)
		}
		r.Rungs = append(r.Rungs, longform.Rung{Measure: m, OutputIDs: ids})
		if length == r.Floors.PromptTokens {
			r.Measure = m
		}
	}
	return r
}

func TestGuardEvidenceContract(t *testing.T) {
	t.Run("complete guard and rejected evidence", func(t *testing.T) {
		if err := validateGuard(guardResult(t)); err != nil {
			t.Fatal(err)
		}
		for name, mutate := range map[string]func(*longform.Result){
			"legacy identity":     func(r *longform.Result) { r.Program = modelrecipe.ProgramIdentity{} },
			"wrong identity kind": func(r *longform.Result) { r.Program.Profile = r.Program.Model },
			"missing allocation":  func(r *longform.Result) { r.Shape.Measure.Memory = driver.MemoryStats{} },
			"over capacity":       func(r *longform.Result) { r.Shape.Measure.Memory.PeakBytes = r.Device.TotalMemoryBytes + 1 },
			"peak below retained": func(r *longform.Result) { r.Shape.Measure.Memory.PeakBytes = r.Shape.Measure.Memory.CurrentBytes - 1 },
			"missing rung":        func(r *longform.Result) { r.Rungs = r.Rungs[:len(r.Rungs)-1] },
			"duplicate rung":      func(r *longform.Result) { r.Rungs = append(r.Rungs, r.Rungs[len(r.Rungs)-1]) },
			"score denominator":   func(r *longform.Result) { r.Rungs[0].Measure.Score.ScoreTokens-- },
			"score NaN":           func(r *longform.Result) { r.Rungs[0].Measure.Score.ShortContextNLL = math.NaN() },
			"invented gain":       func(r *longform.Result) { r.Rungs[0].Measure.Score.ContextGain++ },
			"missing fingerprint": func(r *longform.Result) { r.Shape.OutputIDs = nil },
			"output denominator":  func(r *longform.Result) { r.Rungs[0].Measure.OutputTokens++ },
			"judged measure":      func(r *longform.Result) { r.Measure.DecodeTokensPerSecond++ },
			"failed verdict":      func(r *longform.Result) { r.Verdict.Passed = false },
			"relaxed floor":       func(r *longform.Result) { r.Floors.NLLTolerance++ },
			"nonfinite rate":      func(r *longform.Result) { r.Shape.Measure.DecodeTokensPerSecond = math.Inf(1) },
			"nonfinite benchmark": func(r *longform.Result) { r.Short.DecodeTokensPerSecond = math.NaN() },
			"zero rate":           func(r *longform.Result) { r.Rungs[0].Measure.DecodeTokensPerSecond = 0 },
			"unbounded":           func(r *longform.Result) { r.BudgetNS = 0 },
			"over budget":         func(r *longform.Result) { r.WallNS = r.BudgetNS + 1 },
		} {
			t.Run(name, func(t *testing.T) {
				r := guardResult(t)
				mutate(&r)
				if err := validateGuard(r); err == nil {
					t.Fatal("incomplete guard accepted")
				}
			})
		}
	})
	t.Run("resource and execution regressions", func(t *testing.T) {
		r := guardResult(t)
		for name, mutate := range map[string]func(*longform.Result){
			"retained": func(r *longform.Result) { r.Shape.Measure.Memory.CurrentBytes++ },
			"peak":     func(r *longform.Result) { r.Shape.Measure.Memory.PeakBytes++ },
			"recipe":   func(r *longform.Result) { r.Program.RecipeVersion++ },
			"device":   func(r *longform.Result) { r.Device.Ordinal++ },
			"context":  func(r *longform.Result) { r.ContextLength++ },
		} {
			t.Run(name, func(t *testing.T) {
				fresh := r
				mutate(&fresh)
				if v := longform.Compare(r, fresh, r.Floors, r.Floors.CheckRungCeiling); v.Passed {
					t.Fatal("regression accepted")
				}
			})
		}
		fresh := r
		fresh.Shape.Measure.Memory.CurrentBytes--
		fresh.Shape.Measure.Memory.PeakBytes--
		if v := longform.Compare(r, fresh, r.Floors, r.Floors.CheckRungCeiling); !v.Passed {
			t.Fatal(v)
		}
	})
	t.Run("durable explicit selection and read only acceptance", func(t *testing.T) {
		r := guardResult(t)
		repository := t.TempDir()
		store, err := overgodb.Open(repository)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		accepted, err := longform.Publish(t.Context(), store, r.Inputs.Model, r)
		if err != nil {
			t.Fatal(err)
		}
		read, err := longform.ReadBaseline(t.Context(), store, accepted)
		if err != nil || !reflect.DeepEqual(r, read.Result) {
			t.Fatalf("round trip changed evidence: %v", err)
		}
		failed := r
		failed.Verdict = longform.Verdict{Reasons: []string{"later regression"}}
		bad, err := longform.Publish(t.Context(), store, r.Inputs.Model, failed)
		if err != nil {
			t.Fatal(err)
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		opts := options{Repository: repository, corpusText: "fixed corpus", ValidateBaselines: true, Baselines: []string{accepted.String()}}
		targets := []target{{weights: r.Inputs.Model, entry: discovery.Entry{Model: r.Program.Model, Recipe: r.Program.Recipe}}}
		if err := bindBaselines(t.Context(), opts, targets); err != nil {
			t.Fatal(err)
		}
		var output strings.Builder
		if err := validateSelectedBaselines(&output, targets, r.Surface); err != nil {
			t.Fatal(err)
		}
		if targets[0].record.Record != accepted || !strings.Contains(output.String(), "selected_models=1 explicit_records=1 short_shapes=1 rungs=4") || !strings.Contains(output.String(), "no models loaded") {
			t.Fatal(output.String())
		}
		if err := validateSelectedBaselines(io.Discard, targets, "stale"); err == nil {
			t.Fatal("stale accepted")
		}
		for name, mutate := range map[string]func(*options, *[]target){
			"failed latest":  func(o *options, _ *[]target) { o.Baselines = []string{bad.String()} },
			"duplicate":      func(o *options, _ *[]target) { o.Baselines = []string{accepted.String(), accepted.String()} },
			"missing":        func(o *options, _ *[]target) { o.Baselines = nil },
			"unselected":     func(_ *options, ts *[]target) { *ts = nil },
			"changed recipe": func(_ *options, ts *[]target) { (*ts)[0].entry.Recipe = artifact.ID{} },
			"changed corpus": func(o *options, _ *[]target) { o.corpusText = "different corpus" },
		} {
			t.Run(name, func(t *testing.T) {
				o, ts := opts, append([]target(nil), targets...)
				mutate(&o, &ts)
				if err := bindBaselines(t.Context(), o, ts); err == nil {
					t.Fatal("invalid selection accepted")
				}
			})
		}
	})
	t.Run("bounded guard publishes failure and refuses matching failure", func(t *testing.T) {
		r := guardResult(t)
		measure := func(_ context.Context, _ io.Writer, _ options, _ target, _, _ string, f longform.Floors, ceiling int) (longform.Result, error) {
			if ceiling != f.CheckRungCeiling {
				t.Fatal("unbounded guard")
			}
			fresh := r
			fresh.Verdict.Passed = false
			return fresh, nil
		}
		published := false
		publish := func(_ context.Context, _ string, _ artifact.ID, result longform.Result) (artifact.ID, error) {
			published = true
			if result.Verdict.Passed || len(result.Verdict.Reasons) == 0 {
				t.Fatal("failure was hidden")
			}
			return artifact.ID{}, nil
		}
		targets := []target{{record: longform.Summary{Result: r}}}
		if err := runTargets(t.Context(), io.Discard, options{Guard: true, Publish: true, Budget: time.Minute}, targets, r.Commit, r.Surface, measure, publish); err == nil || !published {
			t.Fatal("failed evidence lost or accepted")
		}
		if err := runTargets(t.Context(), io.Discard, options{Guard: true, Check: true, Budget: time.Minute}, targets, r.Commit, r.Surface, measure, nil); err == nil {
			t.Fatal("matching failed guard passed")
		}
	})
	t.Run("explicit modes", func(t *testing.T) {
		for _, args := range [][]string{
			{"-all", "-budget", "1m", "-guard"},
			{"-all", "-budget", "1m", "-validate-baselines", "-corpus", "fixed"},
			{"-all", "-budget", "1m", "-validate-baselines", "-baseline", "selected"},
			{"-all", "-budget", "1m", "-validate-baselines", "-publish", "-corpus", "fixed", "-baseline", "selected"},
		} {
			if _, err := parseOptions(args); err == nil {
				t.Fatalf("accepted %v", args)
			}
		}
		if _, err := parseOptions([]string{"-all", "-budget", "1m", "-validate-baselines", "-corpus", "fixed", "-baseline", "selected"}); err != nil {
			t.Fatal(err)
		}
	})
}
