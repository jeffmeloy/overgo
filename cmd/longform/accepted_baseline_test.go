package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/longform"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// Failed records are diagnostic counterexamples only; this reader cannot
// promote a failure.
func measuredGuardRecord(t *testing.T, store *overgodb.Store, text string) longform.Summary {
	t.Helper()
	id, err := artifact.ParseID(text)
	if err != nil {
		t.Fatal(err)
	}
	content, found, err := artifact.ReadContent(t.Context(), store, id)
	if err != nil || !found {
		t.Fatalf("counterexample %s absent: %v", id, err)
	}
	record, err := runrecord.ParseModelVerification(content.Data)
	if err != nil || record.ID != id || len(record.Claims) != 1 || record.Claims[0].Capability != longform.Capability || len(record.Claims[0].Evidence) != 1 {
		t.Fatalf("invalid measurement record %s: %v", id, err)
	}
	content, found, err = artifact.ReadContent(t.Context(), store, record.Claims[0].Evidence[0])
	if err != nil || !found {
		t.Fatalf("measurement body absent: %v", err)
	}
	var result longform.Result
	if err := json.Unmarshal(content.Data, &result); err != nil {
		t.Fatal(err)
	}
	// Validate original bytes; newer optional fields change a reserialization.
	body, _, err := artifact.Identify(artifact.KindEvidence, bytes.NewReader(content.Data))
	if err != nil {
		t.Fatal(err)
	}
	claim, _, _, err := longform.Claim(result)
	claim.Evidence = []artifact.ID{body}
	if err != nil || body != record.Claims[0].Evidence[0] || !reflect.DeepEqual(claim, record.Claims[0]) || result.Inputs.Model != record.Model {
		t.Fatalf("counterexample provenance differs: %v", err)
	}
	return longform.Summary{Record: id, Result: result}
}

func guardResult(t *testing.T) longform.Result {
	t.Helper()
	r := contractResult(t, "guard")
	r.Inputs.Protocol = longform.GuardContinuation
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
	t.Run("frozen measured corpus", func(t *testing.T) {
		corpus, err := os.ReadFile("testdata/guard-corpus.txt")
		if err != nil {
			t.Fatal(err)
		}
		// Exact bytes measured by the Qwen/E4B records at 3a75f6fb.
		if digest := fmt.Sprintf("%x", sha256.Sum256(corpus)); digest != "0968f131791aca99b7526836c9e8f643b2e63777a85353d42c36ecad49e0d29b" {
			t.Fatalf("frozen measurement corpus changed: %s", digest)
		}
	})
	t.Run("complete guard and rejected evidence", func(t *testing.T) {
		if err := validateGuard(guardResult(t)); err != nil {
			t.Fatal(err)
		}
		for name, mutate := range map[string]func(*longform.Result){
			"natural stop protocol": func(r *longform.Result) { r.Inputs.Protocol = longform.RawContinuation },
			"incomplete budget": func(r *longform.Result) {
				r.Rungs[0].OutputIDs = r.Rungs[0].OutputIDs[:r.Floors.IdenticalTokens]
				r.Rungs[0].Measure.OutputTokens = r.Floors.IdenticalTokens
			},
			"early stop":          func(r *longform.Result) { r.Shape.Measure.StoppedEarly = true },
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
	t.Run("bounded guard publishes failure", func(t *testing.T) {
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
		publish := func(_ context.Context, _ artifact.ID, result longform.Result) (artifact.ID, error) {
			published = true
			if result.Verdict.Passed || len(result.Verdict.Reasons) == 0 {
				t.Fatal("failure was hidden")
			}
			return artifact.ID{}, nil
		}
		if err := runTargets(t.Context(), io.Discard, options{Guard: true, Publish: true, Budget: time.Minute}, []target{{}}, r.Commit, r.Surface, measure, publish); err == nil || !published {
			t.Fatal("failed evidence lost or accepted")
		}
	})
	t.Run("guard requires a fixed corpus", func(t *testing.T) {
		if _, err := parseOptions([]string{"-budget", "1m", "-guard", "model.gguf"}); err == nil {
			t.Fatal("guard accepted the repository corpus")
		}
	})
}
