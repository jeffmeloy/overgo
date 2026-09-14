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
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/longform"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testskip"
)

// TestAcceptedBaselineEvidence reads immutable measurements, never the latest
// record or a newly generated self-reference. It establishes the historical
// guard; -validate-baselines separately requires current-surface admission.
func TestAcceptedBaselineEvidence(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	if os.Getenv(dataroot.Env) == "" {
		t.Skip("integration: set OVERGO_DATA_ROOT to validate the exact stored guard records; no measurement runs")
	}
	roots, err := dataroot.Resolve(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	const acceptedSurface = "d3ff45edf061f7a2f8833d9625a6154532e1760fd08878cb89343f46a728341f"
	const acceptedCommit = "e4b9d1e6b1b46626f1b934a7360db76c960277ce"
	fixtures := []struct {
		name, before, refused string
		after                 [3]string // Three sequential isolated samples; first is reference, never best-run selection.
	}{
		{
			name:    "Qwen capacity session",
			before:  "evidence:sha256:8fe6efde3661a9f24a481002fbd5696536e52f782e00580e0ab6d991d3b06e39",
			refused: "evidence:sha256:1c61e2821f010632b9b7a99b05261d7ccad9649929a9bfbc1a923d4430bfd8e7",
			after: [3]string{
				"evidence:sha256:0190b70b93219859525aff79e925f30ccc8b6be5647cb36eba2838ca56bef2c8",
				"evidence:sha256:7475932433954855147a950a1aff5db66006d18225266cdae90dd5c1fd4cf81b",
				"evidence:sha256:2a0f7c99ed9d4c8d4dfeca916c494b371c3cf413bfa5f20acfbb21248856c445",
			},
		},
		{
			name:    "E4B request session",
			before:  "evidence:sha256:2da44cefe6b7db08e49315acf50b727421fb2fe212197ee4cba16db680bcdd43",
			refused: "evidence:sha256:83a69ff733e8cd2f52388bb94e28eb870fdd8642e10b189a37923c24b15a0d2a",
			after: [3]string{
				"evidence:sha256:3d6b27ffaf0fa747799b64f85eda10e5261cf09465c0457652e0a2d4f422e928",
				"evidence:sha256:a7804c9c52b752a5b3b80979b741d1662fd3baf8f18733517c1923d3593472ca",
				"evidence:sha256:3497b3b65674f032badd1df1f2395c0dd3c166981d66bed7c5e242455c28eff8",
			},
		},
	}
	requireStoreLineage(t, store, fixtures[0].before)
	seen := make(map[artifact.ID]bool)
	var selected []target
	var ids []string
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			before := measuredGuardRecord(t, store, fixture.before)
			refused := measuredGuardRecord(t, store, fixture.refused)
			if err := validateGuard(refused.Result); err == nil {
				t.Fatal("original failed or incomplete measurement accepted")
			}
			if !before.Result.Verdict.Passed {
				if _, err := longform.ReadBaseline(t.Context(), store, before.Record); err == nil {
					t.Fatal("failed pre-fix measurement accepted as a healthy baseline")
				}
			}
			var first longform.Result
			for index, text := range fixture.after {
				id, err := artifact.ParseID(text)
				if err != nil || seen[id] || id == before.Record || id == refused.Record {
					t.Fatalf("invalid, duplicate or self-selected record %s: %v", text, err)
				}
				seen[id] = true
				accepted, err := longform.ReadBaseline(t.Context(), store, id)
				if err != nil {
					t.Fatal(err)
				}
				fresh := accepted.Result
				if fresh.Surface != acceptedSurface || fresh.Commit != acceptedCommit {
					t.Fatal("record does not identify the committed measured implementation")
				}
				if err := validateGuard(fresh); err != nil {
					t.Fatal(err)
				}
				if v := longform.Compare(before.Result, fresh, longform.DeclaredFloors(), fresh.Floors.CheckRungCeiling); !v.Passed {
					t.Fatalf("pre-fix comparison: %s", v)
				}
				if index > 0 {
					if v := longform.Compare(first, fresh, fresh.Floors, fresh.Floors.CheckRungCeiling); !v.Passed {
						t.Fatalf("repeat comparison: %s", v)
					}
				} else {
					first = fresh
					selected = append(selected, target{weights: fresh.Inputs.Model, entry: discovery.Entry{Model: fresh.Program.Model, Recipe: fresh.Program.Recipe}})
					ids = append(ids, text)
				}
				t.Logf("record=%s judged_decode=%.1f short=1 rungs=%d; tokens, NLL, rates and allocation non-growth checked", id, fresh.Measure.DecodeTokensPerSecond, len(fresh.Rungs))
			}
		})
	}
	if len(selected) != len(fixtures) {
		t.Fatal("missing model from the declared guard denominator")
	}
	opts := options{Repository: roots.Store, Corpus: "testdata/guard-corpus.txt", ValidateBaselines: true, Baselines: ids}
	if err := bindBaselines(t.Context(), opts, selected); err != nil {
		t.Fatal(err)
	}
	var output strings.Builder
	if err := validateSelectedBaselines(&output, selected, acceptedSurface); err != nil {
		t.Fatal(err)
	}
	t.Log(output.String())
	t.Logf("historical guard audit: models=%d repeated_records=%d; chat, full catalog, modalities and fresh performance excluded", len(selected), len(seen))
}

// Failed records are diagnostic counterexamples only. ReadBaseline remains the
// authority for every accepted reference; this reader cannot promote a failure.
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
		publish := func(_ context.Context, _ artifact.ID, result longform.Result) (artifact.ID, error) {
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
