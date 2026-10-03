package evaluation

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// TestBenchmarkProducerFactsFollowEvidence keeps producer facts separate from
// publication: disagreement cannot invent a source, and a platform identifies
// only its own exact bytes. Legacy measured values stay visible as unknowns.
func TestBenchmarkProducerFactsFollowEvidence(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"known", "legacy", "foreign-code", "foreign-environment"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			weights := []byte("benchmark producer weights")
			model, err := artifact.IdentifyBytes(artifact.KindModel, weights)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "model.gguf")
			if err := os.WriteFile(path, weights, 0o600); err != nil {
				t.Fatal(err)
			}
			platform, err := runrecord.NewEnvironment(runrecord.Environment{Host: "fixture", OS: "windows", Arch: "amd64", Device: "fixture-gpu", Backend: "cuda", Driver: "fixture"})
			if err != nil {
				t.Fatal(err)
			}
			body := map[string]any{"code_commit": planTestCommit, "environment_id": platform.ID, "environment": platform,
				"load_ms": 100, "summary": map[string]float64{"decode_tokens_per_second_p50": 12, "prompt_tokens_per_second_p50": 24, "ttft_ms_p50": 5}}
			switch mode {
			case "legacy":
				delete(body, "code_commit")
				delete(body, "environment_id")
				delete(body, "environment")
			case "foreign-code":
				body["code_commit"] = strings.Repeat("a", len(planTestCommit))
			case "foreign-environment":
				platform.Driver = "different"
				body["environment"] = platform
			}
			data, err := json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
			evidence, err := artifact.IdentifyBytes(artifact.KindEvidence, data)
			if err != nil {
				t.Fatal(err)
			}
			record, err := runrecord.NewModelVerification(model, "fixture", []runrecord.CapabilityClaim{{Capability: "benchmark", Tier: runrecord.TierCapabilityMeasured, Commit: planTestCommit, WallNS: 200, PeakDeviceBytes: 300, Evidence: []artifact.ID{evidence}}})
			if err != nil {
				t.Fatal(err)
			}
			if err := runrecord.CommitVerificationClaim(t.Context(), store, record, data, evidence, path); err != nil {
				t.Fatal(err)
			}
			summary := LatestEvidence(t.Context(), store, nil, 0).BenchmarksByLocation[path]
			encoded, err := json.Marshal(summary)
			if err != nil {
				t.Fatal(err)
			}
			var facts struct {
				CodeCommit  string      `json:"code_commit"`
				Environment artifact.ID `json:"environment"`
			}
			if err := json.Unmarshal(encoded, &facts); err != nil {
				t.Fatal(err)
			}
			wantCode, wantEnvironment := planTestCommit, platform.ID
			if mode == "legacy" || mode == "foreign-code" {
				wantCode = ""
			}
			if mode == "legacy" || mode == "foreign-environment" {
				wantEnvironment = artifact.ID{}
			}
			if summary.Record != record.ID || summary.DecodeTokensPerSecond != 12 || summary.PromptTokensPerSecond != 24 || summary.FirstTokenMS != 5 || summary.LoadMS != 100 || summary.WallNS != 200 || summary.PeakDeviceBytes != 300 || facts.CodeCommit != wantCode || facts.Environment != wantEnvironment {
				t.Fatalf("summary = %+v, facts = %+v; want source %q and environment %s without changing measured values", summary, facts, wantCode, wantEnvironment)
			}
		})
	}
}

// TestEvaluationExecutionDoesNotInventMissingAuthorities keeps unknown execution
// facts distinct from a known raw or hosted protocol. The measured value and its
// evidence identity remain visible even when execution authority is absent.
func TestEvaluationExecutionDoesNotInventMissingAuthorities(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"raw", "hosted", "missing-run", "missing-input", "missing-plan", "missing-policy", "missing-environment", "foreign-source", "foreign-recipe", "foreign-dataset", "foreign-environment", "invalid-policy"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			store, err := overgodb.Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			platform, err := runrecord.NewEnvironment(runrecord.Environment{
				Host: "fixture", OS: "windows", Arch: "amd64", Device: "cpu", Backend: "host", Driver: "fixture",
			})
			if mode == "hosted" {
				platform, err = runrecord.NewEnvironment(runrecord.Environment{
					Host: "fixture", OS: "windows", Arch: "amd64", Device: "provider", Backend: runrecord.BackendRemote, Driver: "fixture",
				})
			}
			if err != nil {
				t.Fatal(err)
			}
			if mode != "missing-environment" {
				batch, err := platform.Batch("execution/environment")
				if err != nil {
					t.Fatal(err)
				}
				if _, err := store.Commit(t.Context(), batch); err != nil {
					t.Fatal(err)
				}
			}
			exact, err := CompileExact(exactFixture())
			if err != nil {
				t.Fatal(err)
			}
			authorities := ListingAuthorities()
			authorities.Environment, authorities.CodeCommit = platform.ID, planTestCommit
			if mode == "hosted" {
				authorities.Execution.Prompting = PromptingHostedChat
			}
			plan, err := BindExact(exact, authorities)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "invalid-policy" {
				policy, err := authorityContent(executionContract, ExecutionPolicy{Lifecycle: "automatic"})
				if err != nil {
					t.Fatal(err)
				}
				for index := range plan.authorities {
					if plan.authorities[index].Descriptor.ID == plan.Execution() {
						plan.authorities[index] = policy
					}
				}
				plan.body.Execution = policy.Descriptor.ID
				plan.identity, err = artifact.JSONID(artifact.KindProfile, plan.body)
				if err != nil {
					t.Fatal(err)
				}
			}
			publishPlanFixtureAuthorities(t, store, plan)
			if mode == "missing-policy" {
				plan.authorities = slices.DeleteFunc(plan.authorities, func(content artifact.Content) bool { return content.Descriptor.ID == plan.Execution() })
			}
			output := planID(t, artifact.KindEvaluation, "execution report")
			if _, err := store.Commit(t.Context(), artifact.Batch{Key: "execution/inputs", Artifacts: []artifact.Descriptor{{ID: plan.Identity()}, {ID: plan.Dataset()}, {ID: plan.Execution()}, {ID: output}}}); err != nil {
				t.Fatal(err)
			}
			if mode != "missing-plan" {
				if err := publishAuthorities(t.Context(), store, exact, plan); err != nil {
					t.Fatal(err)
				}
			}
			inputs, commit := []artifact.ID{plan.Identity()}, planTestCommit
			runRecipe, recordDataset, runEnvironment := plan.body.RuntimeRecipe, plan.Dataset(), platform.ID
			if mode == "missing-input" {
				inputs = nil
			}
			if mode == "foreign-source" {
				commit = strings.Repeat("b", len(planTestCommit))
			}
			if mode == "foreign-recipe" {
				runRecipe = planID(t, artifact.KindRecipe, "other recipe")
			}
			if mode == "foreign-dataset" {
				recordDataset = planID(t, artifact.KindDataset, "other dataset")
			}
			if mode == "foreign-environment" {
				runEnvironment = planID(t, artifact.KindEvidence, "other environment")
			}
			if mode == "foreign-recipe" || mode == "foreign-dataset" || mode == "foreign-environment" {
				if _, err := store.Commit(t.Context(), artifact.Batch{Key: "execution/run-parents", Artifacts: []artifact.Descriptor{{ID: runRecipe}, {ID: recordDataset}, {ID: runEnvironment}}}); err != nil {
					t.Fatal(err)
				}
			}
			run, err := runrecord.NewBoundRun(runRecipe, runrecord.OutcomeSucceeded, inputs, []artifact.ID{output}, "", commit, runEnvironment, 1, nil)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "missing-run" {
				if _, err := store.Commit(t.Context(), artifact.Batch{Key: "execution/run", Artifacts: []artifact.Descriptor{{ID: run.ID}}}); err != nil {
					t.Fatal(err)
				}
			} else {
				content, err := run.Content()
				if err != nil {
					t.Fatal(err)
				}
				batch, err := artifact.NewDocumentBatch("execution/run", []artifact.Content{content}, run.Lineage(), nil)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := store.Commit(t.Context(), batch); err != nil {
					t.Fatal(err)
				}
			}
			record, err := runrecord.NewEvaluation(plan.body.RuntimeRecipe, run.ID, recordDataset, []runrecord.Metric{{Name: "accuracy", Value: 0.5, Direction: runrecord.DirectionMaximize}})
			if err != nil {
				t.Fatal(err)
			}
			batch, err := record.Batch("execution/evaluation")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Commit(t.Context(), batch); err != nil {
				t.Fatal(err)
			}
			summaries := LatestEvidence(t.Context(), store, map[artifact.ID]string{recordDataset: "fixture"}, 0).EvaluationsByRecipe[plan.body.RuntimeRecipe]
			if len(summaries) != 1 {
				t.Fatalf("summaries = %+v, want the recorded measurement", summaries)
			}
			summary := summaries[0]
			data, err := json.Marshal(summary)
			var execution struct {
				Known bool `json:"execution_known"`
			}
			if err == nil {
				err = json.Unmarshal(data, &execution)
			}
			known := mode == "raw" || mode == "hosted"
			prompting := "unknown"
			if known {
				prompting = authorities.Execution.Prompting.Label()
			}
			if err != nil || execution.Known != known || summary.Reproducible != (mode == "raw") || summary.Prompting != prompting || summary.Record != record.ID || summary.Metrics["accuracy"] != 0.5 {
				t.Fatalf("summary = %+v (known=%t), want known=%t protocol=%s reproducible=%t with original measurement: %v", summary, execution.Known, known, prompting, mode == "raw", err)
			}
		})
	}
}

// TestEvaluationAlternativesKeepFullContract retains distinct policies and
// environments despite identical labels, while a later producer replaces only
// its own comparable result. Case and scorer changes also retain alternatives.
func TestEvaluationAlternativesKeepFullContract(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	exact, err := CompileExact(exactFixture())
	if err != nil {
		t.Fatal(err)
	}
	authorities := ListingAuthorities()
	authorities.CodeCommit = planTestCommit
	want := map[artifact.ID]float64{}
	suites := map[artifact.ID]string{}
	for index, mode := range []string{"resident", "isolated", "environment", "cases", "scorer", "new-source"} {
		platform, err := runrecord.NewEnvironment(runrecord.Environment{
			Host: "fixture", OS: "windows", Arch: "amd64", Device: "cpu", Backend: "host", Driver: "fixture",
		})
		if mode == "environment" {
			platform, err = runrecord.NewEnvironment(runrecord.Environment{
				Host: "other-fixture", OS: "windows", Arch: "amd64", Device: "cpu", Backend: "host", Driver: "fixture",
			})
		}
		if err != nil {
			t.Fatal(err)
		}
		batch, err := platform.Batch("comparison/environment/" + mode)
		if err != nil {
			t.Fatal(err)
		}
		if mode == "resident" || mode == "environment" {
			if _, err := store.Commit(t.Context(), batch); err != nil {
				t.Fatal(err)
			}
		}
		bound := authorities
		bound.Environment = platform.ID
		if mode == "isolated" {
			bound.Execution.Lifecycle = LifecycleIsolated
		}
		if mode == "new-source" {
			bound.CodeCommit = strings.Repeat("b", len(planTestCommit))
		}
		compiled := exact
		if mode == "cases" {
			changed := exactFixture()
			changed.Cases[0].Prompt = "a distinct case"
			compiled, err = CompileExact(changed)
			if err != nil {
				t.Fatal(err)
			}
		}
		plan, err := BindExact(compiled, bound)
		if err != nil {
			t.Fatal(err)
		}
		if mode == "scorer" {
			scorer, err := authorityContent(scorerProfileContract, struct {
				Version uint16 `json:"version"`
				Kind    string `json:"kind"`
			}{artifact.InitialDocumentVersion, "distinct-fixture-scorer"})
			if err != nil {
				t.Fatal(err)
			}
			plan.authorities = append(plan.authorities, scorer)
			plan.body.Scorer = scorer.Descriptor.ID
			plan.identity, err = artifact.JSONID(artifact.KindProfile, plan.body)
			if err != nil {
				t.Fatal(err)
			}
		}
		if mode == "resident" {
			publishPlanFixtureAuthorities(t, store, plan)
		}
		if err := publishAuthorities(t.Context(), store, compiled, plan); err != nil {
			t.Fatal(err)
		}
		output := planID(t, artifact.KindEvaluation, "comparison-output/"+mode)
		if _, err := store.Commit(t.Context(), artifact.Batch{Key: "comparison/output/" + mode, Artifacts: []artifact.Descriptor{{ID: output}}}); err != nil {
			t.Fatal(err)
		}
		run, err := runrecord.NewBoundRun(bound.RuntimeRecipe, runrecord.OutcomeSucceeded, []artifact.ID{plan.Identity()}, []artifact.ID{output}, "", bound.CodeCommit, platform.ID, 1, nil)
		if err != nil {
			t.Fatal(err)
		}
		content, err := run.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch, err = artifact.NewDocumentBatch("comparison/run/"+mode, []artifact.Content{content}, run.Lineage(), nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
		value := float64(index+1) / 10
		record, err := runrecord.NewEvaluation(bound.RuntimeRecipe, run.ID, compiled.dataset, []runrecord.Metric{{Name: "accuracy", Value: value, Direction: runrecord.DirectionMaximize}})
		if err != nil {
			t.Fatal(err)
		}
		batch, err = record.Batch("comparison/evaluation/" + mode)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
		// The first and last share acceptance and environment, so only the
		// last remains; all the intermediate contracts remain independently.
		if mode != "resident" {
			want[record.ID] = value
		}
		suites[compiled.dataset] = "same-label"
	}
	summaries := LatestEvidence(t.Context(), store, suites, 0).EvaluationsByRecipe[authorities.RuntimeRecipe]
	if len(summaries) != len(want) {
		t.Fatalf("summaries = %+v, want %d distinct contracts", summaries, len(want))
	}
	seen := map[artifact.ID]bool{}
	for _, summary := range summaries {
		data, err := json.Marshal(summary)
		var provenance struct {
			Comparison  artifact.ID `json:"comparison"`
			Environment artifact.ID `json:"environment"`
			CodeCommit  string      `json:"code_commit"`
		}
		if err == nil {
			err = json.Unmarshal(data, &provenance)
		}
		if value, found := want[summary.Record]; err != nil || !found || summary.Metrics["accuracy"] != value || summary.Prompting != "raw-completion" ||
			!summary.ExecutionKnown || !provenance.Comparison.Valid() || !provenance.Environment.Valid() || provenance.CodeCommit == "" || seen[provenance.Comparison] {
			t.Fatalf("summary = %+v, provenance = %+v: %v", summary, provenance, err)
		}
		if summary.Metrics["accuracy"] == 0.6 && provenance.CodeCommit != strings.Repeat("b", len(planTestCommit)) {
			t.Fatalf("new source attributed to %q", provenance.CodeCommit)
		}
		seen[provenance.Comparison] = true
	}
}
