package plan

import (
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

// TestRSICampaignRatchetAndParallelStructure pins the owner decision that
// architecture ratchets precede new RSI construction, a leased ready-frontier
// turns DAG slack into conflict-bound worktrees, and dependencies name exact
// producer steps. Completed rows leave plan.json, so assertions are
// intentionally conditional on a row still being retained.
func TestRSICampaignRatchetAndParallelStructure(t *testing.T) {
	document := loadCampaignPlan(t)
	wantDependencies := map[string][]string{
		"resource-coverage/strategy-identity-binding":          {"resource-coverage/coverage-projection"},
		"resource-coverage/reference-relevance-admission":      {"resource-coverage/coverage-projection"},
		"resource-coverage/fitness-integration":                {"resource-coverage/strategy-identity-binding", "resource-coverage/reference-relevance-admission"},
		"architecture-ratchets/completion-reference-authority": {"resource-coverage/fitness-integration"},
		"architecture-ratchets/core-entry-guards":              {"architecture-ratchets/completion-reference-authority", "storage-modular-core/repository-parity", "common-publication-path/semantic-batches", "storage-operational-boundary/transition-boundary", "process-supervision/process-boundary-guard", "causal-execution/causal-projection", "stimulus-reconciliation/coalesced-followup", "capability-bound-activation/placement-integration", "webhook-ingress/workflow-dispatch", "activation-matrix/profile-coverage"},
		"architecture-ratchets/go-only-guard":                  {"architecture-ratchets/core-entry-guards"},
		"campaign-concurrency/ready-frontier-leases":           {"architecture-ratchets/go-only-guard"},
		"interaction-efficiency/projection-query-plans":        {"storage-projections/atomic-publication", "redundancy-interaction-baseline/interaction-traces", "campaign-concurrency/ready-frontier-leases"},
		"interaction-efficiency/head-bound-deltas":             {"interaction-efficiency/projection-query-plans", "storage-operational-boundary/transition-boundary"},
		"interaction-efficiency/incremental-context":           {"interaction-efficiency/head-bound-deltas", "stimulus-reconciliation/stable-cursors", "capability-bound-activation/placement-integration"},
		"interaction-efficiency/coalesced-coordination":        {"interaction-efficiency/incremental-context", "webhook-ingress/workflow-dispatch"},
		"interaction-efficiency/tool-call-propensity-control":  {"interaction-efficiency/efficiency-gate"},
		"model-prototypes/prototype-contract":                  {"campaign-concurrency/ready-frontier-leases", "capability-bound-activation/capability-identity", "causal-execution/causal-context"},
		"model-recipe-lifecycle/recipe-derivation":             {"model-prototypes/prototype-materialization"},
		"model-recipe-lifecycle/isolated-performance-evidence": {"model-recipe-lifecycle/recipe-validation", "resource-coverage/fitness-integration", "process-supervision/process-boundary-guard"},
		"model-recipe-lifecycle/recipe-evaluation":             {"model-recipe-lifecycle/isolated-performance-evidence"},
		"evidence-routing/decision-contract":                   {"model-recipe-lifecycle/recipe-validation"},
		"evidence-routing/evidence-selection":                  {"evidence-routing/decision-contract", "capability-bound-activation/placement-integration", "model-recipe-lifecycle/recipe-evaluation"},
		"evidence-routing/counterfactual-replay":               {"evidence-routing/evidence-selection", "causal-execution/causal-projection"},
		"deterministic-rollout/rollout-plan":                   {"resource-coverage/fitness-integration", "causal-execution/causal-context", "model-recipe-lifecycle/recipe-evaluation", "evidence-routing/decision-contract"},
		"live-safety/safety-window":                            {"deterministic-rollout/projection-evidence"},
		"live-safety/atomic-rollback":                          {"live-safety/circuit-breaker", "deterministic-rollout/promotion-binding"},
		"architecture-closure/single-entry-guards":             {"architecture-ratchets/go-only-guard", "interaction-efficiency/tool-call-propensity-control", "model-recipe-lifecycle/recipe-activation", "evidence-routing/counterfactual-replay", "deterministic-rollout/promotion-binding", "live-safety/reentry"},
		"architecture-closure/go-only-guard":                   {"architecture-closure/single-entry-guards"},
	}
	for id, want := range wantDependencies {
		if step, found := retainedCampaignStep(document, id); found {
			got := slices.Clone(step.DependsOn)
			want = slices.Clone(want)
			slices.Sort(got)
			slices.Sort(want)
			if !slices.Equal(got, want) {
				t.Errorf("%s dependencies = %v, want %v", id, got, want)
			}
		}
	}
	for id, targets := range map[string][]string{
		"resource-coverage/strategy-identity-binding":          {"TestGateAttemptBindsStrategyIdentity", "TestStrategyComparisonRequiresIdentity"},
		"resource-coverage/reference-relevance-admission":      {"TestReferenceAdmissionRequiresSemanticRelevance"},
		"architecture-ratchets/completion-reference-authority": {"TestPrunedDependencyRequiresGatedCompletion", "TestCompletedPlanIdentityCannotBeReused"},
		"architecture-ratchets/core-entry-guards":              {"TestArchitectureRatchetAlwaysRequired", "TestProductionAuthorityBoundaries"},
		"architecture-ratchets/go-only-guard":                  {"TestArchitectureRatchetIncludesGoOnlyPolicy", "TestRSIRuntimeIsGoOnly"},
		"campaign-concurrency/ready-frontier-leases":           {"TestReadyFrontierLeaseIsolation", "TestMergeAcceptsGatedPrunedCompletion", "TestMergePreservesDependencyOrder"},
	} {
		if step, found := retainedCampaignStep(document, id); found {
			for _, target := range targets {
				if !strings.Contains(step.Verify, target) {
					t.Errorf("%s verifier omits %s", id, target)
				}
			}
		}
	}

	assertCampaignOrder(t, document, "resource-coverage", "architecture-ratchets")
	assertCampaignOrder(t, document, "architecture-ratchets", "campaign-concurrency")
	assertCampaignOrder(t, document, "campaign-concurrency", "interaction-efficiency")
	assertCampaignOrder(t, document, "campaign-concurrency", "model-prototypes")
	for _, id := range []string{
		"architecture-ratchets/completion-reference-authority",
		"architecture-ratchets/core-entry-guards",
		"architecture-ratchets/go-only-guard",
		"architecture-closure/single-entry-guards",
		"architecture-closure/go-only-guard",
	} {
		if step, found := retainedCampaignStep(document, id); found && !slices.Contains(step.Capabilities, "gate-ratchet") {
			t.Errorf("%s is not declared as a gate ratchet", id)
		}
	}
	if step, found := retainedCampaignStep(document, "architecture-ratchets/core-entry-guards"); found {
		for _, required := range []string{"always-required", "manifest-owned", "documentation-only", "wall cost", "redundant second test process"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("early authority ratchet rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "architecture-ratchets/completion-reference-authority"); found {
		for _, required := range []string{"ancestor gate commit", "refuse unknown or reused identities", "second completion ledger"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("completion authority rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "campaign-concurrency/ready-frontier-leases"); found {
		for _, required := range []string{"complete ordered ready frontier", "distinct worktrees", "gates commits serially", "ancestor gate evidence"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("ready-frontier rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "architecture-closure/rsi-end-to-end"); found {
		for _, required := range []string{"exact strategy identity", "semantically relevant typed references", "missing and degraded coverage", "fact, chunk, and raw-byte budgets"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("end-to-end acceptance omits %q", required)
			}
		}
	}

	assertCampaignFrontier(t, document,
		"resource-coverage/strategy-identity-binding",
		"resource-coverage/reference-relevance-admission")
	assertCampaignFrontier(t, document,
		"interaction-efficiency/projection-query-plans",
		"model-prototypes/prototype-contract")
	assertCampaignFrontier(t, document,
		"model-recipe-lifecycle/recipe-activation",
		"evidence-routing/evidence-selection",
		"deterministic-rollout/rollout-plan")
	assertCampaignFrontier(t, document,
		"deterministic-rollout/promotion-binding",
		"live-safety/safety-window")
}

func loadCampaignPlan(t *testing.T) Plan {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("campaign test path is unavailable")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	document, err := Load(filepath.Join(root, filepath.FromSlash(Path)))
	if err != nil {
		t.Fatal(err)
	}
	return document
}

func retainedCampaignStep(document Plan, id string) (Step, bool) {
	itemID, stepID, ok := strings.Cut(id, "/")
	if !ok {
		return Step{}, false
	}
	for _, item := range document.Items {
		if item.ID == itemID {
			for _, step := range item.Steps {
				if step.ID == stepID {
					return step, true
				}
			}
		}
	}
	return Step{}, false
}

func assertCampaignOrder(t *testing.T, document Plan, before, after string) {
	t.Helper()
	positions := map[string]int{}
	for index, item := range document.Items {
		if item.ID == before || item.ID == after {
			positions[item.ID] = index
		}
	}
	if left, leftFound := positions[before]; leftFound {
		if right, rightFound := positions[after]; rightFound && left >= right {
			t.Errorf("campaign order has %s after %s", before, after)
		}
	}
}

func assertCampaignFrontier(t *testing.T, document Plan, ids ...string) {
	t.Helper()
	retained := make([]string, 0, len(ids))
	for _, id := range ids {
		if _, found := retainedCampaignStep(document, id); found {
			retained = append(retained, id)
		}
	}
	for left := range retained {
		for right := left + 1; right < len(retained); right++ {
			if campaignDependsOn(document, retained[left], retained[right], map[string]bool{}) ||
				campaignDependsOn(document, retained[right], retained[left], map[string]bool{}) {
				t.Errorf("campaign frontier is serialized: %s and %s", retained[left], retained[right])
			}
		}
	}
}

func campaignDependsOn(document Plan, source, target string, seen map[string]bool) bool {
	if seen[source] {
		return false
	}
	seen[source] = true
	step, found := retainedCampaignStep(document, source)
	if !found {
		return false
	}
	for _, dependency := range step.DependsOn {
		if dependency == target || campaignDependsOn(document, dependency, target, seen) {
			return true
		}
	}
	return false
}
