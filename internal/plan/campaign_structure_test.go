package plan

import (
	"io/fs"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestSingleCanonicalCampaignPlan(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("campaign test path is unavailable")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	want := filepath.ToSlash(Path)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			switch entry.Name() {
			case ".git", "tmp", "overgodb-store", "external", "vendor":
				if path != root {
					return fs.SkipDir
				}
			}
			return nil
		}
		if entry.Name() != "plan.json" {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if got := filepath.ToSlash(relative); got != want {
			t.Errorf("competing live plan file %s; %s is the sole plan authority", got, want)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestRSICampaignRatchetAndParallelStructure pins the owner decision that
// architecture ratchets precede new RSI construction, a leased ready-frontier
// turns DAG slack into conflict-bound worktrees, and dependencies name exact
// producer steps. Completed rows leave plan.json, so row-specific assertions
// are conditional while the authority and DAG-shape ratchets remain active.
func TestRSICampaignRatchetAndParallelStructure(t *testing.T) {
	document := loadCampaignPlan(t)
	assertIntegratedReconciliationSnapshot(t, document)
	for _, required := range []string{"sole live execution plan", "imported branch plans", "four shared components", "cross-domain candidate spine", "promotion and rollback lifecycle", "evidence-driven budget and stop driver", "transport-neutral UTCP", "deterministic in-process orchestration"} {
		if !strings.Contains(document.Doctrine, required) {
			t.Errorf("campaign doctrine omits shared-spine decision %q", required)
		}
	}
	wantDependencies := map[string][]string{
		"resource-coverage/strategy-identity-binding":          {"resource-coverage/coverage-projection"},
		"resource-coverage/reference-relevance-admission":      {"resource-coverage/coverage-projection"},
		"resource-coverage/fitness-integration":                {"resource-coverage/strategy-identity-binding", "resource-coverage/reference-relevance-admission"},
		"architecture-ratchets/completion-reference-authority": {"resource-coverage/fitness-integration"},
		"architecture-ratchets/core-entry-guards":              {"architecture-ratchets/completion-reference-authority", "storage-modular-core/repository-parity", "common-publication-path/semantic-batches", "storage-operational-boundary/transition-boundary", "process-supervision/process-boundary-guard", "causal-execution/causal-projection", "stimulus-reconciliation/coalesced-followup", "capability-bound-activation/placement-integration", "webhook-ingress/workflow-dispatch", "activation-matrix/profile-coverage"},
		"architecture-ratchets/go-only-guard":                  {"architecture-ratchets/core-entry-guards"},
		"campaign-concurrency/ready-frontier-leases":           {"architecture-ratchets/go-only-guard"},
		"interaction-efficiency/bounded-training-evidence":     {"storage-projections/atomic-publication", "resource-coverage/fitness-integration", "redundancy-interaction-baseline/interaction-traces", "architecture-ratchets/go-only-guard"},
		"interaction-efficiency/projection-query-plans":        {"interaction-efficiency/bounded-training-evidence"},
		"interaction-efficiency/head-bound-deltas":             {"interaction-efficiency/projection-query-plans", "storage-operational-boundary/transition-boundary"},
		"interaction-efficiency/incremental-context":           {"interaction-efficiency/head-bound-deltas", "stimulus-reconciliation/stable-cursors", "capability-bound-activation/placement-integration"},
		"interaction-efficiency/coalesced-coordination":        {"interaction-efficiency/head-bound-deltas", "webhook-ingress/workflow-dispatch"},
		"interaction-efficiency/operator-decisions":            {"interaction-efficiency/head-bound-deltas", "causal-execution/causal-projection"},
		"interaction-efficiency/efficiency-gate":               {"interaction-efficiency/incremental-context", "interaction-efficiency/coalesced-coordination", "interaction-efficiency/operator-decisions"},
		"interaction-efficiency/tool-call-propensity-control":  {"interaction-efficiency/efficiency-gate", "model-prototypes/prototype-contract"},
		"model-prototypes/prototype-contract":                  {"architecture-ratchets/go-only-guard", "capability-bound-activation/capability-identity", "causal-execution/causal-context"},
		"model-prototypes/prototype-admission":                 {"model-prototypes/prototype-contract", "resource-coverage/reference-relevance-admission"},
		"model-prototypes/prototype-materialization":           {"model-prototypes/prototype-compile", "interaction-efficiency/bounded-training-evidence"},
		"model-recipe-lifecycle/recipe-derivation":             {"model-prototypes/prototype-materialization"},
		"model-recipe-lifecycle/recipe-validation":             {"model-recipe-lifecycle/recipe-derivation", "interaction-efficiency/tool-call-propensity-control"},
		"model-recipe-lifecycle/isolated-performance-evidence": {"evidence-routing/evidence-selection", "process-supervision/process-boundary-guard", "interaction-efficiency/bounded-training-evidence"},
		"model-recipe-lifecycle/recipe-evaluation":             {"model-recipe-lifecycle/isolated-performance-evidence"},
		"model-recipe-lifecycle/recipe-activation":             {"model-recipe-lifecycle/recipe-evaluation"},
		"evidence-routing/decision-contract":                   {"model-recipe-lifecycle/recipe-validation"},
		"evidence-routing/evidence-selection":                  {"evidence-routing/decision-contract", "capability-bound-activation/placement-integration", "resource-coverage/reference-relevance-admission"},
		"evidence-routing/counterfactual-replay":               {"evidence-routing/evidence-selection", "causal-execution/causal-projection"},
		"deterministic-rollout/rollout-plan":                   {"model-recipe-lifecycle/recipe-evaluation", "evidence-routing/evidence-selection"},
		"deterministic-rollout/promotion-binding":              {"deterministic-rollout/projection-evidence", "model-recipe-lifecycle/recipe-activation"},
		"live-safety/safety-window":                            {"model-recipe-lifecycle/isolated-performance-evidence"},
		"live-safety/circuit-breaker":                          {"live-safety/safety-window", "deterministic-rollout/projection-evidence", "model-recipe-lifecycle/recipe-activation"},
		"live-safety/atomic-rollback":                          {"live-safety/circuit-breaker", "deterministic-rollout/promotion-binding"},
		"architecture-closure/single-entry-guards":             {"architecture-ratchets/go-only-guard", "interaction-efficiency/tool-call-propensity-control", "evidence-routing/counterfactual-replay", "live-safety/reentry"},
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
		"architecture-ratchets/go-only-guard":                  {"TestSingleCanonicalCampaignPlan", "TestGateAlwaysRunsModernGoRatchet", "TestModernGoRatchetAtClosure", "TestModernGoPublishedCensusMatchesSource", "TestArchitectureRatchetIncludesGoOnlyPolicy", "TestArchitectureRatchetClassifiesCapabilityInvocationBoundaries", "TestRSIMutationEffectApprovalReceiptRatchet", "TestMutationRequiresBoundPreflightInspection", "TestInternalOrchestrationDoesNotInvokeUTCPTransport", "TestCrossDomainCandidateHasSingleAdmissionOwner", "TestCrossDomainPromotionLifecycleHasOneTransitionOwner", "TestEvidenceDriverHasSingleDecisionOwner", "TestDelegatedCapabilityRuntimeConsumesStagedOwners", "TestAgentDelegationStagedSurfaceRetired", "TestPeerCapabilityManualAndPlacement", "TestPeerHTTPJSONStreamInvocationReceipt", "TestCrossLaneCapabilityReuseDoesNotImportRuntimeCode", "TestStagedSurfaceResolvesOnlyCanonicalPlanSteps", "TestStagedSurfaceRetainedClassification", "TestRSIRuntimeIsGoOnly", "TestExternalMechanismProvenanceIsExactAndNoRuntimeDependencyIsAdmitted", "TestDatasetTransformIdentityRejectsCallableAndAdvisoryVersionSemantics"},
		"campaign-concurrency/ready-frontier-leases":           {"TestReadyFrontierLeaseIsolation", "TestMergeAcceptsGatedPrunedCompletion", "TestMergePreservesDependencyOrder"},
		"interaction-efficiency/projection-query-plans":        {"TestRetrievalBuildAndSearchStayWithinBudget"},
		"model-prototypes/prototype-contract":                  {"TestCanonicalComponentDescriptorComparable", "TestSeamAlignmentResidual"},
		"model-prototypes/prototype-admission":                 {"TestExpertParallelCandidateRequiresMeasuredScaleBlockerAndExecutionAuthority", "TestFrontierArchitectureCandidatesRemainIndependentAblations", "TestModelAuthoredCandidatePredictionCalibration"},
		"model-prototypes/prototype-compile":                   {"TestExactComponentRetrievalDeterministic", "TestCompositionCandidateEnumerationRanksByResidualAndFitness", "TestDatasetPipelineStreamsWithinBudgetAndResumesExactStages", "TestExpertPlacementDoesNotChangeDeclaredRouterSemantics"},
		"model-prototypes/prototype-materialization":           {"TestCandidateRealizationFreezesDonorsTrainsAdapter"},
		"model-recipe-lifecycle/isolated-performance-evidence": {"TestExpertParallelEvidenceAttributesTransportCapacityAndDrops"},
		"model-recipe-lifecycle/recipe-evaluation":             {"TestAlignmentResidualBiasAudit", "TestFitnessScoredCompositeSelection", "TestSelectionEmitsAblationGatedPromotion", "TestExpertParallelPrototypeIsBoundedPairedAndExactlyResumable", "TestArchitecturePromotionRequiresPairedBenefitAndNoHiddenRegression"},
		"evidence-routing/decision-contract":                   {"TestCapabilityGapTargetsFromEvalStore"},
		"evidence-routing/evidence-selection":                  {"TestCompositionImprovementLoopClosesUnderBudget", "TestCompositionLoopSaturationStops"},
		"evidence-routing/counterfactual-replay":               {"TestCompositionDriverLearningCurve", "TestCompositionCandidateCalibration"},
		"deterministic-rollout/promotion-binding":              {"TestCompositionAutonomyRatchetGatesOnCurveAndSafety"},
		"architecture-closure/rsi-end-to-end":                  {"TestDeterministicRSIControlPlane", "TestIntegratedRSIRaceRestartBrowserDeviceReleaseMatrix"},
		"architecture-closure/tightness-census":                {"TestFourSpineTightnessCensus", "TestSourcePlanAssumptionsRemainEvidenceOnly", "TestDeferredMechanismsRequireReopenOrRetirementTrigger"},
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
		for _, required := range []string{"complete ordered ready frontier", "distinct worktrees", "gates commits serially", "ancestor gate evidence", "no runtime capability"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("ready-frontier rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "architecture-ratchets/go-only-guard"); found {
		for _, required := range []string{"second live plan file", "historical evidence", "retire_with plan step", "retained classification", "single-owner ratchets", "source modernization", "typed evidence", "compiled Go registrations", "runrecord.StageReceipt", "retire the staged", "http-json-stream", "closed mutation kinds", "action-bound preflight inspection", "must not route through UTCP"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("Go-only ratchet rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "interaction-efficiency/bounded-training-evidence"); found {
		for _, required := range []string{"one reusable Go chunk mechanic", "fact, chunk, and raw-byte budgets", "decode no raw blob", "standalone publication paths"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("bounded training-evidence rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "interaction-efficiency/projection-query-plans"); found {
		for _, required := range []string{"Retrieval construction and exact search", "memory, intermediate-byte, decode", "refuses an unbounded fallback"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("projection query-plan rationale omits consolidated README requirement %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-recipe-lifecycle/isolated-performance-evidence"); found {
		for _, required := range []string{"common bounded training-evidence owner", "assignments, bytes, collectives", "sender and receiver overflow", "cannot duplicate raw telemetry", "partial publication"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("isolated performance-evidence rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-prototypes/prototype-contract"); found {
		for _, required := range []string{"one content-addressed Candidate envelope", "combination of them", "fixed-schema unit-normalized descriptor", "seam alignment residual", "compiled Go registration"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate-spine rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-prototypes/prototype-admission"); found {
		for _, required := range []string{"Model-authored and human-authored proposals", "calibration error", "single-device placement, memory, or throughput evidence", "authorized multi-device target", "independent candidates", "bundled Frontier label"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate admission rationale omits consolidated source-plan requirement %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-prototypes/prototype-compile"); found {
		for _, required := range []string{"exact deterministic retrieval", "enumerates donor, seam, and adapter rungs", "HDC search remains shelved", "memory and intermediate-storage budgets", "durable per-stage checkpoint authority", "transient-versus-permanent failure classification", "router semantics separate from placement"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate compiler rationale omits consolidated source-plan requirement %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-prototypes/prototype-materialization"); found {
		for _, required := range []string{"freezes every donor", "trains only the adapter", "common bounded training-evidence owner", "No raw router or health observation", "plugin-specific publisher"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate materialization rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "interaction-efficiency/tool-call-propensity-control"); found {
		for _, required := range []string{"thin Candidate source and evaluator plugins", "own no recipe or activation path", "common admission, compilation, validation, and evaluation", "alpha zero"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("tool-call propensity rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-recipe-lifecycle/recipe-evaluation"); found {
		for _, required := range []string{"Procrustes or OT seam residual", "post-training bridge parity", "dropped-source and shuffled-source ablations", "one evidence-selected bounded transport", "local reference", "exact checkpoint and retry recovery", "Frontier label is not evidence"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate evaluation rationale omits consolidated source-plan requirement %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "evidence-routing/decision-contract"); found {
		for _, required := range []string{"DriverDecision", "weakest-domain, regression, and frontier-gap targets", "driver remains the decision owner", "budget", "saturation", "intra-model token routing", "parallel task or model-dispatch authority", "evidence only"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("task-routing decision rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-recipe-lifecycle/recipe-activation"); found {
		for _, required := range []string{"one evidence-gated lifecycle", "before any rollout promotion", "supply typed subject validation", "migrate and delete their parallel", "StageReceipt published atomically"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("promotion-spine rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "evidence-routing/evidence-selection"); found {
		for _, required := range []string{"active incumbent plus admitted validated or experimental candidates", "Before promotion", "missing-evidence request", "gap target, exact donor and seam enumeration", "no composition loop or free knob", "after promotion"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("evidence driver rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "deterministic-rollout/promotion-binding"); found {
		for _, required := range []string{"typed promotion request", "adapter derives", "already-established shared lifecycle owner", "positive learning curve", "held predicted-versus-actual calibration", "quiet live-safety evidence", "promotion-specific transition owner"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("promotion binding rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "evidence-routing/counterfactual-replay"); found {
		for _, required := range []string{"fitness delta per compute", "candidate hit rate", "adapter-training cost", "predicted-versus-actual candidate calibration"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("driver replay rationale omits consolidated composition requirement %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "architecture-closure/single-entry-guards"); found {
		for _, required := range []string{"four shared components", "cannot own a second", "documents named promotion", "deterministic internal Go never invokes UTCP transport"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("closure authority rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "architecture-closure/rsi-end-to-end"); found {
		for _, required := range []string{"bounded unattended multi-strategy campaign", "strategy-tagged attempt", "composite Candidate", "exact strategy identity", "semantically relevant typed references", "subject-relevant mechanism", "component and joint ablations", "controller checkpoint resume", "phase-bound training-health evidence", "memory and intermediate-storage budgets", "expert-parallel refusal", "one bounded evidence-selected transport", "http-json-stream inspection capability", "alpha zero whenever mutation", "fact, chunk, and raw-byte budgets", "failed, null, missing, and degraded outcomes", "race, interrupted restart, browser-visible operation state, device selection and failure, and release verification", "without external services"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("end-to-end acceptance omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "architecture-closure/tightness-census"); found {
		for _, required := range []string{"clone, ownership, magic", "copied Marin or other source-plan constants and assumptions", "advisory identity leaks", "competing plan files or free-text future-work triggers", "evidence-bound reopen or retirement triggers"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("tightness census rationale omits consolidated source-plan closure %q", required)
			}
		}
	}

	assertCampaignFrontier(t, document,
		"resource-coverage/strategy-identity-binding",
		"resource-coverage/reference-relevance-admission")
	assertCampaignFrontier(t, document,
		"campaign-concurrency/ready-frontier-leases",
		"interaction-efficiency/bounded-training-evidence",
		"model-prototypes/prototype-contract")
	assertCampaignFrontier(t, document,
		"interaction-efficiency/incremental-context",
		"interaction-efficiency/coalesced-coordination",
		"interaction-efficiency/operator-decisions")
	assertCampaignFrontier(t, document,
		"interaction-efficiency/tool-call-propensity-control",
		"model-prototypes/prototype-admission")
	assertCampaignFrontier(t, document,
		"evidence-routing/counterfactual-replay",
		"model-recipe-lifecycle/isolated-performance-evidence")
	assertCampaignFrontier(t, document,
		"deterministic-rollout/promotion-binding",
		"live-safety/safety-window")
	assertCampaignFrontier(t, document,
		"model-recipe-lifecycle/recipe-evaluation",
		"live-safety/safety-window")
	assertCampaignFrontier(t, document,
		"model-recipe-lifecycle/recipe-activation",
		"deterministic-rollout/rollout-plan")
	assertCampaignFrontier(t, document,
		"deterministic-rollout/promotion-binding",
		"live-safety/circuit-breaker")
}

func assertIntegratedReconciliationSnapshot(t *testing.T, document Plan) {
	t.Helper()
	_, reconciling := retainedCampaignStep(document, "reconcile-integrated-rsi-20260829/do")
	wantIDs := []string{
		"architecture-ratchets/go-only-guard",
		"campaign-concurrency/ready-frontier-leases",
		"interaction-efficiency/bounded-training-evidence",
		"interaction-efficiency/projection-query-plans",
		"interaction-efficiency/head-bound-deltas",
		"interaction-efficiency/incremental-context",
		"interaction-efficiency/coalesced-coordination",
		"interaction-efficiency/operator-decisions",
		"interaction-efficiency/efficiency-gate",
		"interaction-efficiency/tool-call-propensity-control",
		"model-prototypes/prototype-contract",
		"model-prototypes/prototype-admission",
		"model-prototypes/prototype-compile",
		"model-prototypes/prototype-materialization",
		"model-recipe-lifecycle/recipe-derivation",
		"model-recipe-lifecycle/recipe-validation",
		"model-recipe-lifecycle/isolated-performance-evidence",
		"model-recipe-lifecycle/recipe-evaluation",
		"model-recipe-lifecycle/recipe-activation",
		"evidence-routing/decision-contract",
		"evidence-routing/evidence-selection",
		"evidence-routing/counterfactual-replay",
		"deterministic-rollout/rollout-plan",
		"deterministic-rollout/stable-assignment",
		"deterministic-rollout/projection-evidence",
		"deterministic-rollout/promotion-binding",
		"live-safety/safety-window",
		"live-safety/circuit-breaker",
		"live-safety/atomic-rollback",
		"live-safety/reentry",
		"architecture-closure/single-entry-guards",
		"architecture-closure/go-only-guard",
		"architecture-closure/migration-drill",
		"architecture-closure/rsi-end-to-end",
		"architecture-closure/tightness-census",
	}
	want := make(map[string]bool, len(wantIDs))
	for _, id := range wantIDs {
		want[id] = true
	}
	seen := make(map[string]bool, len(wantIDs))
	for _, item := range document.Items {
		if item.ID == "reconcile-integrated-rsi-20260829" {
			continue
		}
		if item.Status != StatusOpen {
			t.Errorf("reconciled item %s status = %q, want %q", item.ID, item.Status, StatusOpen)
		}
		for _, step := range item.Steps {
			id := item.ID + "/" + step.ID
			if !want[id] {
				t.Errorf("reconciled campaign has unexpected retained step %s", id)
				continue
			}
			seen[id] = true
			if step.Status != StatusOpen {
				t.Errorf("reconciled step %s status = %q, want %q", id, step.Status, StatusOpen)
			}
		}
	}
	for _, id := range wantIDs {
		if reconciling && !seen[id] {
			t.Errorf("reconciled campaign lost retained step %s", id)
		}
	}

	layers := retainedCampaignLayers(t, document, map[string]bool{"reconcile-integrated-rsi-20260829": true})
	if allCampaignStepsRetained(document,
		"campaign-concurrency/ready-frontier-leases",
		"interaction-efficiency/bounded-training-evidence",
		"model-prototypes/prototype-contract") {
		maxWidth := 0
		for _, layer := range layers {
			maxWidth = max(maxWidth, len(layer))
		}
		if maxWidth < 3 {
			t.Errorf("reconciled campaign maximum ready-layer width = %d, want at least 3", maxWidth)
		}
	}
	assertSameCampaignLayerIfRetained(t, document, layers,
		"campaign-concurrency/ready-frontier-leases",
		"interaction-efficiency/bounded-training-evidence",
		"model-prototypes/prototype-contract")
	assertSameCampaignLayerIfRetained(t, document, layers,
		"interaction-efficiency/incremental-context",
		"interaction-efficiency/coalesced-coordination",
		"interaction-efficiency/operator-decisions")
	assertSameCampaignLayerIfRetained(t, document, layers,
		"model-recipe-lifecycle/recipe-evaluation",
		"live-safety/safety-window")
	assertSameCampaignLayerIfRetained(t, document, layers,
		"deterministic-rollout/promotion-binding",
		"live-safety/circuit-breaker")
}

func allCampaignStepsRetained(document Plan, ids ...string) bool {
	for _, id := range ids {
		if _, found := retainedCampaignStep(document, id); !found {
			return false
		}
	}
	return true
}

func assertSameCampaignLayerIfRetained(t *testing.T, document Plan, layers [][]string, ids ...string) {
	t.Helper()
	if !allCampaignStepsRetained(document, ids...) {
		return
	}
	assertSameCampaignLayer(t, layers, ids...)
}

func retainedCampaignLayers(t *testing.T, document Plan, excludedItems map[string]bool) [][]string {
	t.Helper()
	steps := map[string]Step{}
	for _, item := range document.Items {
		if excludedItems[item.ID] {
			continue
		}
		for _, step := range item.Steps {
			steps[item.ID+"/"+step.ID] = step
		}
	}
	indegree := make(map[string]int, len(steps))
	dependents := map[string][]string{}
	for id, step := range steps {
		indegree[id] = 0
		for _, dependency := range step.DependsOn {
			if _, retained := steps[dependency]; !retained {
				continue
			}
			indegree[id]++
			dependents[dependency] = append(dependents[dependency], id)
		}
	}
	ready := make([]string, 0, len(steps))
	for id, count := range indegree {
		if count == 0 {
			ready = append(ready, id)
		}
	}
	slices.Sort(ready)
	layers := make([][]string, 0)
	visited := 0
	for len(ready) > 0 {
		layer := slices.Clone(ready)
		layers = append(layers, layer)
		visited += len(layer)
		next := make([]string, 0)
		for _, id := range layer {
			for _, dependent := range dependents[id] {
				indegree[dependent]--
				if indegree[dependent] == 0 {
					next = append(next, dependent)
				}
			}
		}
		slices.Sort(next)
		ready = next
	}
	if visited != len(steps) {
		t.Fatalf("retained campaign DAG visited %d of %d steps", visited, len(steps))
	}
	return layers
}

func assertSameCampaignLayer(t *testing.T, layers [][]string, ids ...string) {
	t.Helper()
	positions := map[string]int{}
	for layer, members := range layers {
		for _, id := range members {
			positions[id] = layer
		}
	}
	if len(ids) == 0 {
		return
	}
	want, found := positions[ids[0]]
	if !found {
		t.Errorf("campaign layer omits %s", ids[0])
		return
	}
	for _, id := range ids[1:] {
		got, found := positions[id]
		if !found {
			t.Errorf("campaign layer omits %s", id)
			continue
		}
		if got != want {
			t.Errorf("campaign steps are not simultaneously ready: %s is layer %d, %s is layer %d", ids[0], want, id, got)
		}
	}
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
	if err := Validate(document); err != nil {
		t.Fatalf("validate campaign plan: %v", err)
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
