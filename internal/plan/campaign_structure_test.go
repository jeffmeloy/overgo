package plan

import (
	"os"
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
	competing, err := CompetingPlans(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range competing {
		t.Errorf("competing live plan file %s; %s is the sole plan authority", path, filepath.ToSlash(Path))
	}

	// The .git-marker rule, pinned against a synthetic tree: a nested
	// checkout (marker file, as worktrees use, or marker directory) hides
	// its plan.json, while a plain nested directory's plan.json competes.
	synthetic := t.TempDir()
	write := func(relative, content string) {
		t.Helper()
		path := filepath.Join(synthetic, filepath.FromSlash(relative))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.ToSlash(Path), "{}")
	write(".claude/worktrees/lane/.git", "gitdir: elsewhere")
	write(".claude/worktrees/lane/docs/plan.json", "{}")
	write("nested-checkout/.git/HEAD", "ref: refs/heads/main")
	write("nested-checkout/docs/plan.json", "{}")
	write("plain/docs/plan.json", "{}")
	competing, err = CompetingPlans(synthetic)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(competing, []string{"plain/docs/plan.json"}) {
		t.Fatalf("synthetic competing plans = %v, want only the plain nested copy", competing)
	}
}

// groundedDoctrineDecisions enumerates the shared-spine decisions the
// grounded campaign's doctrine must state verbatim.
func groundedDoctrineDecisions() []string {
	return []string{
		"sole live execution plan", "imported branch plans", "four shared components",
		"cross-domain candidate spine", "promotion and rollback lifecycle",
		"evidence-driven budget and stop driver", "transport-neutral UTCP",
		"deterministic in-process orchestration", "shared sequential-control owner",
		"common-cause or insufficient evidence cannot request", "first capability proof",
		"supervised non-serving verified state", "built from its persisted trace",
		"Grounded capability learning", "rebuildable projections",
		"structural hybrid retrieval", "evidence-dominant routing", "Markdown skill",
	}
}

// TestRSICampaignRatchetAndParallelStructure pins the owner decision that
// architecture ratchets precede new RSI construction, one bounded supervised
// capability proof precedes optimization and autonomy-scale substrate, a leased
// ready-frontier then turns DAG slack into conflict-bound worktrees, and
// dependencies name exact producer steps. Completed rows leave plan.json, so
// row-specific assertions are conditional while the ratchets remain active.
func TestRSICampaignRatchetAndParallelStructure(t *testing.T) {
	document := loadCampaignPlan(t)
	if strings.HasPrefix(document.Campaign, "Structural GUI redesign") || strings.HasPrefix(document.Campaign, "GUI capability roadmap:") {
		assertConversationGUICampaign(t, document)
		return
	}
	if strings.Contains(document.Campaign, "audio.cpp") {
		assertAudioCapabilityCampaign(t, document)
		return
	}
	// Each campaign pins its own structure ratchet, keyed by campaign
	// identity the way the grounded doctrine binding already is: the
	// integrated-RSI snapshot binds to the RSI control-plane campaign, and
	// the validation freeze campaign that replaced it after the RSI rows
	// completed binds to its own step whitelist below.
	if strings.Contains(document.Campaign, "Professional operator GUI") {
		assertProfessionalGUICampaignSnapshot(t, document)
		return
	}
	if strings.Contains(document.Campaign, "Integrated validation") {
		assertValidationCampaignSnapshot(t, document)
		return
	}
	assertIntegratedReconciliationSnapshot(t, document)
	// The shared-spine doctrine binds once the grounded campaign is adopted;
	// a first-parent-target merge lands under the target's prior campaign,
	// and the adoption commit that renames the campaign re-arms the ratchet.
	groundedCampaign := strings.Contains(document.Campaign, "grounded capability")
	requiredDoctrine := []string{}
	if groundedCampaign {
		requiredDoctrine = groundedDoctrineDecisions()
	}
	for _, required := range requiredDoctrine {
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
		"campaign-concurrency/ready-frontier-leases":           {"architecture-closure/go-only-guard"},
		"interaction-efficiency/bounded-training-evidence":     {"storage-projections/atomic-publication", "resource-coverage/fitness-integration", "redundancy-interaction-baseline/interaction-traces", "architecture-closure/go-only-guard"},
		"interaction-efficiency/projection-query-plans":        {"interaction-efficiency/bounded-training-evidence"},
		"interaction-efficiency/head-bound-deltas":             {"interaction-efficiency/projection-query-plans", "storage-operational-boundary/transition-boundary"},
		"interaction-efficiency/incremental-context":           {"interaction-efficiency/head-bound-deltas", "stimulus-reconciliation/stable-cursors", "capability-bound-activation/placement-integration"},
		"interaction-efficiency/coalesced-coordination":        {"interaction-efficiency/head-bound-deltas", "webhook-ingress/workflow-dispatch"},
		"interaction-efficiency/operator-decisions":            {"interaction-efficiency/head-bound-deltas", "causal-execution/causal-projection"},
		"interaction-efficiency/efficiency-gate":               {"interaction-efficiency/incremental-context", "interaction-efficiency/coalesced-coordination", "interaction-efficiency/operator-decisions", "live-safety/safety-window"},
		"tool-propensity/tool-call-propensity-control":         {"interaction-efficiency/efficiency-gate", "model-prototypes/prototype-contract"},
		"model-prototypes/prototype-contract":                  {"architecture-ratchets/go-only-guard", "capability-bound-activation/capability-identity", "causal-execution/causal-context"},
		"model-prototypes/prototype-admission":                 {"model-prototypes/prototype-contract", "resource-coverage/reference-relevance-admission"},
		"model-prototypes/prototype-compile":                   {"model-prototypes/prototype-admission"},
		"model-prototypes/prototype-materialization":           {"evidence-routing/evidence-selection"},
		"model-recipe-lifecycle/recipe-derivation":             {"model-prototypes/prototype-materialization"},
		"model-recipe-lifecycle/recipe-validation":             {"model-recipe-lifecycle/recipe-derivation"},
		"model-recipe-lifecycle/isolated-performance-evidence": {"model-recipe-lifecycle/recipe-validation", "process-supervision/process-boundary-guard"},
		"model-recipe-lifecycle/recipe-evaluation":             {"model-recipe-lifecycle/isolated-performance-evidence"},
		"model-recipe-lifecycle/recipe-activation":             {"architecture-ratchets/go-only-guard", "model-prototypes/prototype-contract"},
		"evidence-routing/decision-contract":                   {"architecture-ratchets/go-only-guard", "resource-coverage/fitness-integration", "resource-coverage/reference-relevance-admission"},
		"evidence-routing/evidence-selection":                  {"evidence-routing/decision-contract", "model-prototypes/prototype-compile", "capability-bound-activation/placement-integration", "resource-coverage/reference-relevance-admission"},
		"evidence-routing/counterfactual-replay":               {"evidence-routing/evidence-selection", "causal-execution/causal-projection", "architecture-closure/go-only-guard"},
		"supervised-composite/first-win":                       {"model-recipe-lifecycle/recipe-evaluation", "model-recipe-lifecycle/recipe-activation"},
		"composition-expansion/alignment-adapter":              {"architecture-closure/go-only-guard"},
		"composition-expansion/search-ratchet":                 {"composition-expansion/alignment-adapter", "evidence-routing/counterfactual-replay", "live-safety/safety-window"},
		"advanced-domain-plugins/realization":                  {"architecture-closure/go-only-guard"},
		"advanced-domain-plugins/evidence":                     {"advanced-domain-plugins/realization", "live-safety/safety-window"},
		"outward-capabilities/peer-invocation":                 {"architecture-closure/go-only-guard"},
		"deterministic-rollout/rollout-plan":                   {"model-recipe-lifecycle/recipe-evaluation", "evidence-routing/evidence-selection", "supervised-composite/first-win", "live-safety/safety-window"},
		"deterministic-rollout/promotion-binding":              {"deterministic-rollout/projection-evidence", "model-recipe-lifecycle/recipe-activation"},
		"live-safety/safety-window":                            {"model-recipe-lifecycle/isolated-performance-evidence", "architecture-closure/go-only-guard"},
		"live-safety/circuit-breaker":                          {"live-safety/safety-window", "deterministic-rollout/projection-evidence", "model-recipe-lifecycle/recipe-activation"},
		"live-safety/atomic-rollback":                          {"live-safety/circuit-breaker", "deterministic-rollout/promotion-binding"},
		"architecture-closure/single-entry-guards":             {"architecture-ratchets/go-only-guard", "supervised-composite/first-win"},
		"architecture-closure/go-only-guard":                   {"architecture-closure/single-entry-guards"},
		"architecture-closure/migration-drill":                 {"campaign-concurrency/ready-frontier-leases", "tool-propensity/tool-call-propensity-control", "evidence-routing/counterfactual-replay", "live-safety/reentry", "composition-expansion/search-ratchet", "advanced-domain-plugins/evidence", "outward-capabilities/peer-invocation"},
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
		"architecture-ratchets/go-only-guard":                  {"TestSingleCanonicalCampaignPlan", "TestGateAlwaysRunsModernGoRatchet", "TestModernGoRatchetAtClosure", "TestModernGoPublishedCensusMatchesSource", "TestArchitectureRatchetIncludesGoOnlyPolicy", "TestArchitectureRatchetClassifiesCapabilityInvocationBoundaries", "TestRSIMutationEffectApprovalReceiptRatchet", "TestMutationRequiresBoundPreflightInspection", "TestInternalOrchestrationDoesNotInvokeUTCPTransport", "TestCrossDomainCandidateHasSingleAdmissionOwner", "TestCrossDomainPromotionLifecycleHasOneTransitionOwner", "TestEvidenceDriverHasSingleDecisionOwner", "TestStagedSurfaceResolvesOnlyCanonicalPlanSteps", "TestStagedSurfaceRetainedClassification", "TestRSIRuntimeIsGoOnly", "TestExternalMechanismProvenanceIsExactAndNoRuntimeDependencyIsAdmitted", "TestDatasetTransformIdentityRejectsCallableAndAdvisoryVersionSemantics"},
		"campaign-concurrency/ready-frontier-leases":           {"TestReadyFrontierLeaseIsolation", "TestMergeAcceptsGatedPrunedCompletion", "TestMergePreservesDependencyOrder"},
		"interaction-efficiency/projection-query-plans":        {"TestRetrievalBuildAndSearchStayWithinBudget"},
		"model-prototypes/prototype-compile":                   {"TestCompatiblePassthroughAndTaskArithmetic"},
		"model-prototypes/prototype-materialization":           {"TestPersistedDecisionMaterializationCommand", "TestCrossDomainTrialMaterializesThroughOnePublicationOwner", "TestPrototypeMaterializationClosure", "TestStreamingTaskArithmetic", "TestStreamingRefusesSameGeometrySourceShardByteSubstitution", "TestOfflineArtifactGenerationTaskArithmetic"},
		"supervised-composite/first-win":                       {"TestSupervisedCompositeFirstWin"},
		"composition-expansion/alignment-adapter":              {"TestCanonicalComponentDescriptorComparable", "TestSeamAlignmentResidual", "TestExactComponentRetrievalDeterministic", "TestCompositionCandidateEnumerationRanksByResidualAndFitness", "TestCandidateRealizationFreezesDonorsTrainsAdapter", "TestAlignmentResidualBiasAudit", "TestFitnessScoredCompositeSelection", "TestSelectionEmitsAblationGatedPromotion"},
		"composition-expansion/search-ratchet":                 {"TestCapabilityGapTargetsFromEvalStore", "TestCompositionImprovementLoopClosesUnderBudget", "TestCompositionLoopSaturationStops", "TestCompositionDriverLearningCurve", "TestCompositionCandidateCalibration", "TestCompositionAutonomyRatchetGatesOnCurveAndSafety", "TestCompositionAutonomyRatchetIgnoresCommonCause"},
		"advanced-domain-plugins/realization":                  {"TestExpertParallelCandidateRequiresMeasuredScaleBlockerAndExecutionAuthority", "TestFrontierArchitectureCandidatesRemainIndependentAblations", "TestModelAuthoredCandidatePredictionCalibration", "TestDatasetPipelineStreamsWithinBudgetAndResumesExactStages", "TestExpertPlacementDoesNotChangeDeclaredRouterSemantics", "TestDenseToMoEEvidenceScreen", "TestDenseToMoEExpertInventory", "TestDenseToMoEPromotionRefusal"},
		"advanced-domain-plugins/evidence":                     {"TestIsolatedTrainingEvidenceUsesBoundedOwner", "TestExpertParallelEvidenceAttributesTransportCapacityAndDrops", "TestExpertParallelPrototypeIsBoundedPairedAndExactlyResumable", "TestArchitecturePromotionRequiresPairedBenefitAndNoHiddenRegression"},
		"live-safety/safety-window":                            {"TestSequentialControlPhaseICalibratesBeforePhaseIIEnforces", "TestSequentialControlCombinedRulesUseOrderedCorrelatedReplay", "TestSequentialControlReportsRealizedARLAndDetectionDelay", "TestSequentialControlCommonCauseIsNoAction", "TestSequentialControlHasSingleOwner"},
		"outward-capabilities/peer-invocation":                 {"TestDelegatedCapabilityRuntimeConsumesStagedOwners", "TestAgentDelegationStagedSurfaceRetired", "TestPeerCapabilityManualAndPlacement", "TestPeerHTTPJSONStreamInvocationReceipt", "TestCrossLaneCapabilityReuseDoesNotImportRuntimeCode", "TestRemoteRecipeCapabilityRequiresExactUTCPManual"},
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
	assertCampaignOrder(t, document, "evidence-routing", "supervised-composite")
	assertCampaignOrder(t, document, "supervised-composite", "composition-expansion")
	assertCampaignOrder(t, document, "supervised-composite", "advanced-domain-plugins")
	assertCampaignOrder(t, document, "supervised-composite", "outward-capabilities")
	assertCampaignOrder(t, document, "outward-capabilities", "deterministic-rollout")
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
		for _, required := range []string{"supervised first-win trace", "complete ordered post-proof ready frontier", "distinct worktrees", "gates commits serially", "ancestor gate evidence", "no runtime capability"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("ready-frontier rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "architecture-ratchets/go-only-guard"); found {
		for _, required := range []string{"second live plan file", "historical evidence", "retire_with plan step", "retained classification", "single-owner ratchets", "source modernization", "typed evidence", "compiled Go registrations", "runrecord.StageReceipt", "closed mutation kinds", "action-bound preflight inspection", "must not route through UTCP", "post-proof outward-capability step"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("Go-only ratchet rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "interaction-efficiency/bounded-training-evidence"); found {
		for _, required := range []string{"supervised composite first win", "representative execution and interaction trace", "one reusable Go chunk mechanic", "fact, chunk, and raw-byte budgets", "decode no raw blob", "standalone publication paths", "no efficiency claim may precede"} {
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
		for _, required := range []string{"one supervised process", "action-bound preflight inspection", "argument-bound HumanDecision", "admitted-running-terminal StageReceipt", "no UTCP transport", "partial publication", "post-proof plugins"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("isolated performance-evidence rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-prototypes/prototype-contract"); found {
		for _, required := range []string{"one content-addressed Candidate envelope", "combination of them", "compiled Go registration", "post-proof plugins", "not fields or authorities"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate-spine rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-prototypes/prototype-admission"); found {
		for _, required := range []string{"separate non-authorizing publication", "exact canonical bytes and dependency lineage", "one production call site", "unsupported domain fails closed", "later DriverDecision owner records refusal", "cannot construct a candidate"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate admission rationale omits consolidated source-plan requirement %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-prototypes/prototype-compile"); found {
		for _, required := range []string{"narrow composition admission adapter", "same-base task arithmetic", "same base ModelDefinition", "compatible tensor inventory", "DenseToMoE", "post-proof plugins", "cannot create its own candidate, admission"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate compiler rationale omits consolidated source-plan requirement %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-prototypes/prototype-materialization"); found {
		for _, required := range []string{"first composition plugin streams", "same-base task-arithmetic plan", "content-addressed weights", "deterministic rerun evidence", "plugin-specific publisher", "behind the supervised proof"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate materialization rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "tool-propensity/tool-call-propensity-control"); found {
		for _, required := range []string{"thin Candidate source and evaluator plugins", "own no recipe or activation path", "common admission, compilation, validation, and evaluation", "alpha zero"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("tool-call propensity rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-recipe-lifecycle/recipe-evaluation"); found {
		for _, required := range []string{"production incumbent", "parent specialists", "dropped-delta ablations", "independent promotion split", "multidimensional fitness", "measurable Pareto improvement", "records refusal", "cannot mutate"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("candidate evaluation rationale omits consolidated source-plan requirement %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "evidence-routing/decision-contract"); found {
		for _, required := range []string{"DriverDecision", "budget", "saturation", "independent of any particular candidate compiler", "intra-model token routing", "parallel task or model-dispatch authority", "evidence only"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("task-routing decision rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "model-recipe-lifecycle/recipe-activation"); found {
		for _, required := range []string{"one evidence-gated lifecycle", "before any supervised verification", "supply typed subject validation", "migrate and delete parallel", "StageReceipt published atomically", "non-serving-verified path", "preserve the production active alias", "unattended authority widening"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("promotion-spine rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "evidence-routing/evidence-selection"); found {
		for _, required := range []string{"active incumbent plus admitted candidates", "pre-materialization eligible state", "Before supervised verification", "missing-evidence request", "first local same-base composition", "Cross-boundary resolution is a post-proof extension", "no domain loop or free knob", "after production promotion"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("evidence driver rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "deterministic-rollout/promotion-binding"); found {
		for _, required := range []string{"supervised milestone verifies a non-serving candidate", "production rollout activation", "typed promotion request", "adapter derives", "already-established shared lifecycle owner", "Plugin-specific learning, calibration, and safety requirements", "promotion-specific transition owner"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("promotion binding rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "evidence-routing/counterfactual-replay"); found {
		for _, required := range []string{"Only after the supervised trace exists", "identical inputs, budgets, stop conditions", "without granting live dispatch", "composition search and calibration extend this owner"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("driver replay rationale omits post-proof boundary %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "supervised-composite/first-win"); found {
		for _, required := range []string{"same-base task-arithmetic candidate", "production incumbent", "promotion-split evaluation", "dropped-delta ablations", "exact strategy identity", "strategy-tagged attempt", "measurable Pareto improvement", "human-watched and non-serving", "action-bound preflight evidence", "argument-bound HumanDecision", "admitted, running, and terminal StageReceipt", "production active alias", "unattended AutomationPolicyLifecycle remain unchanged", "cold-reopen", "representative execution and interaction trace", "No live traffic"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("supervised first-win rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "composition-expansion/alignment-adapter"); found {
		for _, required := range []string{"Only after the reproducible held-out same-base win", "fixed-schema unit-normalized component descriptors", "exact deterministic retrieval", "Procrustes or OT seam evidence", "Freeze donors", "train only the interface adapter", "dropped-source", "shuffled-source", "not a normalized-catalog service", "HDC"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("composition alignment rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "composition-expansion/search-ratchet"); found {
		for _, required := range []string{"shared DriverDecision", "held-out fitness delta per compute", "candidate hit rate", "adapter cost", "predicted-versus-actual calibration", "positive held-out learning curve", "beneficial special-cause evidence", "common cause and insufficient evidence preserve", "no composition loop", "shared rollout and promotion adapter"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("composition search rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "advanced-domain-plugins/realization"); found {
		for _, required := range []string{"After the simple composite proves the shared loop", "registered Go Candidate source and realization plugins", "DenseToMoE is a later architecture-change rung", "cannot self-promote", "measured single-device", "independent candidates", "introduce no distributed platform or scheduler"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("advanced domain realization rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "advanced-domain-plugins/evidence"); found {
		for _, required := range []string{"already-proven isolated process and evaluator owners", "post-proof bounded training-evidence mechanic", "sender and receiver overflow", "refuse extrapolation or regime mixing", "one bounded transport against the local reference", "cannot duplicate raw telemetry", "partial publication refuses"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("advanced domain evidence rationale omits %q", required)
			}
		}
	}
	if step, found := retainedCampaignStep(document, "outward-capabilities/peer-invocation"); found {
		for _, required := range []string{"Complete and retire the staged", "effect-classed UTCP manual", "instead of importing or duplicating runtime code", "http-json-stream dispatch", "shared StageReceipt", "Remote recipe validation", "mutation stays alpha zero", "Deterministic in-process orchestration", "no second registry"} {
			if !strings.Contains(step.Rationale, required) {
				t.Errorf("outward capability rationale omits %q", required)
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
		"model-prototypes/prototype-contract",
		"evidence-routing/decision-contract")
	assertCampaignFrontier(t, document,
		"campaign-concurrency/ready-frontier-leases",
		"interaction-efficiency/bounded-training-evidence",
		"evidence-routing/counterfactual-replay",
		"live-safety/safety-window",
		"composition-expansion/alignment-adapter",
		"advanced-domain-plugins/realization",
		"outward-capabilities/peer-invocation")
	assertCampaignFrontier(t, document,
		"deterministic-rollout/rollout-plan",
		"composition-expansion/search-ratchet",
		"advanced-domain-plugins/evidence")
	assertCampaignFrontier(t, document,
		"interaction-efficiency/incremental-context",
		"interaction-efficiency/coalesced-coordination",
		"interaction-efficiency/operator-decisions")
	assertCampaignFrontier(t, document,
		"deterministic-rollout/promotion-binding",
		"live-safety/circuit-breaker")
}

// assertAudioCapabilityCampaign admits the branch-specific audio campaign
// without weakening its structure to the historical RSI or validation
// whitelists. Once the campaign adopts the repeated master-sync workflow, the
// plan itself must retain one exact synchronization row before every work row.
// Between the gated synchronization commit and its paired work commit, the
// completed sync row is pruned and the leading work row retains its exact
// dependency on that completion authority.
func assertAudioCapabilityCampaign(t *testing.T, document Plan) {
	t.Helper()
	for _, required := range []string{
		"pinned behavioral oracle", "never a linked runtime", "exact stop and resume", "pre/post WER",
	} {
		if !strings.Contains(document.Doctrine, required) {
			t.Errorf("audio campaign doctrine omits %q", required)
		}
	}
	workflow := strings.Contains(document.Doctrine, "Every retained work row executes as one repeated automation cycle.")
	paired := 0
	for _, item := range document.Items {
		if strings.HasPrefix(item.ID, "merge-") || item.ID == "workflow-cycle" {
			continue
		}
		for _, step := range item.Steps {
			if step.Status != StatusOpen || strings.TrimSpace(step.Verify) == "" {
				t.Errorf("audio campaign step %s/%s lacks an open machine-checked verifier", item.ID, step.ID)
			}
		}
		if !workflow {
			continue
		}
		steps := item.Steps
		if len(steps)%2 != 0 {
			work := steps[0]
			wantDependency := []string{item.ID + "/sync-" + work.ID}
			if strings.HasPrefix(work.ID, "sync-") || !slices.Equal(work.DependsOn, wantDependency) {
				t.Errorf("audio workflow item %s has invalid leading post-sync work row %s with dependencies %v, want %v", item.ID, work.ID, work.DependsOn, wantDependency)
				continue
			}
			steps = steps[1:]
		}
		for index := 0; index < len(steps); index += 2 {
			sync, work := steps[index], steps[index+1]
			if sync.ID != "sync-"+work.ID {
				t.Errorf("audio workflow pair %s[%d] = %s then %s", item.ID, index, sync.ID, work.ID)
			}
			wantVerify := "git merge-base --is-ancestor master HEAD && go run ./cmd/plan -status"
			// The operator deferred master integration for this one CPU row
			// until the tokenizer and continuation-scoring fixes are verified.
			if item.ID == "audio-primitives" && sync.ID == "sync-decode-inspect" &&
				strings.Contains(document.Doctrine, "The decode-inspect step proceeds on audio baseline cda086d8") {
				wantVerify = "git merge-base --is-ancestor cda086d8 HEAD && go run ./cmd/plan -status"
			}
			if sync.Verify != wantVerify {
				t.Errorf("audio workflow sync %s/%s has verifier %q", item.ID, sync.ID, sync.Verify)
			}
			wantDependency := []string{item.ID + "/" + sync.ID}
			if !slices.Equal(work.DependsOn, wantDependency) {
				t.Errorf("audio workflow work %s/%s dependencies = %v, want %v", item.ID, work.ID, work.DependsOn, wantDependency)
			}
			paired++
		}
	}
	if workflow && paired == 0 {
		t.Error("audio workflow doctrine has no retained sync/work pairs")
	}
}

// The conversation redesign supersedes the old GUI snapshot. Validate its
// graph and worktree boundaries without treating unrelated RSI row IDs as
// the authority for newly planned GUI capabilities.
func assertConversationGUICampaign(t *testing.T, document Plan) {
	t.Helper()
	if err := Validate(document); err != nil {
		t.Fatal(err)
	}
	if document.Lane != "professional_overgo_gui" {
		t.Errorf("conversation GUI campaign has unexpected lane %q", document.Lane)
	}
	for _, item := range document.Items {
		if item.Owner != "" && item.Owner != document.Lane && item.Owner != "master" && item.Owner != "operator" {
			t.Errorf("GUI item %s has an unexpected owner %q", item.ID, item.Owner)
		}
		if strings.HasPrefix(item.ID, "merge-") && !preparedMergeBoundary(item) {
			t.Errorf("invalid prepared GUI merge boundary %+v", item)
		}
	}
	for _, required := range []string{"flat", "conversation", "mobile", "settings", "status"} {
		if !strings.Contains(strings.ToLower(document.Doctrine), required) {
			t.Errorf("conversation GUI doctrine omits %q", required)
		}
	}
}

// assertProfessionalGUICampaignSnapshot pins the earlier GUI campaign's
// declared steps and thin-client, retained-workbench and UNAVAILABLE rules.
func assertProfessionalGUICampaignSnapshot(t *testing.T, document Plan) {
	t.Helper()
	wantIDs := []string{
		"gui-bootstrap/author-plan",
		// Simplification precedes capability (owner rule 2026-09-04): the
		// front page is built on one shell, one composer, one capability
		// source and one route table, each row a measured ratchet.
		"gui-simplify/one-shell",
		"gui-simplify/one-composer",
		"gui-simplify/one-capability-source",
		"gui-simplify/one-route-table",
		"gui-shell/front-page",
		"gui-shell/model-switch",
		"gui-conversations/durable-sessions",
		"gui-multimodal/import",
		"gui-multimodal/dynamic-inference",
		"gui-workbench/inspect-turn",
		"gui-workbench/agent-in-thread",
		"gui-workbench/operations-strip",
		"gui-quality/accessibility-responsive",
		"gui-quality/acceptance-lane",
		"gui-library/do",
		"gui-docs/readme-launcher-manifest",
		"gui-closeout/closeout",
		"gui-serve-projectors/do",
		"gui-register-projectors/do",
		"gui-library-fixes/do",
		"gui-generation-workspace/do",
		"gui-generation-declarations/do",
		"gui-media-roundtrip/do",
		"gui-journey-modalities/do",
		"gui-multimodal-closeout/do",
		"merge-519a2690db1a/do",
		"video-from-prompt/do",
		"merge-cd5ce9e66609/do",
		"video-edit-from-clip/do",
		"merge-e1a9cc0deeac/do",
		"server-import-budget/do",
		"gui-clip-control/do",
		"gui-vqa-executor/do",
		"openrouter-relay/do",
		"openrouter-page/do",
		"openrouter-evals/do",
		"audio-staged-surface/do",
		"webui-style-consistency/do",
		"integrate-to-master/do",
		"merge-2553fd7d28ba/do",
		"merge-4e0b9a670603/do",
		"merge-b3977f8cff14/do",
		"merge-7d3ca937764c/do",
		// Rows retained from the master plan at the lane base (9dce4fca):
		// the plan authority protects every identity committed at the
		// protected revision, so the lane keeps them after its own rows and
		// never works them here; they belong to the master worktree's
		// validation campaign and drop out of this lane at merge.
		"long-context-collapse/do",
		"gemma-12b-accuracy/do",
		"benchmark-completion/mmlu-pro-pass",
		"benchmark-completion/published-comparison",
		"model-regression-gate/do",
		"model-regression-baseline/do",
		"modality-verification/capability-census",
		"modality-verification/declared-smoke-expectations",
		"modality-verification/text-and-vision",
		"modality-verification/image-and-video",
		"modality-verification/speech-ocr-tabular-forecast",
		"modality-verification/media-report",
		"failure-recovery/rollout-plan-author",
		"failure-recovery/control-plane-drills",
		"failure-recovery/gate-recovery-drill",
		"simplify-prefill-paths/do",
		"simplify-command-surface/do",
		"simplify-execution-core/do",
		"simplify-checkpoint-locations/do",
		"simplify-capability-report/do",
		"campaign-closeout/closeout",
	}
	want := make(map[string]bool, len(wantIDs))
	for _, id := range wantIDs {
		want[id] = true
	}
	for _, item := range document.Items {
		for _, step := range item.Steps {
			id := item.ID + "/" + step.ID
			if !want[id] {
				t.Errorf("professional GUI campaign has undeclared step %s; the lane admits only the rows its plan names", id)
				continue
			}
			if strings.TrimSpace(step.Verify) == "" {
				t.Errorf("professional GUI step %s has no machine-checked verify", id)
			}
		}
	}
	for _, required := range []string{"thin client", "UNAVAILABLE", "plan -setverify", "professional_overgo_gui", "keeps every capability it has today"} {
		if !strings.Contains(document.Doctrine, required) {
			t.Errorf("professional GUI campaign doctrine omits %q", required)
		}
	}
}

// assertValidationCampaignSnapshot pins the validation-only freeze campaign's
// declared steps, machine-checked verifiers and audit rules.
func assertValidationCampaignSnapshot(t *testing.T, document Plan) {
	t.Helper()
	wantIDs := []string{
		"campaign-bootstrap/author-validation-plan",
		"durability/publish-history",
		"durability/snapshot-store",
		"freeze-capture/capture-identity",
		"hermetic-foundation/hermetic-suite",
		"hermetic-foundation/repeat-agreement",
		"hermetic-foundation/coverage-floors",
		"hermetic-foundation/race-lane",
		"hermetic-foundation/browser-workbench",
		"hermetic-foundation/release-build",
		"hermetic-foundation/store-drills",
		// Fix rows injected under the owner's correct-as-you-go directive
		// (2026-08-31) are declared here when a drill surfaces a defect:
		// the store drill exposed the rebuild/compaction refusal, the
		// stranded-blob swap, and the missing lifecycle evidence lineage.
		"store-lifecycle-admission/do",
		"inference-e2e/verified-matrix",
		"inference-e2e/serving-operations",
		"training-e2e/route-drills",
		"training-e2e/preference-objectives",
		"benchmark-routing/pinned-benchmarks",
		// The benchmark drill (2026-09-01/02) surfaced that no model's
		// provided sampling reached the runtime and that every evaluate
		// worker compiles the whole suite catalog before a two-second
		// suite; both are corrected in-campaign under the same directive.
		"model-generation-settings/do",
		"evaluate-family-compile/do",
		"benchmark-routing/resource-quality-fitness",
		"benchmark-routing/routing-replay",
		"failure-recovery/interruption-drills",
		"failure-recovery/control-plane-drills",
		"failure-recovery/gate-recovery-drill",
		// Phase 2 (owner directive 2026-09-02): repository cleanup, the
		// remaining benchmark suites except MATH, modality verification of
		// every registered model, the generated media report, the fixes the
		// phase-1 drills surfaced, and the close-out.
		"phase2-bootstrap/author-phase2-plan",
		"repo-cleanup/scrap-and-drill-copies",
		"repo-cleanup/stash-reconciliation",
		"repo-cleanup/routing-resource-size",
		"repo-cleanup/findings-disposition",
		"resource-tolerance/do",
		"audit-language/do",
		"choice-scoring-fix/do",
		"group-metric-names/do",
		"native-decode-span/do",
		"native-tensor-core-prefill/do",
		"chat-choice-options/do",
		"gate-authority-memoization/do",
		"chat-protocol-guards/do",
		"graph-exec-staleness/do",
		"decode-session-source-capacity/do",
		"report-wall-time/do",
		"report-throughput/do",
		"benchmark-token-budget/do",
		"worker-budget/do",
		"chat-answer-opener/do",
		"causal-prefill-attention/do",
		"evaluator-batching/do",
		"bpe-merge-heap/do",
		"buffer-pool-classes/do",
		"ifeval-chat-shaping/do",
		"merged-boundary-candidates/do",
		"quantized-prefill-gemm/do",
		"quantized-prefill-f16/do",
		"fp8-prefill-f16/do",
		"long-form-verification/do",
		"readme-2026-09-04/do",
		"long-context-collapse/do",
		"decode-attention-per-key-cost/do",
		"device-memory-retention/do",
		"gemma-12b-accuracy/do",
		"model-regression-fingerprints/do",
		"model-regression-gate/do",
		// Split the existing guard row at its inventory and selection boundaries.
		"model-regression-gate/coverage-inventory",
		"model-regression-gate/coverage-selection",
		"model-regression-gate/coverage-acquisition",
		// Accept the completed model cohorts; retain full-catalog acquisition.
		"e4b-fp8-evidence-refresh/do",
		"model-regression-gate/coverage-12b-fp8",
		"model-regression-gate/repair-retained-output-arena",
		"model-regression-gate/repair-retained-packing",
		"model-regression-gate/readmit-retained-controls",
		"model-regression-gate/readmit-controls",
		"model-regression-gate/repair-pool-release",
		"model-regression-gate/readmit-pool-controls",
		"model-regression-baseline/do",
		"simplify-prefill-paths/do",
		"simplify-command-surface/do",
		"simplify-execution-core/do",
		"simplify-checkpoint-locations/do",
		"simplify-capability-report/do",
		"generation-soak/do",
		"docs-pruning/do",
		"benchmark-completion/bbh-pass",
		"benchmark-completion/musr-pass",
		"benchmark-completion/ifeval-pass",
		"benchmark-completion/dna-pass",
		"benchmark-completion/mmlu-pro-pass",
		"benchmark-completion/published-comparison",
		"modality-verification/capability-census",
		"modality-verification/capability-census-owner",
		"modality-verification/e4b-all-modalities",
		"modality-verification/e4b-validation-producer",
		"modality-verification/e4b-media-resource-producer",
		"modality-verification/e4b-protocol-parity-repair",
		"modality-verification/e4b-resource-recovery",
		"modality-verification/e4b-scored-audio-producer",
		"modality-verification/e4b-serving-repair",
		"modality-verification/e4b-serving-admission",
		"gpu-capacity-admission/do",
		"validation-publication-lifetime/do",
		"gui-conversation-recovery-intake/merge",
		"audio-cpu-production-intake/merge",
		"gui-worktree-integration/merge",
		"gui-vqa-integration/merge",
		"gui-validation-handoff/do",
		"audio-worktree-integration/register-models",
		"validation-integration-replan/do",
		// Accept or remove the staged API from the explicitly authorized audio intake.
		"modality-verification/native-audio-streaming-intake",
		"model-validation-batching/do",
		"modality-verification/declared-smoke-expectations",
		"modality-verification/text-and-vision",
		"modality-verification/image-and-video",
		"modality-verification/speech-ocr-tabular-forecast",
		"modality-verification/media-report",
		"failure-recovery/rollout-plan-author",
		// Owner-directed capability-first replan (2026-09-04): bind
		// acceptance before device work and keep independent host fixes ready.
		"validation-readiness/measurement-contract",
		"validation-replan/do",
		"capability-simplification-replan/do",
		"model-regression-baseline/full-catalog",
		"model-regression-baseline/prepare-guard",
		"baseline-repair-order/do",
		"model-regression-baseline/complete-coverage",
		"model-regression-baseline/repair-throughput",
		"validation-readiness/failure-diagnostics",
		"validation-automation/gate-scope-efficiency",
		"validation-automation/guard-admission-efficiency",
		"validation-automation/benchmark-protocol",
		"validation-batch-control/batch-promotion",
		"validation-batch-control/evidence-resource-producer",
		"validation-batch-control/transaction-writer",
		"validation-batch-control/workbench-writer-lifetime",
		"validation-batch-control/abandoned-gate-recovery",
		"validation-batch-control/selection-external-consumers",
		"validation-batch-control/runtime-input-preflight",
		"validation-batch-control/candidate-source-isolation",
		"validation-batch-control/batch-terminal-obligations",
		"boundary-hardening/cross-origin",
		"boundary-hardening/argv-output",
		"final-model-validation/do",
		"benchmark-27b/mmlu-pro-pass",
		"campaign-closeout/closeout",
	}
	want := make(map[string]bool, len(wantIDs))
	for _, id := range wantIDs {
		want[id] = true
	}
	mergeRows := 0
	for _, item := range document.Items {
		if strings.HasPrefix(item.ID, "merge-") {
			mergeRows++
			if !preparedMergeBoundary(item) {
				t.Errorf("invalid prepared merge boundary %+v", item)
			}
			continue
		}
		for _, step := range item.Steps {
			id := item.ID + "/" + step.ID
			if !want[id] {
				t.Errorf("validation campaign has undeclared step %s; the freeze admits only validation drills", id)
				continue
			}
			if strings.TrimSpace(step.Verify) == "" {
				t.Errorf("validation step %s has no machine-checked verify", id)
			}
		}
	}
	if mergeRows > 1 {
		t.Errorf("validation campaign has %d prepared merge boundaries, want at most one", mergeRows)
	}
	for _, required := range []string{"capability freeze enforced by structure", "UNAVAILABLE", "plan -setverify"} {
		if !strings.Contains(document.Doctrine, required) {
			t.Errorf("validation campaign doctrine omits %q", required)
		}
	}
}

// The gate separately proves the exact parent and prunes this temporary row.
// Campaign closeout therefore cannot depend on it without changing the target plan.
func preparedMergeBoundary(item Item) bool {
	revision, found := strings.CutPrefix(item.ID, "merge-")
	return found && len(revision) == 12 && strings.Trim(revision, "0123456789abcdef") == "" &&
		item.Status == StatusOpen && len(item.Steps) == 1 && item.Steps[0].ID == "do" &&
		item.Steps[0].Status == StatusOpen &&
		(item.Steps[0].Verify == "go run ./cmd/compatibility -check" ||
			strings.HasPrefix(item.Steps[0].Verify, "go run ./cmd/compatibility -check && "))
}

func TestPreparedMergeBoundaryVerification(t *testing.T) {
	item := Item{ID: "merge-fb7de0423660", Status: StatusOpen, Steps: []Step{{ID: "do", Status: StatusOpen}}}
	for _, test := range []struct {
		verify string
		want   bool
	}{
		{"go run ./cmd/compatibility -check", true},
		{"go run ./cmd/compatibility -check && go test ./internal/server -run '^TestConversationHistoryPaging$' -count=1", true},
		{"go test ./internal/server -run '^TestConversationHistoryPaging$' -count=1", false},
		{"go run ./cmd/compatibility -check || go test ./internal/server", false},
	} {
		item.Steps[0].Verify = test.verify
		if got := preparedMergeBoundary(item); got != test.want {
			t.Errorf("verification %q: prepared merge = %v, want %v", test.verify, got, test.want)
		}
	}
}

func assertIntegratedReconciliationSnapshot(t *testing.T, document Plan) {
	t.Helper()
	_, reconciling := retainedCampaignStep(document, "reconcile-integrated-rsi-20260829/do")
	_, resequencing := retainedCampaignStep(document, "resequence-supervised-first-win-20260829/do")
	planning := reconciling || resequencing
	wantIDs := []string{
		"experience-projection/capability-episodes",
		"experience-projection/atomic-interaction-context",
		"grounded-retrieval/structural-hybrid-search",
		"grounded-retrieval/retrieval-evidence",
		"capability-control/production-probes",
		"capability-control/intent-routing",
		"capability-control/trajectory-supervision",
		"knowledge-promotion/episode-dataset-promotion",
		"knowledge-promotion/supervised-grounded-replay",
		"architecture-ratchets/go-only-guard",
		"campaign-concurrency/ready-frontier-leases",
		"interaction-efficiency/bounded-training-evidence",
		"interaction-efficiency/projection-query-plans",
		"interaction-efficiency/head-bound-deltas",
		"interaction-efficiency/incremental-context",
		"interaction-efficiency/coalesced-coordination",
		"interaction-efficiency/operator-decisions",
		"interaction-efficiency/efficiency-gate",
		"tool-propensity/tool-call-propensity-control",
		"server-hardening/route-authentication",
		"server-hardening/approval-decision-separation",
		"server-hardening/download-shutdown-ownership",
		"plan-discovery-hardening/worktree-safe-discovery",
		"generated-authority-refresh/regenerate-authorities",
		"gate-preflight/preflight-command",
		"staged-surface-reconciliation/consume-or-retire",
		"gate-decomposition/extract-phases",
		"server-decomposition/workspace-dependencies",
		"structure-budgets/budget-ratchets",
		"model-abstraction-exactness/family-branch-census",
		"verification-depth/inference-coverage",
		"portability-polish/operational-cleanup",
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
		"supervised-composite/first-win",
		"composition-expansion/alignment-adapter",
		"composition-expansion/search-ratchet",
		"advanced-domain-plugins/realization",
		"advanced-domain-plugins/evidence",
		"outward-capabilities/peer-invocation",
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
	mergeRows := 0
	for _, item := range document.Items {
		if item.ID == "reconcile-integrated-rsi-20260829" || item.ID == "resequence-supervised-first-win-20260829" || item.ID == "grounded-upgrade-plan-admission-20260830" {
			continue
		}
		if strings.HasPrefix(item.ID, "merge-") {
			mergeRows++
			if !preparedMergeBoundary(item) {
				t.Errorf("invalid prepared merge boundary %+v", item)
			}
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
	if mergeRows > 1 {
		t.Errorf("reconciled campaign has %d prepared merge boundaries, want at most one", mergeRows)
	}
	for _, id := range wantIDs {
		if planning && !seen[id] {
			t.Errorf("reconciled campaign lost retained step %s", id)
		}
	}

	layers := retainedCampaignLayers(t, document, map[string]bool{
		"reconcile-integrated-rsi-20260829":        true,
		"resequence-supervised-first-win-20260829": true,
		"grounded-upgrade-plan-admission-20260830": true,
	})
	postProof := []string{
		"campaign-concurrency/ready-frontier-leases",
		"interaction-efficiency/bounded-training-evidence",
		"evidence-routing/counterfactual-replay",
		"live-safety/safety-window",
		"composition-expansion/alignment-adapter",
		"advanced-domain-plugins/realization",
		"outward-capabilities/peer-invocation",
	}
	if allCampaignStepsRetained(document, postProof...) {
		maxWidth := 0
		for _, layer := range layers {
			maxWidth = max(maxWidth, len(layer))
		}
		if maxWidth < len(postProof) {
			t.Errorf("reconciled campaign maximum ready-layer width = %d, want at least %d", maxWidth, len(postProof))
		}
	}
	critical := []string{
		"model-prototypes/prototype-admission",
		"model-prototypes/prototype-compile",
		"evidence-routing/evidence-selection",
		"model-prototypes/prototype-materialization",
		"model-recipe-lifecycle/recipe-derivation",
		"model-recipe-lifecycle/recipe-validation",
		"model-recipe-lifecycle/isolated-performance-evidence",
		"model-recipe-lifecycle/recipe-evaluation",
		"supervised-composite/first-win",
	}
	for index := 1; index < len(critical); index++ {
		before, after := critical[index-1], critical[index]
		_, beforeRetained := retainedCampaignStep(document, before)
		_, afterRetained := retainedCampaignStep(document, after)
		if beforeRetained && afterRetained && !campaignDependsOn(document, after, before, map[string]bool{}) {
			t.Errorf("supervised path %s does not depend on %s", after, before)
		}
	}
	if _, retained := retainedCampaignStep(document, "supervised-composite/first-win"); retained {
		for _, predecessor := range []string{"model-recipe-lifecycle/recipe-evaluation", "model-recipe-lifecycle/recipe-activation"} {
			if _, present := retainedCampaignStep(document, predecessor); present && !campaignDependsOn(document, "supervised-composite/first-win", predecessor, map[string]bool{}) {
				t.Errorf("supervised first win does not join %s", predecessor)
			}
		}
	}
	assertSameCampaignLayerIfRetained(t, document, layers,
		"model-prototypes/prototype-contract",
		"evidence-routing/decision-contract")
	assertSameCampaignLayerIfRetained(t, document, layers, postProof...)
	assertSameCampaignLayerIfRetained(t, document, layers,
		"interaction-efficiency/incremental-context",
		"interaction-efficiency/coalesced-coordination",
		"interaction-efficiency/operator-decisions")
	assertSameCampaignLayerIfRetained(t, document, layers,
		"deterministic-rollout/rollout-plan",
		"composition-expansion/search-ratchet",
		"advanced-domain-plugins/evidence")
	assertSameCampaignLayerIfRetained(t, document, layers,
		"deterministic-rollout/promotion-binding",
		"live-safety/circuit-breaker")
	if _, found := retainedCampaignStep(document, "supervised-composite/first-win"); found {
		for _, deferred := range postProof {
			if campaignDependsOn(document, "supervised-composite/first-win", deferred, map[string]bool{}) {
				t.Errorf("supervised first win is transitively gated by deferred work %s", deferred)
			}
		}
	}
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
