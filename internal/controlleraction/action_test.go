package controlleraction

import (
	"encoding/json"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/plan"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodeltest"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowrecipe"
)

// TestControllerActionsAreAllowlistedTransformations pins the action-language
// contract: every allowlisted kind compiles deterministically through the
// shared authorities into content-addressed artifacts; unknown kinds and
// kind/payload mismatches are unrepresentable; and the chain emission is
// genuinely executable through the generic workflow runtime's catalog.
func TestControllerActionsAreAllowlistedTransformations(t *testing.T) {
	compile := func(action Action) ([]artifact.Content, error) {
		batch, err := CompileTransaction(t.Context(), nil, action)
		return batch.Contents, err
	}
	scorer := testutil.ArtifactID(t, artifact.KindModel, "action-scorer")
	drafter := testutil.ArtifactID(t, artifact.KindModel, "action-drafter")
	chain := Action{Version: ActionVersion, Kind: KindChainRecipes, Chain: &ChainAction{Scorer: scorer, Drafter: drafter}}
	first, err := compile(chain)
	if err != nil {
		t.Fatal(err)
	}
	second, err := compile(chain)
	if err != nil || len(first) != 2 || len(second) != 2 ||
		first[0].Descriptor.ID != second[0].Descriptor.ID || first[1].Descriptor.ID != second[1].Descriptor.ID {
		t.Fatalf("chain compile not deterministic: (%v, %v)", second, err)
	}
	definition, err := recipe.ParseDefinition(first[0].Data)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := recipe.CompileProgram(definition, workflowrecipe.Catalog()); err != nil {
		t.Fatalf("emitted chain recipe is not executable through the catalog: %v", err)
	}

	target := testutil.ArtifactID(t, artifact.KindModel, "action-target")
	donor := testutil.ArtifactID(t, artifact.KindModel, "action-donor")
	proposal := Action{Version: ActionVersion, Kind: KindBridgeProposal, Proposal: &ProposalAction{
		Target:           target,
		Candidates:       []composition.BridgeCandidate{{Donor: donor, Component: "blk.3.ffn_gate.weight", Distance: 0.5}},
		RequiredVerifier: "go test ./internal/adaptiveparity -run '^TestComponentCompositionViability$' -count=1",
		Blocker:          "no recorded experiment evidence",
	}}
	contents, err := compile(proposal)
	if err != nil || len(contents) != 2 {
		t.Fatalf("proposal compile = (%d, %v)", len(contents), err)
	}
	parsed, err := composition.ParseBridgeProposal(contents[1].Data)
	if err != nil || parsed.State != composition.ProposalPromotionBlocked {
		t.Fatalf("compiled proposal = (%+v, %v), want promotion-blocked", parsed, err)
	}

	model := testutil.ArtifactID(t, artifact.KindModel, "action-model")
	decomposition := Action{Version: ActionVersion, Kind: KindComponentDecomposition, Decomposition: &DecompositionAction{
		Model:   model,
		Family:  "qwen2",
		Tensors: []modelartifact.TensorFact{{Name: "blk.0.attn_q.weight", Shape: []uint64{8, 8}, Storage: "f32"}},
	}}
	contents, err = compile(decomposition)
	if err != nil || len(contents) != 1 {
		t.Fatalf("decomposition compile = (%d, %v)", len(contents), err)
	}

	improvement := Action{Version: ActionVersion, Kind: KindImprovementTrial, Improvement: &ImprovementAction{
		Proposal: trainingprogram.ImprovementSpec{
			Kind: trainingprogram.ImprovementRecipe, ParentModel: target,
			Incumbent:        testutil.ArtifactID(t, artifact.KindRecipe, "current recipe"),
			Candidate:        testutil.ArtifactID(t, artifact.KindRecipe, "candidate recipe"),
			Dataset:          testutil.ArtifactID(t, artifact.KindDataset, "improvement dataset"),
			DevelopmentSplit: testutil.ArtifactID(t, artifact.KindDatasetShard, "development split"),
			Recipe:           testutil.ArtifactID(t, artifact.KindRecipe, "execution recipe"),
			Code:             testutil.ArtifactID(t, artifact.KindEvidence, "code"),
			Proposer:         testutil.ArtifactID(t, artifact.KindEvidence, "proposer"),
		},
		PromotionSplit: testutil.ArtifactID(t, artifact.KindDatasetShard, "promotion split"),
		Evaluator:      testutil.ArtifactID(t, artifact.KindEvidence, "evaluator"),
		Authority:      testutil.ArtifactID(t, artifact.KindEvidence, "authority"),
	}}
	batch, err := CompileTransaction(t.Context(), nil, improvement)
	if err != nil || len(batch.Contents) != 2 || len(batch.Lineage) == 0 {
		t.Fatalf("improvement compile = (%+v, %v)", batch, err)
	}
	profile := scratchmodeltest.Profile(t)
	profileAction := improvement
	profileAction.Improvement = &ImprovementAction{
		Proposal:       improvement.Improvement.Proposal,
		Profile:        &profile,
		PromotionSplit: improvement.Improvement.PromotionSplit,
		Evaluator:      improvement.Improvement.Evaluator,
		Authority:      improvement.Improvement.Authority,
	}
	profileAction.Improvement.Proposal.Kind = trainingprogram.ImprovementDerivationProfile
	profileAction.Improvement.Proposal.Incumbent = testutil.ArtifactID(t, artifact.KindProfile, "current profile")
	profileAction.Improvement.Proposal.Candidate = profile.ID
	batch, err = CompileTransaction(t.Context(), nil, profileAction)
	if err != nil || len(batch.Contents) != 3 {
		t.Fatalf("profile improvement compile = (%d, %v)", len(batch.Contents), err)
	}

	budget := Action{Version: ActionVersion, Kind: KindEvaluationBudget, Budget: &EvaluationBudgetAction{
		Unit:      "promotion-query",
		Split:     improvement.Improvement.PromotionSplit,
		Issued:    2,
		Authority: improvement.Improvement.Authority,
		Charges: []BudgetChargeAction{{
			Amount: 1, Consumer: improvement.Improvement.Evaluator, Purpose: "score descendant",
		}},
	}}
	batch, err = CompileTransaction(t.Context(), nil, budget)
	if err != nil || len(batch.Contents) != 2 || len(batch.Lineage) == 0 {
		t.Fatalf("budget compile = (%+v, %v)", batch, err)
	}

	driverStore, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer driverStore.Close()
	interactionBudget, err := runrecord.NewBudget(
		"interaction", testutil.ArtifactID(t, artifact.KindDatasetShard, "driver interaction split"), 2,
		testutil.ArtifactID(t, artifact.KindEvidence, "driver interaction authority"),
	)
	if err != nil {
		t.Fatal(err)
	}
	resourceBudget, err := runrecord.NewBudget(
		"resource", testutil.ArtifactID(t, artifact.KindDatasetShard, "driver resource split"), 3,
		testutil.ArtifactID(t, artifact.KindEvidence, "driver resource authority"),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, parent := range []artifact.ID{
		interactionBudget.Split, interactionBudget.Authority, resourceBudget.Split, resourceBudget.Authority,
	} {
		testutil.PublishArtifact(t, driverStore, parent)
	}
	for _, granted := range []struct {
		key   string
		value runrecord.Budget
	}{
		{key: "driver/interaction-budget", value: interactionBudget},
		{key: "driver/resource-budget", value: resourceBudget},
	} {
		budgetBatch, batchErr := granted.value.Batch(granted.key)
		if batchErr != nil {
			t.Fatal(batchErr)
		}
		if _, batchErr = artifact.CommitBatch(t.Context(), driverStore, budgetBatch); batchErr != nil {
			t.Fatal(batchErr)
		}
	}
	causal, err := runrecord.NewCausalRoot(
		runrecord.TriggerControllerProposal, testutil.ArtifactID(t, artifact.KindEvidence, "driver causal root"),
	)
	if err != nil {
		t.Fatal(err)
	}
	driverCandidate := testutil.ArtifactID(t, artifact.KindRecipe, "driver eligible candidate")
	goal := testutil.ArtifactID(t, artifact.KindRecipe, "driver goal")
	incumbent := runrecord.DriverOption{
		Recipe:    testutil.ArtifactID(t, artifact.KindRecipe, "driver incumbent"),
		Lifecycle: testutil.ArtifactID(t, artifact.KindEvidence, "driver incumbent lifecycle"),
		State:     runrecord.DriverIncumbentActive,
		Placement: testutil.ArtifactID(t, artifact.KindProfile, "driver incumbent placement"),
	}
	eligible := runrecord.DriverOption{
		Candidate: driverCandidate,
		Admission: testutil.ArtifactID(t, artifact.KindEvidence, "driver candidate admission"),
		State:     runrecord.DriverCandidateEligible,
		Missing:   []runrecord.DriverEvidenceGap{{Need: runrecord.DriverNeedRealization, Target: driverCandidate}},
	}
	for _, parent := range []artifact.ID{
		goal, causal.Root, incumbent.Recipe, incumbent.Lifecycle, incumbent.Placement,
		eligible.Candidate, eligible.Admission,
	} {
		testutil.PublishArtifact(t, driverStore, parent)
	}
	driverHead, _ := driverStore.Head()
	if !driverHead.Valid() {
		t.Fatal("driver store has no head")
	}
	driver := Action{Version: ActionVersion, Kind: KindDriverDecision, Driver: &DriverDecisionAction{Facts: runrecord.DriverDecisionFacts{
		Goal: goal, Causal: causal, Head: driverHead, Options: []runrecord.DriverOption{incumbent, eligible},
		InteractionBudget: runrecord.DriverBudgetReference{Grant: interactionBudget.ID},
		ResourceBudget:    runrecord.DriverBudgetReference{Grant: resourceBudget.ID},
	}}}
	driverJSON, err := json.Marshal(driver)
	if err != nil {
		t.Fatal(err)
	}
	parsedDriver, err := ParseAction(driverJSON)
	if err != nil {
		t.Fatal(err)
	}
	batch, err = CompileTransaction(t.Context(), driverStore, parsedDriver)
	if err != nil || len(batch.Contents) != 1 || len(batch.Lineage) == 0 || len(batch.Causality) != 1 ||
		batch.ExpectedHead == nil || *batch.ExpectedHead != driver.Driver.Facts.Head {
		t.Fatalf("driver decision compile = (%+v, %v)", batch, err)
	}
	testutil.PublishArtifact(t, driverStore, testutil.ArtifactID(t, artifact.KindEvidence, "driver concurrent write"))
	if _, err := artifact.CommitBatch(t.Context(), driverStore, batch); err == nil {
		t.Fatal("stale driver decision committed after the repository head advanced")
	}
	freshHead, _ := driverStore.Head()
	if !freshHead.Valid() {
		t.Fatal("driver store lost its head")
	}
	parsedDriver.Driver.Facts.Head = freshHead
	batch, err = CompileTransaction(t.Context(), driverStore, parsedDriver)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), driverStore, batch); err != nil {
		t.Fatalf("fresh driver decision did not commit: %v", err)
	}

	strategy := testutil.ArtifactID(t, artifact.KindProfile, "scheduler strategy")
	scheduled := testutil.ArtifactID(t, artifact.KindModelDefinition, "scheduled candidate")
	scheduling := Action{Version: ActionVersion, Kind: KindCandidateScheduling, Scheduling: &CandidateSchedulingAction{
		Outcomes: []plan.CandidateOutcome{{
			Evidence: testutil.ArtifactID(t, artifact.KindEvidence, "scheduler outcome"), Candidate: scheduled, Strategy: strategy,
			WallNS: 2, PeakDeviceBytes: 2,
			Changes: []plan.CapabilityChange{{Name: "quality", Before: 0, After: 1, Direction: runrecord.DirectionMaximize}},
		}},
		Candidates: []plan.ScheduledCandidate{{Candidate: scheduled, Strategy: strategy, PredictedWallNS: 2, PredictedVRAM: 2}},
		Budget:     plan.SchedulerBudget{WallNS: 2, VRAMBytes: 2},
	}}
	batch, err = CompileTransaction(t.Context(), nil, scheduling)
	if err != nil || len(batch.Contents) != 1 || len(batch.Lineage) == 0 {
		t.Fatalf("scheduler compile = (%+v, %v)", batch, err)
	}

	if _, err := compile(Action{Version: ActionVersion, Kind: "run-shell", Chain: chain.Chain}); err == nil {
		t.Fatal("kind outside the allowlist compiled")
	}
	if _, err := compile(Action{Version: ActionVersion, Kind: KindChainRecipes, Proposal: proposal.Proposal}); err == nil {
		t.Fatal("kind/payload mismatch compiled")
	}
	if _, err := compile(Action{Version: ActionVersion, Kind: KindChainRecipes, Chain: chain.Chain, Proposal: proposal.Proposal}); err == nil {
		t.Fatal("multiple payloads compiled")
	}
	if _, err := ParseAction([]byte(`{"version":1,"kind":"chain-recipes","chain":{"scorer":"` +
		scorer.String() + `","drafter":"` + drafter.String() + `"},"shell":"rm -rf /"}`)); err == nil {
		t.Fatal("action document with a code-carrying field parsed; the language must refuse unknown fields")
	}

	serialized, err := json.Marshal(chain)
	if err != nil {
		t.Fatal(err)
	}
	roundtrip, err := ParseAction(serialized)
	if err != nil || roundtrip.Kind != KindChainRecipes || roundtrip.Chain == nil {
		t.Fatalf("roundtrip = (%+v, %v)", roundtrip, err)
	}
}
