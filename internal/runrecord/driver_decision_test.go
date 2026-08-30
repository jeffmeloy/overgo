package runrecord

import (
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

func TestEvidenceDriverDecisionContract(t *testing.T) {
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	goal := testutil.ArtifactID(t, artifact.KindEvidence, "driver-goal")
	causal, err := NewCausalRoot(
		TriggerControllerProposal,
		testutil.ArtifactID(t, artifact.KindEvidence, "driver-root"),
		testutil.ArtifactID(t, artifact.KindEvidence, "driver-motivation-b"),
		testutil.ArtifactID(t, artifact.KindEvidence, "driver-motivation-a"),
	)
	if err != nil {
		t.Fatal(err)
	}
	interaction, err := NewBudget(
		"queries",
		testutil.ArtifactID(t, artifact.KindDatasetShard, "driver-interaction-split"),
		10,
		testutil.ArtifactID(t, artifact.KindEvidence, "driver-interaction-authority"),
	)
	if err != nil {
		t.Fatal(err)
	}
	resources, err := NewBudget(
		"compute-units",
		testutil.ArtifactID(t, artifact.KindDatasetShard, "driver-resource-split"),
		20,
		testutil.ArtifactID(t, artifact.KindEvidence, "driver-resource-authority"),
	)
	if err != nil {
		t.Fatal(err)
	}
	interactionChargeA, err := NewBudgetCharge(
		interaction.ID, 1, testutil.ArtifactID(t, artifact.KindEvidence, "driver-interaction-consumer-a"), "inspect",
	)
	if err != nil {
		t.Fatal(err)
	}
	interactionChargeB, err := NewBudgetCharge(
		interaction.ID, 2, testutil.ArtifactID(t, artifact.KindEvidence, "driver-interaction-consumer-b"), "compare",
	)
	if err != nil {
		t.Fatal(err)
	}
	resourceChargeA, err := NewBudgetCharge(
		resources.ID, 3, testutil.ArtifactID(t, artifact.KindEvidence, "driver-resource-consumer-a"), "realize",
	)
	if err != nil {
		t.Fatal(err)
	}
	resourceChargeB, err := NewBudgetCharge(
		resources.ID, 4, testutil.ArtifactID(t, artifact.KindEvidence, "driver-resource-consumer-b"), "evaluate",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "driver-decision/budget-parents",
		Artifacts: []artifact.Descriptor{
			{ID: interaction.Split}, {ID: interaction.Authority},
			{ID: resources.Split}, {ID: resources.Authority},
			{ID: interactionChargeA.Consumer}, {ID: interactionChargeB.Consumer},
			{ID: resourceChargeA.Consumer}, {ID: resourceChargeB.Consumer},
		},
	}); err != nil {
		t.Fatal(err)
	}
	contents := make([]artifact.Content, 0, 6)
	var budgetLineage []artifact.Lineage
	for _, budget := range []Budget{interaction, resources} {
		content, err := budget.Content()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, content)
		budgetLineage = append(budgetLineage, budget.Lineage()...)
	}
	for _, charge := range []BudgetCharge{interactionChargeA, interactionChargeB, resourceChargeA, resourceChargeB} {
		content, err := charge.Content()
		if err != nil {
			t.Fatal(err)
		}
		contents = append(contents, content)
		budgetLineage = append(budgetLineage, charge.Lineage()...)
	}
	budgetBatch, err := artifact.NewDocumentBatch("driver-decision/budgets", contents, budgetLineage, nil)
	if err != nil {
		t.Fatal(err)
	}
	head, err := store.Commit(ctx, budgetBatch)
	if err != nil {
		t.Fatal(err)
	}

	incumbent := DriverOption{
		Recipe:    testutil.ArtifactID(t, artifact.KindRecipe, "driver-incumbent"),
		Lifecycle: testutil.ArtifactID(t, artifact.KindEvidence, "driver-incumbent-lifecycle"),
		State:     DriverIncumbentActive,
		Placement: testutil.ArtifactID(t, artifact.KindProfile, "driver-incumbent-placement"),
		Evidence: []artifact.ID{
			testutil.ArtifactID(t, artifact.KindEvaluation, "driver-incumbent-evaluation"),
		},
	}
	eligible := DriverOption{
		Candidate: testutil.ArtifactID(t, artifact.KindRecipe, "driver-eligible-candidate"),
		Admission: testutil.ArtifactID(t, artifact.KindEvidence, "driver-eligible-admission"),
		State:     DriverCandidateEligible,
		Missing: []DriverEvidenceGap{{
			Need: DriverNeedRealization,
		}},
	}
	eligible.Missing[0].Target = eligible.Candidate
	validated := DriverOption{
		Candidate: testutil.ArtifactID(t, artifact.KindRecipe, "driver-validated-candidate"),
		Admission: testutil.ArtifactID(t, artifact.KindEvidence, "driver-validated-admission"),
		Recipe:    testutil.ArtifactID(t, artifact.KindRecipe, "driver-validated-recipe"),
		Lifecycle: testutil.ArtifactID(t, artifact.KindEvidence, "driver-validated-lifecycle"),
		State:     DriverCandidateValidated,
		Placement: testutil.ArtifactID(t, artifact.KindProfile, "driver-validated-placement"),
		Evidence: []artifact.ID{
			testutil.ArtifactID(t, artifact.KindEvidence, "driver-known-evidence-b"),
			testutil.ArtifactID(t, artifact.KindEvidence, "driver-known-evidence-a"),
		},
		Missing: []DriverEvidenceGap{
			{Need: DriverNeedAblation, Target: testutil.ArtifactID(t, artifact.KindRecipe, "driver-ablation")},
			{Need: DriverNeedEvaluation, Target: testutil.ArtifactID(t, artifact.KindEvaluation, "driver-evaluation")},
		},
	}
	facts := DriverDecisionFacts{
		Goal: goal, Causal: causal, Head: head,
		Options: []DriverOption{validated, incumbent, eligible},
		InteractionBudget: DriverBudgetReference{
			Grant: interaction.ID,
		},
		ResourceBudget: DriverBudgetReference{
			Grant: resources.ID,
		},
	}

	decision, err := NewDriverDecision(ctx, store, facts)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decision.Content(); err != nil {
		t.Fatal(err)
	}
	if decision.Head != head || decision.Incumbent != incumbent.Recipe ||
		decision.Stop != DriverStopContinue || decision.Action != DriverActionRealize ||
		decision.Subject != eligible.Candidate || decision.Target != eligible.Candidate ||
		decision.Need != DriverNeedRealization || decision.TieBreak != DriverTieActionThenIdentity {
		t.Fatalf("derived decision = %+v", decision)
	}
	if decision.Interaction.Issued != 10 || decision.Interaction.Consumed != 3 || decision.Interaction.Remaining != 7 ||
		decision.Resources.Issued != 20 || decision.Resources.Consumed != 7 || decision.Resources.Remaining != 13 {
		t.Fatalf("derived budgets = interaction %+v, resources %+v", decision.Interaction, decision.Resources)
	}
	var admitted DriverOption
	for _, option := range decision.Options {
		if option.Candidate == eligible.Candidate {
			admitted = option
			break
		}
	}
	if admitted.State != DriverCandidateEligible || admitted.Recipe.Valid() || admitted.Lifecycle.Valid() || admitted.Placement.Valid() ||
		len(admitted.Missing) != 1 || admitted.Missing[0].Need != DriverNeedRealization || admitted.Missing[0].Target != admitted.Candidate {
		t.Fatalf("pre-realization admission gained lifecycle state: %+v", admitted)
	}

	permutedValidated := validated
	permutedValidated.Evidence = slices.Clone(validated.Evidence)
	permutedValidated.Missing = slices.Clone(validated.Missing)
	slices.Reverse(permutedValidated.Evidence)
	slices.Reverse(permutedValidated.Missing)
	permuted := facts
	permuted.Options = []DriverOption{eligible, permutedValidated, incumbent}
	canonical, err := NewDriverDecision(ctx, store, permuted)
	if err != nil {
		t.Fatal(err)
	}
	_, canonicalErr := canonical.Content()
	if canonical.ID != decision.ID || canonicalErr != nil {
		t.Fatalf("permuted decision = %s, validation %v; want %s", canonical.ID, canonicalErr, decision.ID)
	}

	wantParents := []artifact.ID{
		goal, incumbent.Recipe, incumbent.Lifecycle, incumbent.Placement, incumbent.Evidence[0],
		interaction.ID, interactionChargeA.ID, interactionChargeB.ID,
		resources.ID, resourceChargeA.ID, resourceChargeB.ID,
		eligible.Candidate, eligible.Admission,
		validated.Candidate, validated.Admission, validated.Recipe, validated.Lifecycle, validated.Placement,
		validated.Evidence[0], validated.Evidence[1], validated.Missing[0].Target, validated.Missing[1].Target,
	}
	lineage := decision.Lineage()
	if len(lineage) != len(wantParents) {
		t.Fatalf("lineage count = %d, want %d: %+v", len(lineage), len(wantParents), lineage)
	}
	for _, parent := range wantParents {
		if !slices.ContainsFunc(lineage, func(edge artifact.Lineage) bool {
			return edge.Child == decision.ID && edge.Parent == parent && edge.Relation == artifact.RelationDependsOn
		}) {
			t.Fatalf("authority %s is absent from decision lineage", parent)
		}
	}
	if slices.ContainsFunc(lineage, func(edge artifact.Lineage) bool { return edge.Parent == causal.Root }) {
		t.Fatal("causal projection was conflated with artifact lineage")
	}

	for _, test := range []struct {
		name   string
		option DriverOption
	}{
		{name: "admission claims validated", option: func() DriverOption {
			value := eligible
			value.State = DriverCandidateValidated
			return value
		}()},
		{name: "admission carries lifecycle", option: func() DriverOption {
			value := eligible
			value.Recipe = testutil.ArtifactID(t, artifact.KindRecipe, "driver-premature-recipe")
			value.Lifecycle = testutil.ArtifactID(t, artifact.KindEvidence, "driver-premature-lifecycle")
			return value
		}()},
		{name: "active candidate lacks placement", option: DriverOption{
			Candidate: validated.Candidate, Admission: validated.Admission, Recipe: validated.Recipe,
			Lifecycle: validated.Lifecycle, State: DriverCandidateActive,
		}},
	} {
		t.Run("refuses lifecycle state "+test.name, func(t *testing.T) {
			invalid := facts
			invalid.Options = []DriverOption{incumbent, test.option}
			if _, err := NewDriverDecision(ctx, store, invalid); err == nil {
				t.Fatal("incoherent lifecycle state was admitted")
			}
		})
	}

	overlap := validated
	overlap.Evidence = append(slices.Clone(validated.Evidence), validated.Missing[0].Target)
	disjoint := facts
	disjoint.Options = []DriverOption{incumbent, overlap}
	if _, err := NewDriverDecision(ctx, store, disjoint); err == nil {
		t.Fatal("the same artifact was accepted as both present and missing evidence")
	}

	for _, test := range []struct {
		name   string
		mutate func(*DriverDecision)
	}{
		{name: "head", mutate: func(value *DriverDecision) { value.Head = artifact.CommitID{9} }},
		{name: "action", mutate: func(value *DriverDecision) { value.Action = DriverActionSelect }},
		{name: "stop", mutate: func(value *DriverDecision) { value.Stop = DriverStopSaturated }},
		{name: "subject", mutate: func(value *DriverDecision) { value.Subject = incumbent.Recipe }},
		{name: "target", mutate: func(value *DriverDecision) { value.Target = validated.Recipe }},
		{name: "need", mutate: func(value *DriverDecision) { value.Need = DriverNeedPromotion }},
		{name: "tie break", mutate: func(value *DriverDecision) { value.TieBreak = "caller-choice" }},
		{name: "budget balance", mutate: func(value *DriverDecision) { value.Interaction.Remaining++ }},
	} {
		t.Run("refuses mutation "+test.name, func(t *testing.T) {
			mutated := cloneDriverDecision(decision)
			test.mutate(&mutated)
			if _, err := mutated.Content(); err == nil {
				t.Fatal("mutated decision retained valid identity")
			}
		})
	}

	saturationA := testutil.ArtifactID(t, artifact.KindEvidence, "driver-saturation-a")
	saturationB := testutil.ArtifactID(t, artifact.KindEvidence, "driver-saturation-b")
	saturatedFacts := facts
	saturatedFacts.SaturationEvidence = []artifact.ID{saturationB, saturationA}
	saturated, err := NewDriverDecision(ctx, store, saturatedFacts)
	if err != nil {
		t.Fatal(err)
	}
	wantSaturationTarget := saturationA
	if artifact.CompareID(saturationB, saturationA) < 0 {
		wantSaturationTarget = saturationB
	}
	if saturated.Stop != DriverStopSaturated || saturated.Action != DriverActionStop ||
		saturated.Target != wantSaturationTarget || saturated.Subject.Valid() || saturated.Need != "" {
		t.Fatalf("saturated decision = %+v", saturated)
	}

	operatorEvidence := testutil.ArtifactID(t, artifact.KindEvidence, "driver-operator-stop")
	operatorFacts := saturatedFacts
	operatorFacts.OperatorStop = operatorEvidence
	operator, err := NewDriverDecision(ctx, store, operatorFacts)
	if err != nil {
		t.Fatal(err)
	}
	if operator.Stop != DriverStopOperator || operator.Action != DriverActionStop || operator.Target != operatorEvidence {
		t.Fatalf("operator stop decision = %+v", operator)
	}

	refusalEvidence := testutil.ArtifactID(t, artifact.KindEvidence, "driver-refusal")
	refusalFacts := facts
	refusalFacts.Refusal = refusalEvidence
	refused, err := NewDriverDecision(ctx, store, refusalFacts)
	if err != nil {
		t.Fatal(err)
	}
	if refused.Stop != DriverStopRefused || refused.Action != DriverActionRefuse || refused.Target != refusalEvidence {
		t.Fatalf("refused decision = %+v", refused)
	}

	fullInteraction, err := NewBudgetCharge(
		interaction.ID, decision.Interaction.Remaining,
		testutil.ArtifactID(t, artifact.KindEvidence, "driver-exhaustion-consumer"), "exhaust",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key:       "driver-decision/exhaustion-parent",
		Artifacts: []artifact.Descriptor{{ID: fullInteraction.Consumer}},
	}); err != nil {
		t.Fatal(err)
	}
	fullInteractionBatch, err := fullInteraction.Batch("driver-decision/exhaustion-charge")
	if err != nil {
		t.Fatal(err)
	}
	exhaustedHead, err := store.Commit(ctx, fullInteractionBatch)
	if err != nil {
		t.Fatal(err)
	}
	// The caller repeats the same grant-only reference. The committed charge is
	// discovered from lineage and therefore cannot be omitted to retain budget.
	exhaustedFacts := facts
	exhaustedFacts.Head = exhaustedHead
	exhausted, err := NewDriverDecision(ctx, store, exhaustedFacts)
	if err != nil {
		t.Fatal(err)
	}
	if exhausted.Stop != DriverStopBudgetExhausted || exhausted.Action != DriverActionRefuse ||
		exhausted.Subject.Valid() || exhausted.Target.Valid() || exhausted.Need != "" ||
		exhausted.Interaction.Remaining != 0 || len(exhausted.Interaction.Charges) != 3 ||
		!slices.Contains(exhausted.Interaction.Charges, fullInteraction.ID) {
		t.Fatalf("exhausted decision omitted a committed charge: %+v", exhausted)
	}
}
