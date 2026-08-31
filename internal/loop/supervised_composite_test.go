package loop

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/invocation"
	"overgo/internal/modelrecipe"
	"overgo/internal/operatoraction"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type supervisedCompositeProof struct {
	Candidate         modelrecipe.Candidate
	Decision          runrecord.DriverDecision
	Compilation       modelrecipe.CandidateCompilation
	Materialization   modelrecipe.CandidateMaterialization
	Evaluation        evaluation.CrossDomainCandidateEvaluation
	Definition        recipe.Definition
	HumanDecision     artifact.ID
	Receipts          []runrecord.StageReceipt
	Lifecycle         recipe.LifecycleEvent
	Strategy          artifact.ID
	Attempt           runrecord.AttemptRecord
	Trace             runrecord.InteractionTrace
	ParentDefinitions []artifact.ID
	ParentInventories []artifact.ID
	IncumbentAlias    string
	Incumbent         artifact.ID
	AutomationAlias   string
	AutomationGrant   artifact.ID
}

func verifySupervisedCompositeFirstWin(
	ctx context.Context,
	reader artifact.Reader,
	proof supervisedCompositeProof,
) error {
	if ctx == nil || reader == nil || proof.Candidate.ValidateIdentity() != nil ||
		proof.Compilation.ValidateIdentity() != nil || proof.Materialization.ValidateIdentity() != nil ||
		proof.Attempt.ValidateIdentity() != nil || proof.Trace.ValidateIdentity() != nil ||
		proof.Definition.ValidateIdentity() != nil || proof.Lifecycle.Validate() != nil ||
		proof.Strategy.Kind() != artifact.KindProfile || proof.HumanDecision.Kind() != artifact.KindEvidence ||
		proof.IncumbentAlias == "" || proof.AutomationAlias == "" {
		return errors.New("loop: invalid supervised composite proof")
	}
	if _, err := proof.Decision.Content(); err != nil {
		return err
	}
	if _, err := proof.Evaluation.Content(); err != nil {
		return err
	}
	replayedEvaluation, err := evaluation.EvaluateCrossDomainCandidate(
		proof.Evaluation.Contract, proof.Decision, proof.Compilation, proof.Evaluation.Arms,
	)
	if err != nil || replayedEvaluation.ID != proof.Evaluation.ID {
		return errors.Join(errors.New("loop: supervised candidate evaluation does not replay"), err)
	}
	trial := proof.Compilation.Trial
	if proof.Decision.Subject != proof.Candidate.ID() || trial.Candidate != proof.Candidate.ID() ||
		proof.Materialization.Decision != proof.Decision.ID || proof.Materialization.Candidate != proof.Candidate.ID() ||
		proof.Materialization.Trial != trial.ID || proof.Materialization.EvaluationPlan != proof.Compilation.EvaluationPlan.ID ||
		proof.Evaluation.Contract.Decision != proof.Decision.ID ||
		proof.Evaluation.Contract.Candidate != proof.Candidate.ID() ||
		proof.Evaluation.Contract.Trial != trial.ID ||
		proof.Evaluation.Verdict != evaluation.CandidateEvaluationImproved {
		return errors.New("loop: supervised candidate evidence spine differs")
	}
	if len(proof.ParentDefinitions) == 0 || len(proof.ParentInventories) == 0 ||
		!slices.Contains(proof.ParentDefinitions, trial.Parent) {
		return errors.New("loop: supervised composite omits resolved parent authority")
	}
	for _, arm := range proof.Materialization.Arms {
		if !slices.Contains(proof.ParentDefinitions, arm.ModelDefinition) && arm.Ablation.Valid() ||
			!slices.Contains(proof.ParentInventories, arm.TensorInventory) {
			return errors.New("loop: supervised materialization leaves resolved definitions or inventories")
		}
	}
	if proof.Definition.Model != proof.Materialization.Arms[0].Model ||
		proof.Lifecycle.Recipe != proof.Definition.ID || proof.Lifecycle.To != recipe.StatusVerified ||
		!slices.Contains(proof.Lifecycle.Evidence, proof.Evaluation.ID) {
		return errors.New("loop: supervised lifecycle is not the evaluated non-serving recipe")
	}
	human, err := runrecord.RequireHumanDecision(ctx, reader, proof.HumanDecision)
	if err != nil || human.Answer != operatoraction.AnswerGrant {
		return errors.Join(errors.New("loop: supervised proof lacks argument-bound human grant"), err)
	}
	states := []runrecord.StageState{runrecord.StageAdmitted, runrecord.StageRunning, runrecord.StageCompleted}
	if len(proof.Receipts) != len(states) {
		return errors.New("loop: supervised verification receipt chain is incomplete")
	}
	for index, receipt := range proof.Receipts {
		if receipt.State != states[index] || receipt.Invocation == nil ||
			receipt.Invocation.Boundary != invocation.BoundaryInternal ||
			receipt.Invocation.Kind != invocation.MutationPromotion ||
			receipt.Invocation.Authority != human.ID || receipt.Operation != human.Operation ||
			index > 0 && receipt.Previous != proof.Receipts[index-1].ID {
			return errors.New("loop: supervised verification receipt authority differs")
		}
		stored, found, readErr := artifact.ReadContent(ctx, reader, receipt.ID)
		if readErr != nil || !found {
			return errors.Join(errors.New("loop: supervised verification receipt is absent"), readErr)
		}
		parsed, parseErr := runrecord.ParseStageReceipt(stored.Data)
		if parseErr != nil || parsed.ID != receipt.ID {
			return errors.Join(errors.New("loop: supervised verification receipt content differs"), parseErr)
		}
	}
	terminal := proof.Receipts[len(proof.Receipts)-1]
	if !slices.Contains(proof.Lifecycle.Evidence, terminal.ID) ||
		proof.Attempt.StrategyID != proof.Strategy || proof.Attempt.Result != proof.Evaluation.ID ||
		proof.Trace.Strategy != proof.Strategy || !slices.Contains(proof.Trace.Attempts, proof.Attempt.ID) ||
		!slices.Contains(proof.Trace.Decisions, proof.Decision.ID) ||
		!slices.Contains(proof.Trace.Decisions, proof.HumanDecision) ||
		!slices.Contains(proof.Trace.FinalArtifacts, proof.Materialization.ID) ||
		!slices.Contains(proof.Trace.FinalArtifacts, proof.Evaluation.ID) {
		return errors.New("loop: supervised strategy attempt, receipt, or interaction trace differs")
	}
	for _, id := range append(slices.Clone(proof.ParentDefinitions), proof.ParentInventories...) {
		if _, found, readErr := reader.Artifact(ctx, id); readErr != nil || !found {
			return errors.Join(errors.New("loop: resolved parent artifact is absent"), readErr)
		}
	}
	contents := slices.Clone(proof.Compilation.Contents)
	for _, content := range []func() (artifact.Content, error){
		proof.Candidate.Content, proof.Decision.Content, proof.Materialization.Content,
		proof.Evaluation.Content, proof.Definition.ArtifactContent, proof.Lifecycle.Content,
		proof.Attempt.Content, proof.Trace.Content,
	} {
		value, contentErr := content()
		if contentErr != nil {
			return contentErr
		}
		contents = append(contents, value)
	}
	for _, expected := range contents {
		stored, found, readErr := artifact.ReadContent(ctx, reader, expected.Descriptor.ID)
		if readErr != nil || !found || stored.Descriptor != expected.Descriptor || !bytes.Equal(stored.Data, expected.Data) {
			return errors.Join(errors.New("loop: supervised composite durable content differs"), readErr)
		}
	}
	for alias, expected := range map[string]artifact.ID{
		proof.IncumbentAlias: proof.Incumbent, proof.AutomationAlias: proof.AutomationGrant,
	} {
		current, found, resolveErr := artifact.ResolveAlias(ctx, reader, alias)
		if resolveErr != nil || !found || current != expected {
			return errors.Join(errors.New("loop: supervised proof changed production authority"), resolveErr)
		}
	}
	return nil
}

// TestSupervisedCompositeFirstWin closes the first local RSI vertical slice
// without live traffic or activation. The same content and aliases must replay
// after a cold reopen.
func TestSupervisedCompositeFirstWin(t *testing.T) {
	path := t.TempDir()
	store, err := overgodb.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	proof := newSupervisedCompositeProof(t, store)
	if err := verifySupervisedCompositeFirstWin(t.Context(), store, proof); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = reopened.Close() })
	if err := verifySupervisedCompositeFirstWin(t.Context(), reopened, proof); err != nil {
		t.Fatal(err)
	}
}

func newSupervisedCompositeProof(t *testing.T, store *overgodb.Store) supervisedCompositeProof {
	t.Helper()
	ctx := t.Context()
	id := func(kind artifact.Kind, name string) artifact.ID {
		return testutil.ArtifactID(t, kind, "supervised composite "+name)
	}
	subject := id(artifact.KindRecipe, "same-base specification")
	parent := id(artifact.KindModelDefinition, "base definition")
	omitted := id(artifact.KindModelDefinition, "specialist delta")
	admission := id(artifact.KindEvidence, "admission")
	falsifier := id(artifact.KindRecipe, "falsifier")
	developmentSplit := id(artifact.KindDatasetShard, "development split")
	promotionSplit := id(artifact.KindDatasetShard, "promotion split")
	developmentBudget := id(artifact.KindEvidence, "development budget")
	promotionBudget := id(artifact.KindEvidence, "promotion budget")
	code := id(artifact.KindEvidence, "code")
	environment := id(artifact.KindEvidence, "environment")
	referenceEvidence := id(artifact.KindEvidence, "baseline evidence")
	candidate, err := modelrecipe.NewCandidate(modelrecipe.CandidateSpec{
		Subject: subject, Parent: parent,
		Components: []modelrecipe.CandidateComponent{{
			Domain: modelrecipe.CandidateComposition, Specification: subject,
		}},
		Prediction: recipe.SteeringPrediction{Metric: "held-out capability", Benefit: 1, Cost: 1, Unit: "ratio"},
		CostUnit:   "queries", Falsifier: falsifier,
		References: []modelrecipe.CandidateReference{{
			Role: modelrecipe.CandidateReferenceBaseline, Subject: parent, Evidence: referenceEvidence,
		}},
		DevelopmentSplit: developmentSplit, PromotionSplit: promotionSplit,
		DevelopmentBudget: developmentBudget, PromotionBudget: promotionBudget,
		Code: code, Environment: environment,
	})
	if err != nil {
		t.Fatal(err)
	}
	compiled := supervisedCandidateCompilation(t, candidate, admission)

	incumbent := id(artifact.KindRecipe, "production incumbent")
	incumbentLifecycle := id(artifact.KindEvidence, "incumbent lifecycle")
	incumbentPlacement := id(artifact.KindProfile, "incumbent placement")
	goal := id(artifact.KindEvidence, "goal")
	trigger := id(artifact.KindEvidence, "trigger")
	causal, err := runrecord.NewCausalRoot(runrecord.TriggerControllerProposal, trigger, goal)
	if err != nil {
		t.Fatal(err)
	}
	interaction, err := runrecord.NewBudget("queries", id(artifact.KindDatasetShard, "interaction split"), 5, id(artifact.KindEvidence, "interaction authority"))
	if err != nil {
		t.Fatal(err)
	}
	resources, err := runrecord.NewBudget("queries", id(artifact.KindDatasetShard, "resource split"), 5, id(artifact.KindEvidence, "resource authority"))
	if err != nil {
		t.Fatal(err)
	}
	candidateContent, err := candidate.Content()
	if err != nil {
		t.Fatal(err)
	}
	interactionContent, _ := interaction.Content()
	resourceContent, _ := resources.Content()
	initialDescriptors := []artifact.Descriptor{
		{ID: subject}, {ID: parent}, {ID: omitted}, {ID: admission}, {ID: falsifier},
		{ID: developmentSplit}, {ID: promotionSplit}, {ID: developmentBudget}, {ID: promotionBudget},
		{ID: code}, {ID: environment}, {ID: referenceEvidence}, {ID: incumbent},
		{ID: incumbentLifecycle}, {ID: incumbentPlacement}, {ID: goal}, {ID: trigger},
		{ID: interaction.Split}, {ID: interaction.Authority}, {ID: resources.Split}, {ID: resources.Authority},
	}
	head, err := commitSupervisedBatch(t, store, "initial", []artifact.Content{
		candidateContent, interactionContent, resourceContent,
	}, append(append(candidate.Lineage(), interaction.Lineage()...), resources.Lineage()...), initialDescriptors, nil)
	if err != nil {
		t.Fatal(err)
	}
	decision, err := runrecord.NewDriverDecision(ctx, store, runrecord.DriverDecisionFacts{
		Goal: goal, Causal: causal, Head: head,
		Options: []runrecord.DriverOption{
			{Recipe: incumbent, Lifecycle: incumbentLifecycle, State: runrecord.DriverIncumbentActive, Placement: incumbentPlacement},
			{Candidate: candidate.ID(), Admission: admission, State: runrecord.DriverCandidateEligible,
				Missing: []runrecord.DriverEvidenceGap{{
					Need: runrecord.DriverNeedRealization, Target: candidate.ID(), CostUnits: 1,
					CostUnit: "queries", CostAuthority: candidate.ID(),
				}}},
		},
		InteractionBudget: runrecord.DriverBudgetReference{Grant: interaction.ID},
		ResourceBudget:    runrecord.DriverBudgetReference{Grant: resources.ID},
	})
	if err != nil {
		t.Fatal(err)
	}
	decisionContent, _ := decision.Content()
	if _, err := commitSupervisedBatch(t, store, "decision", []artifact.Content{decisionContent}, decision.Lineage(), nil, nil); err != nil {
		t.Fatal(err)
	}

	charge, err := runrecord.NewBudgetCharge(resources.ID, 1, decision.ID, "candidate materialization")
	if err != nil {
		t.Fatal(err)
	}
	primaryModel := id(artifact.KindModel, "joint model")
	ablationModel := id(artifact.KindModel, "ablation model")
	primaryInventory := id(artifact.KindTensorInventory, "joint inventory")
	ablationInventory := id(artifact.KindTensorInventory, "ablation inventory")
	primaryDefinition := id(artifact.KindModelDefinition, "joint definition")
	ablationDefinition := id(artifact.KindModelDefinition, "ablation definition")
	primaryArm := supervisedMaterializedArm(t, compiled.Components[0], modelrecipe.CandidateAblation{}, primaryModel, primaryInventory, primaryDefinition, "joint")
	ablationArm := supervisedMaterializedArm(t, compiled.Components[0], compiled.Ablations[0], ablationModel, ablationInventory, ablationDefinition, "ablation")
	materialization, err := modelrecipe.NewCandidateMaterialization(decision, compiled, charge, []modelrecipe.CandidateMaterializedArm{primaryArm, ablationArm})
	if err != nil {
		t.Fatal(err)
	}
	chargeContent, _ := charge.Content()
	materializationContent, _ := materialization.Content()
	armDescriptors := supervisedArmDescriptors(primaryArm, ablationArm)
	for _, component := range compiled.Components {
		for _, id := range append([]artifact.ID{component.Realization, component.ResourcePolicy}, component.Documents...) {
			armDescriptors = append(armDescriptors, artifact.Descriptor{ID: id})
		}
	}
	if _, err := commitSupervisedBatch(t, store, "materialization",
		append(slices.Clone(compiled.Contents), chargeContent, materializationContent),
		append(append(slices.Clone(compiled.Lineage), charge.Lineage()...), materialization.Lineage()...),
		armDescriptors, nil,
	); err != nil {
		t.Fatal(err)
	}

	evaluator := id(artifact.KindProfile, "shared evaluator")
	heldOutInputs := id(artifact.KindDatasetShard, "held-out inputs")
	contract, err := evaluation.NewCrossDomainEvaluationContract(
		decision, compiled, evaluator, heldOutInputs,
		[]evaluation.CandidateFitnessRequirement{
			{Name: "capability", Direction: runrecord.DirectionMaximize, MinimumImprovement: 0.05},
			{Name: "wall-ns", Direction: runrecord.DirectionMinimize, MinimumImprovement: 5},
		}, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	fitness := func(capability, wall float64) []runrecord.Metric {
		return []runrecord.Metric{
			{Name: "capability", Value: capability, Unit: "ratio", Direction: runrecord.DirectionMaximize},
			{Name: "wall-ns", Value: wall, Unit: "ns", Direction: runrecord.DirectionMinimize},
		}
	}
	arm := func(role evaluation.CandidateEvaluationRole, subject artifact.ID, values []runrecord.Metric) evaluation.CandidateEvaluationArm {
		return evaluation.CandidateEvaluationArm{
			Role: role, Subject: subject, Evaluator: evaluator, Inputs: heldOutInputs,
			Split: contract.PromotionSplit, Budget: contract.PromotionBudget,
			Evidence: id(artifact.KindEvidence, "evaluation "+string(role)+subject.String()),
			Fitness:  values, Covered: 16, Total: 16,
		}
	}
	evaluationArms := []evaluation.CandidateEvaluationArm{
		arm(evaluation.CandidateEvaluationIncumbent, incumbent, fitness(0.60, 100)),
		arm(evaluation.CandidateEvaluationParent, parent, fitness(0.64, 96)),
		arm(evaluation.CandidateEvaluationJoint, candidate.ID(), fitness(0.72, 84)),
		arm(evaluation.CandidateEvaluationAblation, compiled.Ablations[0].Realization, fitness(0.63, 93)),
	}
	evaluationArms[2].Dimensions = slices.Clone(contract.CandidateDimensions)
	evaluationArms[3].Ablation = compiled.Ablations[0].ID
	evaluationArms[3].Omitted = compiled.Ablations[0].Omitted
	evaluationArms[3].Dimensions = []evaluation.CandidateDeltaDimension{evaluation.CandidateDeltaComposition}
	evaluated, err := evaluation.EvaluateCrossDomainCandidate(contract, decision, compiled, evaluationArms)
	if err != nil {
		t.Fatal(err)
	}
	contractContent, _ := contract.Content()
	evaluationContent, _ := evaluated.Content()
	evaluationDescriptors := []artifact.Descriptor{{ID: evaluator}, {ID: heldOutInputs}}
	for _, measured := range evaluationArms {
		evaluationDescriptors = append(evaluationDescriptors, artifact.Descriptor{ID: measured.Evidence})
	}
	if _, err := commitSupervisedBatch(t, store, "evaluation", []artifact.Content{contractContent, evaluationContent},
		append(contract.Lineage(), evaluated.Lineage()...), evaluationDescriptors, nil); err != nil {
		t.Fatal(err)
	}

	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskForecast, primaryModel)
	if err != nil {
		t.Fatal(err)
	}
	definitionContent, _ := definition.ArtifactContent()
	strategy := id(artifact.KindProfile, "strategy")
	task := id(artifact.KindRecipe, "verification task")
	operation := id(artifact.KindEvidence, "verification operation")
	arguments := id(artifact.KindEvidence, "verification arguments")
	effect := id(artifact.KindEvidence, "verification effect")
	preflight := id(artifact.KindEvidence, "verification preflight")
	inspection := id(artifact.KindEvidence, "verification inspection")
	causalContext := id(artifact.KindEvidence, "verification causal context")
	requestArtifact := id(artifact.KindEvidence, "trace request")
	if _, err := commitSupervisedBatch(t, store, "verification-authorities", []artifact.Content{definitionContent},
		nil, []artifact.Descriptor{
			{ID: strategy}, {ID: task}, {ID: operation}, {ID: arguments}, {ID: effect},
			{ID: preflight}, {ID: inspection}, {ID: causalContext}, {ID: requestArtifact},
		}, nil); err != nil {
		t.Fatal(err)
	}
	action := operatoraction.Action{
		Code: string(invocation.MutationPromotion), Summary: string(invocation.MutationPromotion),
		Argv: invocation.ApprovalArguments(definition.ID, arguments, effect, preflight),
	}
	request, err := operatoraction.NewApprovalRequest(operation, task, action, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	human, err := runrecord.NewHumanDecision(request, operatoraction.AnswerGrant)
	if err != nil {
		t.Fatal(err)
	}
	if err := runrecord.PublishHumanDecision(ctx, store, request, human); err != nil {
		t.Fatal(err)
	}
	binding := invocation.ReceiptBinding{
		Boundary: invocation.BoundaryInternal, Kind: invocation.MutationPromotion,
		Action: string(invocation.MutationPromotion), Subject: definition.ID,
		Arguments: arguments, Effect: effect, Preflight: preflight, Inspection: inspection,
		Ceiling: task, Authority: human.ID, CausalContext: causalContext, Head: head,
	}
	input := []runrecord.StageBinding{{Port: "evaluation", Artifacts: []artifact.ID{evaluated.ID}}}
	receipts := make([]runrecord.StageReceipt, 3)
	for index, state := range []runrecord.StageState{runrecord.StageAdmitted, runrecord.StageRunning, runrecord.StageCompleted} {
		receipts[index], err = runrecord.PublishStageReceipt(ctx, store, runrecord.StageReceipt{
			Recipe: task, Node: recipe.NodeID(invocation.MutationPromotion), Operation: operation,
			Attempt: 1, State: state, Inputs: input, Invocation: &binding,
		}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	candidateEvent, err := recipe.NewLifecycleEvent(definition, "", recipe.StatusCandidate, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	validatedEvent, err := recipe.NewLifecycleEvent(definition, recipe.StatusCandidate, recipe.StatusValidated, &candidateEvent.ID, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	verifiedEvent, err := recipe.NewLifecycleEvent(definition, recipe.StatusValidated, recipe.StatusVerified,
		&validatedEvent.ID, nil, []artifact.ID{evaluated.ID, receipts[2].ID})
	if err != nil {
		t.Fatal(err)
	}
	lifecycleContents := make([]artifact.Content, 0, 3)
	var lifecycleLineage []artifact.Lineage
	for _, event := range []recipe.LifecycleEvent{candidateEvent, validatedEvent, verifiedEvent} {
		content, contentErr := event.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		lifecycleContents = append(lifecycleContents, content)
		lifecycleLineage = append(lifecycleLineage, event.Lineage()...)
	}
	if _, err := commitSupervisedBatch(t, store, "verified-lifecycle", lifecycleContents, lifecycleLineage, nil, nil); err != nil {
		t.Fatal(err)
	}

	attempt, err := runrecord.NewAttemptRecord(runrecord.AttemptRecord{
		PlanItem: "supervised-composite", PlanStep: "first-win", Strategy: "same-base",
		Result: evaluated.ID, Recipe: contract.ID, CodeCommit: strings.Repeat("a", 40),
		Outcome: runrecord.OutcomeSucceeded, WallNS: 100, StrategyID: strategy,
	})
	if err != nil {
		t.Fatal(err)
	}
	attemptContent, _ := attempt.Content()
	if _, err := commitSupervisedBatch(t, store, "attempt", []artifact.Content{attemptContent}, attempt.Lineage(), nil, nil); err != nil {
		t.Fatal(err)
	}
	trace, err := runrecord.NewAgentTrajectory(runrecord.InteractionTrace{
		Version: artifact.InitialDocumentVersion, Recipe: definition.ID, Model: primaryModel,
		Operation: operation, Request: requestArtifact,
		Events: []runrecord.InteractionTraceEvent{{
			Sequence: 1, Kind: runrecord.InteractionEventRequest,
			Message: runrecord.InteractionMessage{Role: "user", Content: "verify the supervised composite"},
		}},
		Decisions:      []artifact.ID{decision.ID, human.ID},
		FinalArtifacts: []artifact.ID{materialization.ID, evaluated.ID},
		TaskContract:   task, Strategy: strategy, Attempts: []artifact.ID{attempt.ID},
		Terminal: runrecord.OutcomeSucceeded,
	})
	if err != nil {
		t.Fatal(err)
	}
	traceContent, _ := trace.Content()
	if _, err := commitSupervisedBatch(t, store, "trace", []artifact.Content{traceContent}, trace.Lineage(), nil, nil); err != nil {
		t.Fatal(err)
	}
	incumbentAlias := "test/supervised/production-incumbent"
	automationAlias := "test/supervised/automation-grant"
	automationGrant := id(artifact.KindProfile, "unchanged automation grant")
	if _, err := commitSupervisedBatch(t, store, "unchanged-authority", nil, nil,
		[]artifact.Descriptor{{ID: automationGrant}}, []artifact.AliasBinding{
			{Name: incumbentAlias, Target: incumbent}, {Name: automationAlias, Target: automationGrant},
		}); err != nil {
		t.Fatal(err)
	}
	return supervisedCompositeProof{
		Candidate: candidate, Decision: decision, Compilation: compiled,
		Materialization: materialization, Evaluation: evaluated, Definition: definition,
		HumanDecision: human.ID, Receipts: receipts, Lifecycle: verifiedEvent,
		Strategy: strategy, Attempt: attempt, Trace: trace,
		ParentDefinitions: []artifact.ID{parent, ablationDefinition},
		ParentInventories: []artifact.ID{primaryInventory, ablationInventory},
		IncumbentAlias:    incumbentAlias, Incumbent: incumbent,
		AutomationAlias: automationAlias, AutomationGrant: automationGrant,
	}
}

func supervisedCandidateCompilation(
	t *testing.T,
	candidate modelrecipe.Candidate,
	admission artifact.ID,
) modelrecipe.CandidateCompilation {
	t.Helper()
	spec := candidate.Spec()
	omitted := testutil.ArtifactID(t, artifact.KindModelDefinition, "supervised composite specialist delta")
	ablation := modelrecipe.CandidateAblation{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), Admission: admission,
		Domain: modelrecipe.CandidateComposition, Specification: spec.Subject,
		Omitted: omitted, Realization: testutil.ArtifactID(t, artifact.KindProfile, "supervised composite ablation realization"),
	}
	ablation.ID = identifySupervisedJSON(t, artifact.KindProfile, ablation)
	component := modelrecipe.CandidateComponentPlan{
		Version: artifact.InitialDocumentVersion, Domain: modelrecipe.CandidateComposition,
		Specification: spec.Subject, Subject: spec.Subject,
		Realization:    testutil.ArtifactID(t, artifact.KindProfile, "supervised composite joint realization"),
		ResourcePolicy: testutil.ArtifactID(t, artifact.KindProfile, "supervised composite resource policy"),
		Inputs:         []artifact.ID{spec.Parent, omitted}, Documents: []artifact.ID{
			testutil.ArtifactID(t, artifact.KindProfile, "supervised composite joint realization"),
			testutil.ArtifactID(t, artifact.KindProfile, "supervised composite resource policy"),
			ablation.Realization,
		},
		Ablations: []artifact.ID{ablation.ID}, PeakResidentBytes: 8, ArtifactBytes: 8,
	}
	slices.SortFunc(component.Inputs, artifact.CompareID)
	slices.SortFunc(component.Documents, artifact.CompareID)
	component.ID = identifySupervisedJSON(t, artifact.KindProfile, component)
	plan := modelrecipe.CandidateEvaluationPlan{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), Admission: admission,
		Subject: spec.Subject, Parent: spec.Parent, Prediction: spec.Prediction, CostUnit: spec.CostUnit,
		Falsifier: spec.Falsifier, DevelopmentSplit: spec.DevelopmentSplit, PromotionSplit: spec.PromotionSplit,
		DevelopmentBudget: spec.DevelopmentBudget, PromotionBudget: spec.PromotionBudget,
		ComponentPlans: []artifact.ID{component.ID}, Ablations: []artifact.ID{ablation.ID},
		CausalReferences: slices.Clone(spec.References), Code: spec.Code, Environment: spec.Environment,
	}
	plan.ID = identifySupervisedJSON(t, artifact.KindProfile, plan)
	trial := modelrecipe.CandidateTrial{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), Admission: admission,
		Subject: spec.Subject, Parent: spec.Parent, EvaluationPlan: plan.ID,
		Components: []artifact.ID{component.ID}, Ablations: []artifact.ID{ablation.ID},
		Code: spec.Code, Environment: spec.Environment,
	}
	trial.ID = identifySupervisedJSON(t, artifact.KindProfile, trial)
	contents := []artifact.Content{
		supervisedDocumentContent(t, artifact.KindProfile, modelrecipe.CandidateComponentPlanMediaType, modelrecipe.CandidateComponentPlanSchema, component.ID, component),
		supervisedDocumentContent(t, artifact.KindProfile, modelrecipe.CandidateAblationMediaType, modelrecipe.CandidateAblationSchema, ablation.ID, ablation),
		supervisedDocumentContent(t, artifact.KindProfile, modelrecipe.CandidateEvaluationPlanMediaType, modelrecipe.CandidateEvaluationPlanSchema, plan.ID, plan),
		supervisedDocumentContent(t, artifact.KindProfile, modelrecipe.CandidateTrialMediaType, modelrecipe.CandidateTrialSchema, trial.ID, trial),
	}
	lineage := append(append(append(component.Lineage(), ablation.Lineage()...), plan.Lineage()...), trial.Lineage()...)
	compiled := modelrecipe.CandidateCompilation{
		Trial: trial, EvaluationPlan: plan, Components: []modelrecipe.CandidateComponentPlan{component},
		Ablations: []modelrecipe.CandidateAblation{ablation}, Contents: contents, Lineage: lineage,
	}
	if err := compiled.ValidateIdentity(); err != nil {
		t.Fatal(err)
	}
	return compiled
}

func supervisedMaterializedArm(
	t *testing.T,
	component modelrecipe.CandidateComponentPlan,
	ablation modelrecipe.CandidateAblation,
	model, inventory, definition artifact.ID,
	label string,
) modelrecipe.CandidateMaterializedArm {
	t.Helper()
	realization := component.Realization
	if ablation.ID.Valid() {
		realization = ablation.Realization
	}
	return modelrecipe.CandidateMaterializedArm{
		Component: component.ID, Ablation: ablation.ID, Omitted: ablation.Omitted, Realization: realization,
		Model: model, TensorInventory: inventory, ModelDefinition: definition,
		Output:      testutil.ArtifactID(t, artifact.KindOutput, "supervised composite "+label+" output"),
		Run:         testutil.ArtifactID(t, artifact.KindRun, "supervised composite "+label+" run"),
		Observation: testutil.ArtifactID(t, artifact.KindEvidence, "supervised composite "+label+" observation"),
		TensorBytes: 4, StoredBytes: 4, PeakResidentBytes: 4,
	}
}

func supervisedArmDescriptors(arms ...modelrecipe.CandidateMaterializedArm) []artifact.Descriptor {
	var result []artifact.Descriptor
	for _, arm := range arms {
		for _, id := range []artifact.ID{
			arm.Model, arm.TensorInventory, arm.ModelDefinition, arm.Output, arm.Run, arm.Observation,
		} {
			result = append(result, artifact.Descriptor{ID: id})
		}
	}
	return result
}

func identifySupervisedJSON(t *testing.T, kind artifact.Kind, value any) artifact.ID {
	t.Helper()
	id, err := artifact.JSONID(kind, value)
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func supervisedDocumentContent(
	t *testing.T,
	kind artifact.Kind,
	mediaType, schema string,
	id artifact.ID,
	value any,
) artifact.Content {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	content, err := (artifact.DocumentContract{Kind: kind, MediaType: mediaType, Schema: schema}).Content(id, data)
	if err != nil {
		t.Fatal(err)
	}
	return content
}

func commitSupervisedBatch(
	t *testing.T,
	store *overgodb.Store,
	key string,
	contents []artifact.Content,
	lineage []artifact.Lineage,
	descriptors []artifact.Descriptor,
	aliases []artifact.AliasBinding,
) (artifact.CommitID, error) {
	t.Helper()
	return store.Commit(t.Context(), artifact.Batch{
		Key: "supervised-composite/" + key, Contents: contents, Lineage: lineage,
		Artifacts: descriptors, Aliases: aliases,
	})
}
