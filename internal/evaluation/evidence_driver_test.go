package evaluation

import (
	"cmp"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type evidenceDriverFixture struct {
	store             *overgodb.Store
	request           EvidenceDriverRequest
	probe             CapabilityProbeResult
	production        productionProbeFixture
	codeCandidate     modelrecipe.Candidate
	codeAdmission     runrecord.CandidateAdmission
	composition       modelrecipe.Candidate
	compositionPermit runrecord.CandidateAdmission
	unadmitted        modelrecipe.Candidate
	unrelated         modelrecipe.Candidate
	unrelatedPermit   runrecord.CandidateAdmission
	invalidIntent     modelrecipe.Candidate
	invalidPermit     runrecord.CandidateAdmission
}

func TestEvidenceDriverConsumesCrossDomainCandidates(t *testing.T) {
	fixture := newEvidenceDriverFixture(t)
	decision, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, fixture.request)
	if err != nil {
		t.Fatal(err)
	}

	activation, active, err := modelrecipe.ActiveRecord(
		t.Context(), fixture.store, fixture.production.model, fixture.production.testCase.Task,
	)
	if err != nil || !active {
		t.Fatalf("active incumbent = (%+v, %t, %v)", activation, active, err)
	}
	incumbent := evidenceDriverOption(t, decision, activation.Definition.ID)
	if incumbent.State != runrecord.DriverIncumbentActive || incumbent.Lifecycle != activation.Event.ID ||
		incumbent.Placement != fixture.production.request.Placement.ID ||
		!slices.Equal(incumbent.Evidence, []artifact.ID{fixture.probe.ID}) {
		t.Fatalf("incumbent option = %+v", incumbent)
	}

	for _, expected := range []struct {
		candidate modelrecipe.Candidate
		admission runrecord.CandidateAdmission
		domain    modelrecipe.CandidateDomain
	}{
		{fixture.codeCandidate, fixture.codeAdmission, modelrecipe.CandidateCode},
		{fixture.composition, fixture.compositionPermit, modelrecipe.CandidateComposition},
	} {
		option := evidenceDriverOption(t, decision, expected.candidate.ID())
		spec := expected.candidate.Spec()
		if len(spec.Components) != 1 || spec.Components[0].Domain != expected.domain ||
			option.Admission != expected.admission.ID || option.State != runrecord.DriverCandidateEligible ||
			option.Recipe.Valid() || option.Lifecycle.Valid() || option.Placement.Valid() ||
			len(option.Missing) != 1 || option.Missing[0].Need != runrecord.DriverNeedRealization ||
			option.Missing[0].Target != expected.candidate.ID() ||
			option.Missing[0].CostUnits != spec.Prediction.Cost ||
			option.Missing[0].CostUnit != spec.CostUnit ||
			option.Missing[0].CostAuthority != expected.candidate.ID() {
			t.Fatalf("%s option = %+v; spec = %+v", expected.domain, option, spec)
		}
	}
}

func TestEvidenceDerivedSelection(t *testing.T) {
	fixture := newEvidenceDriverFixture(t)
	selected, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, fixture.request)
	if err != nil {
		t.Fatal(err)
	}
	if selected.Stop != runrecord.DriverStopContinue || selected.Action != runrecord.DriverActionRealize ||
		selected.Goal != fixture.probe.CheckDecision ||
		selected.Subject != fixture.composition.ID() || selected.Target != fixture.composition.ID() ||
		selected.Need != runrecord.DriverNeedRealization ||
		fixture.composition.Spec().Prediction.Cost >= fixture.codeCandidate.Spec().Prediction.Cost ||
		fixture.codeCandidate.Spec().Prediction.Cost <= selected.Resources.Remaining {
		t.Fatalf("lowest affordable selection = %+v", selected)
	}

	permutedRequest := fixture.request
	permutedRequest.Candidates = slices.Clone(fixture.request.Candidates)
	slices.Reverse(permutedRequest.Candidates)
	permuted, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, permutedRequest)
	if err != nil || permuted.ID != selected.ID || permuted.Subject != selected.Subject {
		t.Fatalf("permuted selection = (%+v, %v), want %s", permuted, err, selected.ID)
	}

	t.Run("forged incumbent", func(t *testing.T) {
		request := fixture.request
		request.Incumbent = testutil.ArtifactID(t, artifact.KindEvidence, "forged driver incumbent")
		if _, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, request); err == nil {
			t.Fatal("forged incumbent was accepted")
		}
	})
	t.Run("foreign production goal", func(t *testing.T) {
		foreign := newProductionProbeFixtureInStore(t, fixture.store, "evidence-driver-foreign-goal")
		probe, _, err := PublishProductionCapabilityProbe(t.Context(), fixture.store, foreign.request)
		if err != nil {
			t.Fatal(err)
		}
		request := fixture.request
		request.Goal = probe.CheckDecision
		if _, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, request); err == nil {
			t.Fatal("another valid production check was accepted as the incumbent goal")
		}
	})
	t.Run("foreign admission", func(t *testing.T) {
		request := fixture.request
		request.Candidates = []EvidenceDriverCandidate{{
			Candidate: fixture.composition.ID(), Admission: fixture.codeAdmission.ID,
		}}
		if _, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, request); err == nil {
			t.Fatal("candidate paired with a foreign admission was accepted")
		}
	})
	t.Run("unadmitted candidate", func(t *testing.T) {
		request := fixture.request
		request.Candidates = []EvidenceDriverCandidate{{
			Candidate: fixture.unadmitted.ID(),
			Admission: testutil.ArtifactID(t, artifact.KindEvidence, "absent driver admission"),
		}}
		if _, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, request); err == nil {
			t.Fatal("unadmitted candidate was accepted")
		}
	})
	t.Run("admitted candidate unrelated to goal", func(t *testing.T) {
		request := fixture.request
		request.Candidates = []EvidenceDriverCandidate{{
			Candidate: fixture.unrelated.ID(), Admission: fixture.unrelatedPermit.ID,
		}}
		if _, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, request); err == nil {
			t.Fatal("admitted candidate with a foreign baseline goal was accepted")
		}
	})
	t.Run("candidate evaluation intent differs", func(t *testing.T) {
		request := fixture.request
		request.Candidates = []EvidenceDriverCandidate{{
			Candidate: fixture.invalidIntent.ID(), Admission: fixture.invalidPermit.ID,
		}}
		if _, err := CompileEvidenceDriverDecision(t.Context(), fixture.store, request); err == nil {
			t.Fatal("candidate with a mismatched typed evaluation intent was accepted")
		}
	})
	t.Run("stale incumbent", func(t *testing.T) {
		stale := newEvidenceDriverFixture(t)
		previous := stale.production.model
		if _, err := stale.store.Commit(t.Context(), artifact.Batch{
			Key: "evidence-driver/stale-incumbent",
			Aliases: []artifact.AliasBinding{{
				Name: stale.production.alias, Target: stale.production.model, Previous: &previous, Remove: true,
			}},
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := CompileEvidenceDriverDecision(t.Context(), stale.store, stale.request); err == nil {
			t.Fatal("stale incumbent placement authority was accepted")
		}
	})
}

func newEvidenceDriverFixture(t *testing.T) evidenceDriverFixture {
	t.Helper()
	ctx := t.Context()
	store := newCapabilityEvaluationStore(t)
	production := newProductionProbeFixtureInStore(t, store, "evidence-driver-incumbent")
	probe, _, err := PublishProductionCapabilityProbe(ctx, store, production.request)
	if err != nil {
		t.Fatal(err)
	}

	promotionSplit := testutil.ArtifactID(t, artifact.KindDatasetShard, "evidence driver promotion split")
	developmentSplit := testutil.ArtifactID(t, artifact.KindDatasetShard, "evidence driver development split")
	identifier := func(label string) artifact.ID {
		return testutil.ArtifactID(t, artifact.KindEvidence, "evidence driver "+label)
	}
	binding, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Proposer:     runrecord.AuthorityDomain{Name: "driver-proposer", Identity: identifier("proposer")},
		Evaluator:    runrecord.AuthorityDomain{Name: "driver-evaluator", Identity: identifier("evaluator")},
		Decider:      runrecord.AuthorityDomain{Name: "driver-decider", Identity: identifier("decider")},
		SealedInputs: promotionSplit,
		CleanWorker:  identifier("clean worker"),
	})
	if err != nil {
		t.Fatal(err)
	}
	code, err := runrecord.NewCodeRevision(strings.Repeat("d", 40))
	if err != nil {
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "evidence-driver", OS: "test", Arch: "test", Device: "host",
		Backend: "go", Driver: "test", Runtime: "go-test",
	})
	if err != nil {
		t.Fatal(err)
	}
	bindingContent, err := binding.Content()
	if err != nil {
		t.Fatal(err)
	}
	codeBatch, err := code.Publication()
	if err != nil {
		t.Fatal(err)
	}
	environmentContent, err := environment.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "evidence-driver/common-authorities",
		Artifacts: []artifact.Descriptor{
			{ID: promotionSplit}, {ID: developmentSplit},
			{ID: binding.Proposer.Identity}, {ID: binding.Evaluator.Identity},
			{ID: binding.Decider.Identity}, {ID: binding.CleanWorker},
		},
		Contents: append([]artifact.Content{bindingContent, environmentContent}, codeBatch.Contents...),
	}); err != nil {
		t.Fatal(err)
	}

	base := publishEvidenceDriverModelDefinition(t, store, "base")
	delta := publishEvidenceDriverModelDefinition(t, store, "delta")
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "evidence-driver/composition-derivation",
		Lineage: []artifact.Lineage{
			{Child: delta.Document.ID, Parent: base.Document.ID, Relation: artifact.RelationDerivedFrom},
			{Child: delta.Document.Model, Parent: base.Document.Model, Relation: artifact.RelationDerivedFrom},
		},
	}); err != nil {
		t.Fatal(err)
	}
	compositionSpec := newEvidenceDriverCompositionSpec(t, base.Document.ID, delta.Document.ID)
	compositionContent, err := compositionSpec.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:      "evidence-driver/composition-spec",
		Contents: []artifact.Content{compositionContent}, Lineage: compositionSpec.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}

	codeSubject := evidenceDriverContent(t, artifact.KindProfile, "code subject")
	codeParent := evidenceDriverContent(t, artifact.KindProfile, "code parent")
	spareSubject := evidenceDriverContent(t, artifact.KindProfile, "unadmitted code subject")
	unrelatedSubject := evidenceDriverContent(t, artifact.KindProfile, "unrelated code subject")
	invalidIntentSubject := evidenceDriverContent(t, artifact.KindProfile, "invalid intent code subject")
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:      "evidence-driver/code-subjects",
		Contents: []artifact.Content{codeSubject, codeParent, spareSubject, unrelatedSubject, invalidIntentSubject},
	}); err != nil {
		t.Fatal(err)
	}

	cheapCost := uint64(artifact.InitialDocumentVersion)
	resourceLimit := uint64(artifact.SecondDocumentVersion)
	expensiveCost := cheapCost + resourceLimit
	compositionCandidate, compositionAdmission := publishEvidenceDriverCandidate(
		t, store, binding, developmentSplit, promotionSplit, code.ID, environment.ID,
		modelrecipe.CandidateComponent{Domain: modelrecipe.CandidateComposition, Specification: compositionSpec.ID},
		compositionSpec.ID, base.Document.ID, cheapCost, probe.CheckDecision, "", "composition", true,
	)
	codeCandidate, codeAdmission := publishEvidenceDriverCandidate(
		t, store, binding, developmentSplit, promotionSplit, code.ID, environment.ID,
		modelrecipe.CandidateComponent{Domain: modelrecipe.CandidateCode, Specification: code.ID},
		codeSubject.Descriptor.ID, codeParent.Descriptor.ID, expensiveCost, probe.CheckDecision, "", "code", true,
	)
	unadmitted, _ := publishEvidenceDriverCandidate(
		t, store, binding, developmentSplit, promotionSplit, code.ID, environment.ID,
		modelrecipe.CandidateComponent{Domain: modelrecipe.CandidateCode, Specification: code.ID},
		spareSubject.Descriptor.ID, codeParent.Descriptor.ID, resourceLimit, probe.CheckDecision, "", "unadmitted", false,
	)
	unrelated, unrelatedPermit := publishEvidenceDriverCandidate(
		t, store, binding, developmentSplit, promotionSplit, code.ID, environment.ID,
		modelrecipe.CandidateComponent{Domain: modelrecipe.CandidateCode, Specification: code.ID},
		unrelatedSubject.Descriptor.ID, codeParent.Descriptor.ID, cheapCost, artifact.ID{}, "", "unrelated", true,
	)
	invalidIntent, invalidPermit := publishEvidenceDriverCandidate(
		t, store, binding, developmentSplit, promotionSplit, code.ID, environment.ID,
		modelrecipe.CandidateComponent{Domain: modelrecipe.CandidateCode, Specification: code.ID},
		invalidIntentSubject.Descriptor.ID, codeParent.Descriptor.ID, cheapCost,
		probe.CheckDecision, "foreign held-out capability", "invalid-intent", true,
	)

	root := evidenceDriverContent(t, artifact.KindEvidence, "causal root")
	causal, err := runrecord.NewCausalRoot(
		runrecord.TriggerControllerProposal, root.Descriptor.ID, probe.CheckDecision,
	)
	if err != nil {
		t.Fatal(err)
	}
	interaction, err := runrecord.NewBudget(
		"queries", testutil.ArtifactID(t, artifact.KindDatasetShard, "driver interaction split"),
		cheapCost, identifier("interaction authority"),
	)
	if err != nil {
		t.Fatal(err)
	}
	resources, err := runrecord.NewBudget(
		"queries", testutil.ArtifactID(t, artifact.KindDatasetShard, "driver resource split"),
		unadmitted.Spec().Prediction.Cost, identifier("resource authority"),
	)
	if err != nil {
		t.Fatal(err)
	}
	interactionContent, err := interaction.Content()
	if err != nil {
		t.Fatal(err)
	}
	resourceContent, err := resources.Content()
	if err != nil {
		t.Fatal(err)
	}
	head, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "evidence-driver/request-authorities",
		Artifacts: []artifact.Descriptor{
			{ID: interaction.Split}, {ID: interaction.Authority},
			{ID: resources.Split}, {ID: resources.Authority},
		},
		Contents: []artifact.Content{root, interactionContent, resourceContent},
		Lineage:  append(interaction.Lineage(), resources.Lineage()...),
	})
	if err != nil {
		t.Fatal(err)
	}
	return evidenceDriverFixture{
		store: store, probe: probe, production: production,
		codeCandidate: codeCandidate, codeAdmission: codeAdmission,
		composition: compositionCandidate, compositionPermit: compositionAdmission,
		unadmitted: unadmitted, unrelated: unrelated, unrelatedPermit: unrelatedPermit,
		invalidIntent: invalidIntent, invalidPermit: invalidPermit,
		request: EvidenceDriverRequest{
			Goal: probe.CheckDecision, Causal: causal, Head: head, Incumbent: probe.ID,
			Candidates: []EvidenceDriverCandidate{
				{Candidate: codeCandidate.ID(), Admission: codeAdmission.ID},
				{Candidate: compositionCandidate.ID(), Admission: compositionAdmission.ID},
			},
			InteractionBudget: interaction.ID, ResourceBudget: resources.ID,
		},
	}
}

func publishEvidenceDriverCandidate(
	t *testing.T,
	store *overgodb.Store,
	binding runrecord.AdmissionBinding,
	developmentSplit, promotionSplit, code, environment artifact.ID,
	component modelrecipe.CandidateComponent,
	subject, parent artifact.ID,
	cost uint64,
	baselineGoal artifact.ID,
	falsifierMetric string,
	label string,
	admit bool,
) (modelrecipe.Candidate, runrecord.CandidateAdmission) {
	t.Helper()
	prediction := recipe.SteeringPrediction{
		Metric: "held-out capability", Benefit: 1, Cost: cost, Unit: "ratio",
	}
	development, err := runrecord.NewBudget("queries", developmentSplit, cost, binding.Decider.Identity)
	if err != nil {
		t.Fatal(err)
	}
	promotion, err := runrecord.NewBudget("queries", promotionSplit, cost, binding.Decider.Identity)
	if err != nil {
		t.Fatal(err)
	}
	falsifierMetric = cmp.Or(falsifierMetric, prediction.Metric)
	falsifier, err := modelrecipe.NewCandidateEvaluationIntent(
		subject, falsifierMetric, "queries", modelrecipe.CandidateStopOnBudgetOrNonPositiveIsolatedImprovement,
	)
	if err != nil {
		t.Fatal(err)
	}
	componentSubject := component.Specification
	type observedReference struct {
		role     modelrecipe.CandidateReferenceRole
		subject  artifact.ID
		decision recipe.Decision
		source   artifact.Content
	}
	references := []observedReference{
		{role: modelrecipe.CandidateReferenceBaseline, subject: parent},
		{role: modelrecipe.CandidateReferenceGap, subject: subject},
		{role: modelrecipe.CandidateReferenceProvenance, subject: componentSubject},
	}
	for index := range references {
		references[index].source = evidenceDriverContent(
			t, artifact.KindEvidence, label+" reference "+string(references[index].role),
		)
		evidence := []artifact.ID{references[index].source.Descriptor.ID}
		if references[index].role == modelrecipe.CandidateReferenceBaseline && baselineGoal.Valid() {
			evidence = append(evidence, baselineGoal)
		}
		references[index].decision, err = recipe.NewDecision(
			references[index].subject, recipe.DecisionObserved, recipe.EvidenceVerified, "",
			recipe.Decider{CodeCommit: strings.Repeat("e", 40), Derivation: binding.Evaluator.Identity},
			evidence,
		)
		if err != nil {
			t.Fatal(err)
		}
	}
	candidateReferences := make([]modelrecipe.CandidateReference, len(references))
	for index, reference := range references {
		candidateReferences[index] = modelrecipe.CandidateReference{
			Role: reference.role, Subject: reference.subject, Evidence: reference.decision.ID,
		}
	}
	candidate, err := modelrecipe.NewCandidate(modelrecipe.CandidateSpec{
		Subject: subject, Parent: parent, Components: []modelrecipe.CandidateComponent{component},
		Prediction: prediction, CostUnit: "queries", Falsifier: falsifier.ID,
		References: candidateReferences, DevelopmentSplit: developmentSplit, PromotionSplit: promotionSplit,
		DevelopmentBudget: development.ID, PromotionBudget: promotion.ID, Code: code, Environment: environment,
	})
	if err != nil {
		t.Fatal(err)
	}
	developmentContent, err := development.Content()
	if err != nil {
		t.Fatal(err)
	}
	promotionContent, err := promotion.Content()
	if err != nil {
		t.Fatal(err)
	}
	falsifierContent, err := falsifier.Content()
	if err != nil {
		t.Fatal(err)
	}
	candidateContent, err := candidate.Content()
	if err != nil {
		t.Fatal(err)
	}
	contents := []artifact.Content{developmentContent, promotionContent, falsifierContent, candidateContent}
	lineage := append(development.Lineage(), promotion.Lineage()...)
	lineage = append(lineage, falsifier.Lineage()...)
	lineage = append(lineage, candidate.Lineage()...)
	for _, reference := range references {
		decisionContent, contentErr := reference.decision.Content()
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		contents = append(contents, reference.source, decisionContent)
		lineage = append(lineage, reference.decision.Lineage()...)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "evidence-driver/candidate/" + label, Contents: contents, Lineage: lineage,
	}); err != nil {
		t.Fatal(err)
	}
	if !admit {
		return candidate, runrecord.CandidateAdmission{}
	}
	adapters := append(modelrecipe.CandidateAdmissionAdapters(), composition.CandidateAdmissionAdapter())
	admission, err := runrecord.AdmitCandidate(t.Context(), store, candidate, binding.ID, adapters...)
	if err != nil {
		t.Fatal(err)
	}
	admissionContent, err := admission.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key:      "evidence-driver/admission/" + label,
		Contents: []artifact.Content{admissionContent}, Lineage: admission.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	return candidate, admission
}

func publishEvidenceDriverModelDefinition(
	t *testing.T,
	store *overgodb.Store,
	label string,
) modelrecipe.ResolvedModelDefinition {
	t.Helper()
	modelID := testutil.ArtifactID(t, artifact.KindModel, "evidence driver composition "+label)
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key: "evidence-driver/model-parent/" + label, Artifacts: []artifact.Descriptor{{ID: modelID}},
	}); err != nil {
		t.Fatal(err)
	}
	seed, err := modelrecipetest.PublishModelDefinition(
		t.Context(), store, "evidence-driver/model-seed/"+label, modelID,
	)
	if err != nil {
		t.Fatal(err)
	}
	const tensorBytes = uint64(4)
	tensors, err := modelartifact.NewTensorInventoryDocument(
		modelID, modelartifact.TensorFormatSafetensors,
		[]modelartifact.TensorFact{{Name: "weight", Shape: []uint64{1, 1}, Storage: "f32", Bytes: tensorBytes}},
	)
	if err != nil {
		t.Fatal(err)
	}
	document, err := modelrecipe.NewModelDefinitionDocument(seed.Profile, tensors, seed.Spec)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := document.Resolve(seed.Profile, tensors)
	if err != nil {
		t.Fatal(err)
	}
	tensorContent, err := tensors.Content()
	if err != nil {
		t.Fatal(err)
	}
	documentContent, err := document.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key:      "evidence-driver/model/" + label,
		Contents: []artifact.Content{tensorContent, documentContent},
		Lineage: []artifact.Lineage{
			{Child: tensors.ID, Parent: modelID, Relation: artifact.RelationDerivedFrom},
			{Child: document.ID, Parent: modelID, Relation: artifact.RelationDerivedFrom},
			{Child: document.ID, Parent: seed.Profile.ID, Relation: artifact.RelationDependsOn},
			{Child: document.ID, Parent: tensors.ID, Relation: artifact.RelationDependsOn},
		},
	}); err != nil {
		t.Fatal(err)
	}
	return resolved
}

func newEvidenceDriverCompositionSpec(
	t *testing.T,
	base, delta artifact.ID,
) composition.SameBaseTaskArithmeticSpec {
	t.Helper()
	const tensorBytes = uint64(4)
	value := composition.SameBaseTaskArithmeticSpec{
		Version: artifact.InitialDocumentVersion, BaseDefinition: base,
		Deltas:       []composition.SameBaseTaskDelta{{Definition: delta, Numerator: 1, Denominator: 2}},
		OutputFormat: modelartifact.TensorFormatSafetensors, Placement: recipe.PlacementHost,
		MaxResidentBytes: tensorBytes * 2, MaxShardBytes: tensorBytes, MaxArtifactBytes: tensorBytes * 2,
		Rationale:     "bounded by the exact typed tensor inventories",
		ReopenTrigger: "recompile when either exact model definition changes",
	}
	id, err := artifact.JSONID(artifact.KindRecipe, value)
	if err != nil {
		t.Fatal(err)
	}
	value.ID = id
	if _, err := value.Content(); err != nil {
		t.Fatal(err)
	}
	return value
}

func evidenceDriverContent(t *testing.T, kind artifact.Kind, label string) artifact.Content {
	t.Helper()
	content, err := artifact.JSONContent(
		artifact.JSONContract(kind, "overgo.evidence-driver-test.v1"),
		struct {
			Label string `json:"label"`
		}{Label: label},
	)
	if err != nil {
		t.Fatal(err)
	}
	return content
}

func evidenceDriverOption(
	t *testing.T,
	decision runrecord.DriverDecision,
	subject artifact.ID,
) runrecord.DriverOption {
	t.Helper()
	for _, option := range decision.Options {
		if option.Candidate == subject || option.Recipe == subject {
			return option
		}
	}
	t.Fatalf("driver option for %s is absent: %+v", subject, decision.Options)
	return runrecord.DriverOption{}
}
