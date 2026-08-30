package composition

import (
	"bytes"
	"cmp"
	"context"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestCandidateDomainPluginsCompileWithoutParallelRuntime(t *testing.T) {
	fixture, err := newSameBaseCandidateFixture(t, sameBaseCandidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer fixture.store.Close()

	spy := &candidatePluginSpy{delegate: SameBaseCandidatePlugin{}, mutateFacts: true}
	compiled, err := modelrecipe.CompileAdmittedCandidate(
		t.Context(), fixture.store, fixture.candidate.ID(), fixture.admission.ID, spy,
	)
	if err != nil {
		t.Fatal(err)
	}
	if spy.calls != 1 || len(compiled.Components) != 1 || len(compiled.Ablations) != 1 {
		t.Fatalf("plugin dispatch = calls %d, components %d, ablations %d",
			spy.calls, len(compiled.Components), len(compiled.Ablations))
	}
	if !slices.Equal(compiled.EvaluationPlan.CausalReferences, fixture.candidate.Spec().References) {
		t.Fatal("plugin mutation escaped its isolated compile-facts copy")
	}
	before := make(map[artifact.ID][]byte, len(compiled.Contents))
	for _, content := range compiled.Contents {
		before[content.Descriptor.ID] = bytes.Clone(content.Data)
	}
	if len(spy.retained) == 0 {
		t.Fatal("spy did not retain a plugin-owned content buffer")
	}
	spy.retained[0] ^= 0xff
	for _, content := range compiled.Contents {
		if !bytes.Equal(content.Data, before[content.Descriptor.ID]) {
			t.Fatalf("plugin retained mutable compilation content %s", content.Descriptor.ID)
		}
	}

	priorCalls := spy.calls
	if _, err := modelrecipe.CompileAdmittedCandidate(
		t.Context(), fixture.store, fixture.candidate.ID(), fixture.admission.ID,
		spy, SameBaseCandidatePlugin{},
	); err == nil || spy.calls != priorCalls {
		t.Fatalf("duplicate compiled domain catalog was dispatched: calls %d, error %v", spy.calls, err)
	}
	if _, err := modelrecipe.CompileAdmittedCandidate(
		t.Context(), fixture.store, fixture.candidate.ID(), fixture.admission.ID,
	); err == nil {
		t.Fatal("candidate compiled without its declared Go domain plugin")
	}

	for _, content := range compiled.Contents {
		if content.Descriptor.MediaType != modelrecipe.CandidateTrialMediaType &&
			content.Descriptor.MediaType != modelrecipe.CandidateEvaluationPlanMediaType &&
			content.Descriptor.MediaType != modelrecipe.CandidateComponentPlanMediaType &&
			content.Descriptor.MediaType != modelrecipe.CandidateAblationMediaType {
			continue
		}
		body := string(content.Data)
		for _, forbidden := range []string{"callback", "alias", "activation", "transport", "runtime"} {
			if strings.Contains(body, forbidden) {
				t.Fatalf("common compile document carries %q runtime authority: %s", forbidden, body)
			}
		}
	}
}

func TestPrototypeCompilesClosedWorldTrial(t *testing.T) {
	fixture, err := newSameBaseCandidateFixture(t, sameBaseCandidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer fixture.store.Close()

	first, err := modelrecipe.CompileAdmittedCandidate(
		t.Context(), fixture.store, fixture.candidate.ID(), fixture.admission.ID,
		SameBaseCandidatePlugin{},
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := modelrecipe.CompileAdmittedCandidate(
		t.Context(), fixture.store, fixture.candidate.ID(), fixture.admission.ID,
		SameBaseCandidatePlugin{},
	)
	if err != nil || first.Trial.ID != second.Trial.ID ||
		first.EvaluationPlan.ID != second.EvaluationPlan.ID {
		t.Fatalf("candidate compilation is not deterministic: (%s, %s, %v)",
			first.Trial.ID, second.Trial.ID, err)
	}
	if len(first.Components) != 1 || len(first.Ablations) != len(fixture.spec.Deltas) ||
		!slices.Equal(first.Trial.Components, first.EvaluationPlan.ComponentPlans) ||
		!slices.Equal(first.Trial.Ablations, first.EvaluationPlan.Ablations) {
		t.Fatalf("closed-world trial = %+v; evaluation = %+v", first.Trial, first.EvaluationPlan)
	}
	component := first.Components[0]
	tensorBytes := offlineTensorWidth * offlineTensorWidth * offlineFloatBytes
	if component.Domain != modelrecipe.CandidateComposition || component.Specification != fixture.spec.ID ||
		component.PeakResidentBytes != tensorBytes*2 || component.ArtifactBytes != tensorBytes*2 ||
		component.ResourcePolicy.Kind() != artifact.KindProfile || component.Realization.Kind() != artifact.KindProfile ||
		!slices.Contains(component.Inputs, fixture.base) || !slices.Contains(component.Inputs, fixture.delta) ||
		component.Subject != fixture.spec.ID {
		t.Fatalf("compiled component plan = %+v", component)
	}
	if first.EvaluationPlan.DevelopmentBudget != fixture.candidate.Spec().DevelopmentBudget ||
		first.EvaluationPlan.PromotionBudget != fixture.candidate.Spec().PromotionBudget ||
		first.EvaluationPlan.Falsifier != fixture.evaluator.ID ||
		!slices.Equal(first.EvaluationPlan.CausalReferences, fixture.candidate.Spec().References) {
		t.Fatalf("common evaluation authority = %+v", first.EvaluationPlan)
	}
	if first.Ablations[0].Omitted != fixture.delta ||
		first.Ablations[0].Realization == component.Realization {
		t.Fatalf("drop-delta ablation = %+v", first.Ablations[0])
	}

	if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
		Key: "fixture/candidate-compile/closed-world", Contents: first.Contents, Lineage: first.Lineage,
	}); err != nil {
		t.Fatal(err)
	}
	replayed, err := modelrecipe.CompileAdmittedCandidate(
		t.Context(), fixture.store, first.Trial.Candidate, first.Trial.Admission,
		SameBaseCandidatePlugin{},
	)
	if err != nil || replayed.Trial.ID != first.Trial.ID {
		t.Fatalf("cold trial replay = (%+v, %v)", replayed, err)
	}
	extraParent := testutil.ArtifactID(t, artifact.KindEvidence, "candidate compile extra realization lineage")
	if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
		Key:       "fixture/candidate-compile/extra-realization-lineage",
		Artifacts: []artifact.Descriptor{{ID: extraParent}},
		Lineage: []artifact.Lineage{{
			Child: component.Realization, Parent: extraParent, Relation: artifact.RelationDependsOn,
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := modelrecipe.CompileAdmittedCandidate(
		t.Context(), fixture.store, first.Trial.Candidate, first.Trial.Admission,
		SameBaseCandidatePlugin{},
	); err == nil || !strings.Contains(err.Error(), "lineage differs") {
		t.Fatalf("cold replay accepted added domain lineage: %v", err)
	}

	t.Run("missing exact base lineage", func(t *testing.T) {
		broken, admitErr := newSameBaseCandidateFixture(t, sameBaseCandidateOptions{omitDerivation: true})
		defer broken.store.Close()
		if admitErr == nil || !strings.Contains(admitErr.Error(), "exact base") {
			t.Fatalf("unrelated delta admitted: %v", admitErr)
		}
	})

	t.Run("incompatible inventory", func(t *testing.T) {
		broken, admitErr := newSameBaseCandidateFixture(t, sameBaseCandidateOptions{
			deltaWidth: offlineTensorWidth + 1,
		})
		defer broken.store.Close()
		if admitErr == nil || !strings.Contains(admitErr.Error(), "incompatible") {
			t.Fatalf("inventory drift admitted: %v", admitErr)
		}
	})

	t.Run("ambiguous exact base", func(t *testing.T) {
		broken, admitErr := newSameBaseCandidateFixture(t, sameBaseCandidateOptions{extraDerivation: true})
		defer broken.store.Close()
		if admitErr == nil || !strings.Contains(admitErr.Error(), "exact base") {
			t.Fatalf("multiply-derived delta admitted: %v", admitErr)
		}
	})

	t.Run("resident budget", func(t *testing.T) {
		bounded, admitErr := newSameBaseCandidateFixture(t, sameBaseCandidateOptions{maxResidentBytes: 1})
		if admitErr != nil {
			bounded.store.Close()
			t.Fatal(admitErr)
		}
		defer bounded.store.Close()
		if _, err := modelrecipe.CompileAdmittedCandidate(
			t.Context(), bounded.store, bounded.candidate.ID(), bounded.admission.ID,
			SameBaseCandidatePlugin{},
		); err == nil || !strings.Contains(err.Error(), "resident policy") {
			t.Fatalf("undersized resident budget compiled: %v", err)
		}
	})

	t.Run("total artifact budget", func(t *testing.T) {
		bounded, admitErr := newSameBaseCandidateFixture(t, sameBaseCandidateOptions{
			maxArtifactBytes: tensorBytes,
		})
		if admitErr != nil {
			bounded.store.Close()
			t.Fatal(admitErr)
		}
		defer bounded.store.Close()
		if _, err := modelrecipe.CompileAdmittedCandidate(
			t.Context(), bounded.store, bounded.candidate.ID(), bounded.admission.ID,
			SameBaseCandidatePlugin{},
		); err == nil || !strings.Contains(err.Error(), "exceeding budget") {
			t.Fatalf("undersized artifact budget compiled: %v", err)
		}
	})
}

type candidatePluginSpy struct {
	delegate    modelrecipe.CandidateDomainPlugin
	calls       int
	mutateFacts bool
	retained    []byte
}

func (value *candidatePluginSpy) CandidateDomain() string { return value.delegate.CandidateDomain() }

func (value *candidatePluginSpy) AdmissionAdapter() runrecord.CandidateComponentAdmissionAdapter {
	return value.delegate.AdmissionAdapter()
}

func (value *candidatePluginSpy) EvaluatorPlugin() modelrecipe.CandidateEvaluatorPlugin {
	return value.delegate.EvaluatorPlugin()
}

func (value *candidatePluginSpy) CompileCandidateComponent(
	ctx context.Context,
	reader artifact.Reader,
	facts modelrecipe.CandidateCompileFacts,
	component modelrecipe.CandidateComponent,
) (modelrecipe.CandidateComponentCompilation, error) {
	value.calls++
	if value.mutateFacts && len(facts.References) > 0 {
		facts.References[0].Role = modelrecipe.CandidateReferenceMeasurement
	}
	result, err := value.delegate.CompileCandidateComponent(ctx, reader, facts, component)
	if err == nil && len(result.Contents) > 0 {
		value.retained = result.Contents[0].Data
	}
	return result, err
}

type sameBaseCandidateOptions struct {
	omitDerivation   bool
	extraDerivation  bool
	deltaWidth       uint64
	maxResidentBytes uint64
	maxArtifactBytes uint64
}

type sameBaseCandidateFixture struct {
	store     *overgodb.Store
	base      artifact.ID
	delta     artifact.ID
	spec      SameBaseTaskArithmeticSpec
	evaluator modelrecipe.CandidateEvaluationIntent
	candidate modelrecipe.Candidate
	admission runrecord.CandidateAdmission
}

func newSameBaseCandidateFixture(
	t *testing.T,
	options sameBaseCandidateOptions,
) (sameBaseCandidateFixture, error) {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	deltaWidth := cmp.Or(options.deltaWidth, offlineTensorWidth)
	baseID := publishOfflineModel(t, store, "candidate-base", offlineTensorWidth, false)
	deltaID := publishOfflineModel(t, store, "candidate-delta", deltaWidth, false)
	base, err := modelrecipe.ResolveModelDefinition(ctx, store, baseID)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	delta, err := modelrecipe.ResolveModelDefinition(ctx, store, deltaID)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if !options.omitDerivation {
		lineage := []artifact.Lineage{
			{Child: delta.Document.ID, Parent: base.Document.ID, Relation: artifact.RelationDerivedFrom},
			{Child: delta.Document.Model, Parent: base.Document.Model, Relation: artifact.RelationDerivedFrom},
		}
		if options.extraDerivation {
			extraID := publishOfflineModel(t, store, "candidate-extra-base", offlineTensorWidth, false)
			extra, resolveErr := modelrecipe.ResolveModelDefinition(ctx, store, extraID)
			if resolveErr != nil {
				store.Close()
				t.Fatal(resolveErr)
			}
			lineage = append(lineage,
				artifact.Lineage{Child: delta.Document.ID, Parent: extra.Document.ID, Relation: artifact.RelationDerivedFrom},
				artifact.Lineage{Child: delta.Document.Model, Parent: extra.Document.Model, Relation: artifact.RelationDerivedFrom},
			)
		}
		if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
			Key:     "fixture/candidate-compile/same-base-lineage",
			Lineage: lineage,
		}); err != nil {
			store.Close()
			t.Fatal(err)
		}
	}

	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	developmentSplit := id(artifact.KindDatasetShard, "candidate compile development split")
	promotionSplit := id(artifact.KindDatasetShard, "candidate compile promotion split")
	binding, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Proposer: runrecord.AuthorityDomain{
			Name: "candidate-compile-proposer", Identity: id(artifact.KindEvidence, "candidate compile proposer"),
		},
		Evaluator: runrecord.AuthorityDomain{
			Name: "candidate-compile-evaluator", Identity: id(artifact.KindEvidence, "candidate compile evaluator"),
		},
		Decider: runrecord.AuthorityDomain{
			Name: "candidate-compile-decider", Identity: id(artifact.KindEvidence, "candidate compile decider"),
		},
		SealedInputs: promotionSplit, CleanWorker: id(artifact.KindEvidence, "candidate compile clean worker"),
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	code, err := runrecord.NewCodeRevision(strings.Repeat("d", 40))
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "candidate-host", OS: "windows", Arch: "amd64", Device: "cpu",
		Backend: "go", Driver: "process", Runtime: "go-test",
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	prediction := recipe.SteeringPrediction{
		Metric: "held-out quality", Benefit: 0.2, Cost: 3, Unit: "ratio", Uncertainty: 0.1,
	}
	development, err := runrecord.NewBudget("queries", developmentSplit, prediction.Cost, binding.Decider.Identity)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	promotion, err := runrecord.NewBudget("queries", promotionSplit, prediction.Cost, binding.Decider.Identity)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	tensorBytes := offlineTensorWidth * offlineTensorWidth * offlineFloatBytes
	maxResident := cmp.Or(options.maxResidentBytes, tensorBytes*2)
	maxArtifact := cmp.Or(options.maxArtifactBytes, tensorBytes*2)
	compositionSpec, err := newSameBaseTaskArithmeticSpec(SameBaseTaskArithmeticSpec{
		BaseDefinition: base.Document.ID,
		Deltas:         []SameBaseTaskDelta{{Definition: delta.Document.ID, Numerator: 1, Denominator: 2}},
		OutputFormat:   modelartifact.TensorFormatSafetensors, Placement: recipe.PlacementHost,
		MaxResidentBytes: maxResident, MaxShardBytes: tensorBytes, MaxArtifactBytes: maxArtifact,
		Rationale:     "bounded by the exact compatible tensor inventories",
		ReopenTrigger: "recompile when a model definition or resource envelope changes",
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	evaluator, err := modelrecipe.NewCandidateEvaluationIntent(
		compositionSpec.ID, prediction.Metric, "queries",
		modelrecipe.CandidateStopOnBudgetOrNonPositiveIsolatedImprovement,
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	observationDecider := recipe.Decider{
		CodeCommit: strings.Repeat("e", 40), Derivation: binding.Evaluator.Identity,
	}
	type observation struct {
		decision recipe.Decision
		source   artifact.Content
	}
	observe := func(subject artifact.ID, label string) observation {
		body := struct {
			Label string `json:"label"`
		}{Label: label}
		id, identifyErr := artifact.JSONID(artifact.KindEvidence, body)
		if identifyErr != nil {
			t.Fatal(identifyErr)
		}
		source, contentErr := (artifact.DocumentContract{
			Kind: artifact.KindEvidence, MediaType: "application/vnd.overgo.test-compile-measurement+json",
			Schema: "overgo/test-compile-measurement/v1",
		}).ContentJSON(id, body)
		if contentErr != nil {
			t.Fatal(contentErr)
		}
		decision, decisionErr := recipe.NewDecision(
			subject, recipe.DecisionObserved, recipe.EvidenceVerified, "",
			observationDecider, []artifact.ID{source.Descriptor.ID},
		)
		if decisionErr != nil {
			t.Fatal(decisionErr)
		}
		return observation{decision: decision, source: source}
	}
	decisions := []observation{
		observe(base.Document.ID, "baseline"),
		observe(compositionSpec.ID, "gap"),
		observe(compositionSpec.ID, "provenance"),
	}
	candidate, err := modelrecipe.NewCandidate(modelrecipe.CandidateSpec{
		Subject: compositionSpec.ID, Parent: base.Document.ID,
		Components: []modelrecipe.CandidateComponent{{
			Domain: modelrecipe.CandidateComposition, Specification: compositionSpec.ID,
		}},
		Prediction: prediction, CostUnit: "queries", Falsifier: evaluator.ID,
		References: []modelrecipe.CandidateReference{
			{Role: modelrecipe.CandidateReferenceBaseline, Subject: base.Document.ID, Evidence: decisions[0].decision.ID},
			{Role: modelrecipe.CandidateReferenceGap, Subject: compositionSpec.ID, Evidence: decisions[1].decision.ID},
			{Role: modelrecipe.CandidateReferenceProvenance, Subject: compositionSpec.ID, Evidence: decisions[2].decision.ID},
		},
		DevelopmentSplit: developmentSplit, PromotionSplit: promotionSplit,
		DevelopmentBudget: development.ID, PromotionBudget: promotion.ID,
		Code: code.ID, Environment: environment.ID,
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	contents := []artifact.Content{
		mustCandidateCompileContent(t, binding), mustCandidateCompileContent(t, development),
		mustCandidateCompileContent(t, promotion), mustCandidateCompileContent(t, environment),
		mustCandidateCompileContent(t, compositionSpec), mustCandidateCompileContent(t, evaluator),
		mustCandidateCompileContent(t, candidate),
	}
	codeBatch, err := code.Publication()
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	contents = append(contents, codeBatch.Contents...)
	lineage := append(compositionSpec.Lineage(), evaluator.Lineage()...)
	lineage = append(lineage, candidate.Lineage()...)
	lineage = append(lineage, development.Lineage()...)
	lineage = append(lineage, promotion.Lineage()...)
	for _, decision := range decisions {
		contents = append(contents, decision.source, mustCandidateCompileContent(t, decision.decision))
		lineage = append(lineage, decision.decision.Lineage()...)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key: "fixture/candidate-compile/authority",
		Artifacts: []artifact.Descriptor{
			{ID: developmentSplit}, {ID: promotionSplit},
			{ID: binding.Proposer.Identity}, {ID: binding.Evaluator.Identity},
			{ID: binding.Decider.Identity}, {ID: binding.CleanWorker},
		},
		Contents: contents, Lineage: lineage,
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	adapters := append(modelrecipe.CandidateAdmissionAdapters(), CandidateAdmissionAdapter())
	admission, admitErr := runrecord.AdmitCandidate(ctx, store, candidate, binding.ID, adapters...)
	fixture := sameBaseCandidateFixture{
		store: store, base: base.Document.ID, delta: delta.Document.ID,
		spec: compositionSpec, evaluator: evaluator, candidate: candidate, admission: admission,
	}
	if admitErr != nil {
		return fixture, admitErr
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:      "fixture/candidate-compile/admission",
		Contents: []artifact.Content{mustCandidateCompileContent(t, admission)}, Lineage: admission.Lineage(),
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	return fixture, nil
}

type candidateCompileContentDocument interface {
	Content() (artifact.Content, error)
}

func mustCandidateCompileContent(
	t *testing.T,
	value candidateCompileContentDocument,
) (content artifact.Content) {
	t.Helper()
	var err error
	if content, err = value.Content(); err != nil {
		t.Fatalf("candidate compile content: %v", err)
	}
	return content
}
