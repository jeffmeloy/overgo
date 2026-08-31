package modelrecipe

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/representation"
	"overgo/internal/runrecord"
	"overgo/internal/tensor/dtype"
	"overgo/internal/testutil"
)

type steeringCandidateFixture struct {
	store     *overgodb.Store
	direction representation.ResidualDirection
	candidate Candidate
	admission runrecord.CandidateAdmission
	binding   runrecord.AdmissionBinding
}

type steeringCandidateOptions struct {
	omitMeasurement bool
	omitBudget      bool
}

func newSteeringCandidateFixture(t *testing.T, options steeringCandidateOptions) (steeringCandidateFixture, error) {
	t.Helper()
	ctx := t.Context()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	fail := func(err error) {
		t.Helper()
		t.Fatal(err)
	}
	blob := func(kind artifact.Kind, text string) (artifact.ID, artifact.Content) {
		data := []byte(text)
		id, blobErr := artifact.IdentifyBytes(kind, data)
		if blobErr != nil {
			fail(blobErr)
		}
		return id, artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data))}, Data: data}
	}
	model := testutil.ArtifactID(t, artifact.KindModel, "steering fixture model")
	definition, definitionContent := blob(artifact.KindModelDefinition, "steering fixture definition")
	vector, vectorContent := blob(artifact.KindTensorSet, "steering fixture direction vector bytes")
	layer := uint32(5)
	contract, err := representation.NewContract(representation.Contract{
		Producer: representation.Producer{Model: model, Definition: definition, Tap: representation.TapLayerInput, Layer: &layer},
		Modality: representation.ModalityText,
		Tensor: representation.TensorContract{DataType: dtype.F32, Axes: []representation.Axis{
			{Kind: representation.AxisChannel, Bounds: representation.AxisBounds{Extent: 8}},
			{Kind: representation.AxisSequence, Bounds: representation.AxisBounds{Minimum: 1, Maximum: 16}},
		}},
		Sequence: representation.SequenceContract{
			Axis: representation.AxisSequence, Mask: representation.MaskPrefix,
			Padding: representation.PaddingSuffix, Position: representation.PositionSequential,
			PositionAxes: []representation.AxisKind{representation.AxisSequence},
		},
		Normalization: representation.NormalizationContract{
			Kind: representation.NormalizationNone, Magnitude: representation.MagnitudeNative,
		},
	})
	if err != nil {
		fail(err)
	}
	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	direction, err := representation.NewResidualDirection(representation.ResidualDirection{
		Contract: contract.ID, Model: model, Definition: definition,
		Manuals:     id(artifact.KindProfile, "steering fixture manual catalog"),
		Readout:     id(artifact.KindEvidence, "steering fixture tool-open readout"),
		Corpus:      id(artifact.KindDataset, "steering fixture corpus"),
		Split:       id(artifact.KindDatasetShard, "steering fixture extraction split"),
		Quantiles:   []float64{0.25, 0.5, 0.75},
		Extraction:  id(artifact.KindEvidence, "steering fixture extraction run"),
		Environment: id(artifact.KindEvidence, "steering fixture extraction environment"),
		Vector:      vector,
		Selector: representation.AlphaSelector{
			Objective:     "maximize realized inspection-call rate",
			Normalization: "unit norm at the contract layer",
			Target:        0.5,
			TieBreak:      "smallest alpha",
		},
	})
	if err != nil {
		fail(err)
	}
	developmentSplit := id(artifact.KindDatasetShard, "steering development split")
	promotionSplit := id(artifact.KindDatasetShard, "steering promotion split")
	binding, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Proposer: runrecord.AuthorityDomain{Name: "steering-proposer", Identity: id(artifact.KindEvidence, "steering proposer")},
		Evaluator: runrecord.AuthorityDomain{
			Name: "steering-evaluator", Identity: id(artifact.KindEvidence, "steering evaluator"),
		},
		Decider:      runrecord.AuthorityDomain{Name: "steering-decider", Identity: id(artifact.KindEvidence, "steering decider")},
		SealedInputs: promotionSplit, CleanWorker: id(artifact.KindEvidence, "steering clean worker"),
	})
	if err != nil {
		fail(err)
	}
	code, err := runrecord.NewCodeRevision(strings.Repeat("f", 40))
	if err != nil {
		fail(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "steering-host", OS: "windows", Arch: "amd64", Device: "cpu",
		Backend: "go", Driver: "process", Runtime: "go-test",
	})
	if err != nil {
		fail(err)
	}
	prediction := recipe.SteeringPrediction{
		Metric: "realized inspection-call rate", Benefit: 0.15, Cost: 4, Unit: "ratio", Uncertainty: 0.2,
	}
	development, err := runrecord.NewBudget("queries", developmentSplit, prediction.Cost, binding.Decider.Identity)
	if err != nil {
		fail(err)
	}
	promotion, err := runrecord.NewBudget("queries", promotionSplit, prediction.Cost, binding.Decider.Identity)
	if err != nil {
		fail(err)
	}
	falsifier, err := NewCandidateEvaluationIntent(
		direction.ID, prediction.Metric, "queries",
		CandidateStopOnBudgetOrNonPositiveIsolatedImprovement,
	)
	if err != nil {
		fail(err)
	}
	decider := recipe.Decider{CodeCommit: strings.Repeat("a", 40), Derivation: binding.Evaluator.Identity}
	type observation struct {
		decision recipe.Decision
		source   artifact.Content
	}
	observe := func(subject artifact.ID, label string) observation {
		body := struct {
			Label string `json:"label"`
		}{Label: label}
		observationID, identifyErr := artifact.JSONID(artifact.KindEvidence, body)
		if identifyErr != nil {
			fail(identifyErr)
		}
		source, contentErr := (artifact.DocumentContract{
			Kind: artifact.KindEvidence, MediaType: "application/vnd.overgo.test-steering-measurement+json",
			Schema: "overgo/test-steering-measurement/v1",
		}).ContentJSON(observationID, body)
		if contentErr != nil {
			fail(contentErr)
		}
		decision, decisionErr := recipe.NewDecision(
			subject, recipe.DecisionObserved, recipe.EvidenceVerified, "",
			decider, []artifact.ID{source.Descriptor.ID},
		)
		if decisionErr != nil {
			fail(decisionErr)
		}
		return observation{decision: decision, source: source}
	}
	observations := []observation{
		observe(definition, "baseline"),
		observe(direction.ID, "gap"),
		observe(direction.ID, "provenance"),
	}
	references := []CandidateReference{
		{Role: CandidateReferenceBaseline, Subject: definition, Evidence: observations[0].decision.ID},
		{Role: CandidateReferenceGap, Subject: direction.ID, Evidence: observations[1].decision.ID},
		{Role: CandidateReferenceProvenance, Subject: direction.ID, Evidence: observations[2].decision.ID},
	}
	var traceContent artifact.Content
	if !options.omitMeasurement {
		trace, traceErr := runrecord.NewEfficiencyTrace(runrecord.EfficiencyTrace{
			Surface: runrecord.SurfaceTool, Task: "steering extraction harness",
			Work:     runrecord.InteractionWork{ToolCalls: 12, ModelTurns: 3, WallNS: 1_000_000},
			Result:   direction.Extraction,
			Evidence: direction.Readout,
		})
		if traceErr != nil {
			fail(traceErr)
		}
		traceContent, traceErr = trace.Content()
		if traceErr != nil {
			fail(traceErr)
		}
		decision, decisionErr := recipe.NewDecision(
			direction.ID, recipe.DecisionObserved, recipe.EvidenceVerified, "",
			decider, []artifact.ID{trace.ID},
		)
		if decisionErr != nil {
			fail(decisionErr)
		}
		observations = append(observations, observation{decision: decision, source: traceContent})
		references = append(references, CandidateReference{
			Role: CandidateReferenceMeasurement, Subject: direction.ID, Evidence: decision.ID,
		})
	}
	developmentBudget := development.ID
	if options.omitBudget {
		// Present as a bare artifact so lineage resolves, absent as a
		// budget document, so admission itself must refuse it.
		developmentBudget = id(artifact.KindEvidence, "steering absent budget")
	}
	candidate, err := NewCandidate(CandidateSpec{
		Subject: direction.ID, Parent: definition,
		Components: []CandidateComponent{{Domain: CandidateSteering, Specification: direction.ID}},
		Prediction: prediction, CostUnit: "queries", Falsifier: falsifier.ID,
		References:       references,
		DevelopmentSplit: developmentSplit, PromotionSplit: promotionSplit,
		DevelopmentBudget: developmentBudget, PromotionBudget: promotion.ID,
		Code: code.ID, Environment: environment.ID,
	})
	if err != nil {
		fail(err)
	}
	mustContent := func(value interface {
		Content() (artifact.Content, error)
	}) artifact.Content {
		content, contentErr := value.Content()
		if contentErr != nil {
			fail(contentErr)
		}
		return content
	}
	contents := []artifact.Content{
		definitionContent, vectorContent,
		mustContent(contract), mustContent(direction),
		mustContent(binding), mustContent(development), mustContent(promotion),
		mustContent(environment), mustContent(falsifier), mustContent(candidate),
	}
	codeBatch, err := code.Publication()
	if err != nil {
		fail(err)
	}
	contents = append(contents, codeBatch.Contents...)
	descriptors := []artifact.Descriptor{
		{ID: model}, {ID: developmentSplit}, {ID: promotionSplit},
		{ID: direction.Manuals}, {ID: direction.Readout}, {ID: direction.Corpus},
		{ID: direction.Split}, {ID: direction.Extraction}, {ID: direction.Environment},
		{ID: binding.Proposer.Identity}, {ID: binding.Evaluator.Identity},
		{ID: binding.Decider.Identity}, {ID: binding.CleanWorker},
	}
	if options.omitBudget {
		descriptors = append(descriptors, artifact.Descriptor{ID: developmentBudget})
	}
	lineage := append(direction.Lineage(), falsifier.Lineage()...)
	lineage = append(lineage, candidate.Lineage()...)
	lineage = append(lineage, development.Lineage()...)
	lineage = append(lineage, promotion.Lineage()...)
	for _, observed := range observations {
		contents = append(contents, observed.source, mustContent(observed.decision))
		lineage = append(lineage, observed.decision.Lineage()...)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:       "fixture/steering-candidate/authority",
		Artifacts: descriptors,
		Contents:  contents, Lineage: lineage,
	}); err != nil {
		fail(err)
	}
	admission, admitErr := runrecord.AdmitCandidate(ctx, store, candidate, binding.ID, CandidateAdmissionAdapters()...)
	fixture := steeringCandidateFixture{
		store: store, direction: direction, candidate: candidate, admission: admission, binding: binding,
	}
	if admitErr != nil {
		return fixture, admitErr
	}
	admissionContent, err := admission.Content()
	if err != nil {
		fail(err)
	}
	if _, err := artifact.CommitBatch(ctx, store, artifact.Batch{
		Key:      "fixture/steering-candidate/admission",
		Contents: []artifact.Content{admissionContent}, Lineage: admission.Lineage(),
	}); err != nil {
		fail(err)
	}
	return fixture, nil
}

// TestRecipeBoundToolCallPropensityControl pins the thin steering plugin over
// the common Candidate spine: an admitted steering candidate compiles into a
// recipe-bound plan whose realization carries the direction and its
// predeclared selector -- never an alpha value -- beside the alpha-zero
// identity ablation and the exact resident resource bound. A candidate whose
// direction lacks a direct measurement never admits.
func TestRecipeBoundToolCallPropensityControl(t *testing.T) {
	ctx := t.Context()
	fixture, err := newSteeringCandidateFixture(t, steeringCandidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := CompileAdmittedCandidate(
		ctx, fixture.store, fixture.candidate.ID(), fixture.admission.ID, SteeringCandidatePlugin{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(compiled.Components) != 1 || len(compiled.Ablations) != 1 {
		t.Fatalf("compiled shape = %d components, %d ablations", len(compiled.Components), len(compiled.Ablations))
	}
	plan := compiled.Components[0]
	if plan.Domain != CandidateSteering || plan.Specification != fixture.direction.ID ||
		!plan.Realization.Valid() || !plan.ResourcePolicy.Valid() || plan.PeakResidentBytes == 0 {
		t.Fatalf("component plan = %+v", plan)
	}
	var steered SteeringRealization
	found := false
	for _, content := range compiled.Contents {
		if content.Descriptor.ID == plan.Realization {
			if err := json.Unmarshal(content.Data, &steered); err != nil {
				t.Fatal(err)
			}
			if strings.Contains(string(content.Data), `"alpha"`) {
				t.Fatalf("realization exposes an alpha knob: %s", content.Data)
			}
			found = true
		}
	}
	if !found || !steered.InspectionOnly || steered.AlphaZero ||
		steered.Direction != fixture.direction.ID || steered.Selector != fixture.direction.Selector {
		t.Fatalf("steered realization = %+v found=%t", steered, found)
	}
	if compiled.Ablations[0].Omitted != fixture.direction.Vector {
		t.Fatalf("ablation omits %s, want the direction vector", compiled.Ablations[0].Omitted)
	}

	if _, err := newSteeringCandidateFixture(t, steeringCandidateOptions{omitMeasurement: true}); err == nil ||
		!strings.Contains(err.Error(), "measurement") {
		t.Fatalf("unmeasured direction admitted: %v", err)
	}
}

// TestToolBudgetProposalAdmission pins the migrated proposal path: a tool
// propensity proposal is eligible only through the one cross-domain
// admission entry, its decision replays byte-identically, and a proposal
// whose budget authority is absent refuses -- there is no direct steering
// eligibility owner left to bypass this.
func TestToolBudgetProposalAdmission(t *testing.T) {
	ctx := t.Context()
	fixture, err := newSteeringCandidateFixture(t, steeringCandidateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := runrecord.RequireReplayedCandidateAdmission(
		ctx, fixture.store, fixture.admission.ID, fixture.candidate, CandidateAdmissionAdapters()...,
	)
	if err != nil || replayed.ID != fixture.admission.ID {
		t.Fatalf("replayed admission = %+v, %v", replayed, err)
	}
	if _, err := newSteeringCandidateFixture(t, steeringCandidateOptions{omitBudget: true}); err == nil ||
		!strings.Contains(err.Error(), "budget") {
		t.Fatalf("budget-less proposal admitted: %v", err)
	}
}
