package controlleraction

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/artifact/repositorytest"
	"overgo/internal/composition"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
)

const materializationCodeCommit = "dddddddddddddddddddddddddddddddddddddddd"

func TestCrossDomainTrialMaterializesThroughOnePublicationOwner(t *testing.T) {
	fixture := newCandidateMaterializationFixture(t)
	t.Cleanup(func() { _ = fixture.store.Close() })
	useMaterializationCodeCommit(t, materializationCodeCommit)

	counting := &repositorytest.CountingRepository{Repository: fixture.store}
	outputRoot := filepath.Join(t.TempDir(), "materialized")
	result, err := MaterializeCandidateTrial(
		t.Context(), counting, fixture.decision.ID, outputRoot,
	)
	if err != nil {
		t.Fatal(err)
	}
	if counting.Commits != 1 || !result.Commit.Valid() {
		t.Fatalf("materialization commits = %d, commit = %s", counting.Commits, result.Commit)
	}
	if result.Materialization.Decision != fixture.decision.ID ||
		result.Materialization.Candidate != fixture.candidate.ID() ||
		result.Materialization.Admission != fixture.admission.ID ||
		result.Materialization.Trial != fixture.compiled.Trial.ID ||
		len(result.Materialization.Arms) != 2 || len(result.Directories) != 2 {
		t.Fatalf("materialization closure = %+v, directories = %+v", result.Materialization, result.Directories)
	}
	assertCandidateMaterializationBuildClosure(t, fixture.store, result, fixture.delta.Document.ID)
	addCandidateMaterializationProducer(t, fixture.store, result)
	if _, err := RequireCandidateMaterialization(
		t.Context(), fixture.store, result.Materialization.ID,
	); err != nil {
		t.Fatalf("materialization with another typed producer: %v", err)
	}
	retry := &repositorytest.CountingRepository{Repository: fixture.store}
	recovered, err := MaterializeCandidateTrial(
		t.Context(), retry, fixture.decision.ID, outputRoot,
	)
	if err != nil {
		t.Fatal(err)
	}
	if retry.Commits != 0 || !recovered.Recovered || recovered.Commit.Valid() ||
		recovered.Materialization.ID != result.Materialization.ID ||
		len(recovered.Directories) != len(result.Directories) {
		t.Fatalf("materialization recovery = %+v, commits = %d", recovered, retry.Commits)
	}

	if err := fixture.store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.Open(fixture.storeRoot)
	if err != nil {
		t.Fatal(err)
	}
	fixture.store = reopened
	loaded, err := RequireCandidateMaterialization(
		t.Context(), reopened, result.Materialization.ID,
	)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.ID != result.Materialization.ID || loaded.Decision != fixture.decision.ID ||
		loaded.Trial != fixture.compiled.Trial.ID || len(loaded.Arms) != 2 {
		t.Fatalf("cold materialization = %+v", loaded)
	}
	assertCandidateMaterializationBuildClosure(t, reopened, result, fixture.delta.Document.ID)
	aliases := 0
	if err := reopened.VisitAliases(t.Context(), "", func(overgodb.AliasView) error {
		aliases++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if aliases != 0 {
		t.Fatalf("materialization published %d aliases", aliases)
	}
}

func addCandidateMaterializationProducer(
	t *testing.T,
	repository artifact.Repository,
	result CandidateMaterializationResult,
) {
	t.Helper()
	arm := result.Materialization.Arms[0]
	run, err := runrecord.RequireExactRun(t.Context(), repository, arm.Run)
	if err != nil {
		t.Fatal(err)
	}
	duration := run.MeasuredNS + 1
	producer, err := runrecord.NewBoundRun(
		run.Recipe, runrecord.OutcomeSucceeded, run.Inputs, run.Outputs, "",
		run.CodeCommit, run.Environment, duration,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseBuild, DurationNS: duration}},
	)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := producer.Batch("fixture/candidate-materialization/additional-producer")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), repository, batch); err != nil {
		t.Fatal(err)
	}
}

func TestPrototypeMaterializationClosure(t *testing.T) {
	t.Run("poisoned resumable output", func(t *testing.T) {
		fixture := newCandidateMaterializationFixture(t)
		t.Cleanup(func() { _ = fixture.store.Close() })
		useMaterializationCodeCommit(t, materializationCodeCommit)
		root := filepath.Join(t.TempDir(), "poisoned-resume")
		if err := os.MkdirAll(root, 0o755); err != nil {
			t.Fatal(err)
		}
		realization := fixture.compiled.Components[0].Realization
		destination := filepath.Join(root, realization.DigestHex())
		writer, err := safetensors.NewStreamingWriter(destination, safetensors.StreamingPlan{
			Shards: []safetensors.StreamingShardSpec{{
				Name: "model.safetensors",
				Tensors: []safetensors.StreamingTensorSpec{{
					Name: "weight", DType: "F32", Shape: []uint64{2},
				}},
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		poison := make([]byte, 8)
		binary.LittleEndian.PutUint32(poison[0:], math.Float32bits(99))
		binary.LittleEndian.PutUint32(poison[4:], math.Float32bits(101))
		if err := writer.WriteTensor("weight", uint64(len(poison)), bytes.NewReader(poison)); err != nil {
			t.Fatal(err)
		}
		result, err := MaterializeCandidateTrial(
			t.Context(), fixture.store, fixture.decision.ID, root,
		)
		if err != nil {
			t.Fatal(err)
		}
		if got := readCandidateMaterializationTensor(
			t, result.Directories[realization], "weight",
		); !slices.Equal(got, []float32{6, 12}) {
			t.Fatalf("materialization adopted poisoned resume bytes: %v", got)
		}
	})

	t.Run("cold replay refuses lineage drift", func(t *testing.T) {
		fixture := newCandidateMaterializationFixture(t)
		t.Cleanup(func() { _ = fixture.store.Close() })
		useMaterializationCodeCommit(t, materializationCodeCommit)
		result, err := MaterializeCandidateTrial(
			t.Context(), fixture.store, fixture.decision.ID, filepath.Join(t.TempDir(), "lineage"),
		)
		if err != nil {
			t.Fatal(err)
		}
		var primary modelrecipe.CandidateMaterializedArm
		for _, arm := range result.Materialization.Arms {
			if !arm.Ablation.Valid() {
				primary = arm
			}
		}
		plan, err := composition.LoadOfflineTensorExecutionPlan(
			t.Context(), fixture.store, primary.Realization,
		)
		if err != nil {
			t.Fatal(err)
		}
		foreign, _, _ := publishCandidateMaterializationSource(
			t, fixture.store, "foreign-lineage", []float32{7, 9},
		)
		cases := []struct {
			name    string
			target  artifact.ID
			rewrite func([]artifact.Lineage) []artifact.Lineage
		}{
			{
				name: "missing tensor inventory owner", target: primary.TensorInventory,
				rewrite: func(edges []artifact.Lineage) []artifact.Lineage {
					return slices.DeleteFunc(edges, func(edge artifact.Lineage) bool {
						return edge.Parent == primary.Model && edge.Relation == artifact.RelationDerivedFrom
					})
				},
			},
			{
				name: "foreign model definition authority", target: primary.ModelDefinition,
				rewrite: func(edges []artifact.Lineage) []artifact.Lineage {
					return append(edges, artifact.Lineage{
						Child: primary.ModelDefinition, Parent: fixture.candidate.ID(),
						Relation: artifact.RelationDependsOn,
					})
				},
			},
			{
				name: "missing model source", target: primary.Model,
				rewrite: func(edges []artifact.Lineage) []artifact.Lineage {
					return slices.DeleteFunc(edges, func(edge artifact.Lineage) bool {
						return edge.Parent == plan.Inputs[0].Model &&
							edge.Relation == artifact.RelationDerivedFrom
					})
				},
			},
			{
				name: "foreign model derivation", target: primary.Model,
				rewrite: func(edges []artifact.Lineage) []artifact.Lineage {
					return append(edges, artifact.Lineage{
						Child: primary.Model, Parent: foreign.Document.Model,
						Relation: artifact.RelationDerivedFrom,
					})
				},
			},
		}
		for _, test := range cases {
			t.Run(test.name, func(t *testing.T) {
				reader := candidateMaterializationParentReader{
					Reader: fixture.store, target: test.target, rewrite: test.rewrite,
				}
				if _, err := RequireCandidateMaterialization(
					t.Context(), reader, result.Materialization.ID,
				); err == nil {
					t.Fatal("lineage drift survived materialization replay")
				}
			})
		}
	})

	t.Run("source byte drift", func(t *testing.T) {
		fixture := newCandidateMaterializationFixture(t)
		t.Cleanup(func() { _ = fixture.store.Close() })
		useMaterializationCodeCommit(t, materializationCodeCommit)
		path := filepath.Join(fixture.baseDirectory, "model.safetensors")
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
		if err := safetensors.Save(
			path, map[string][]float32{"weight": {3, 4}}, map[string][]int{"weight": {2}}, nil,
		); err != nil {
			t.Fatal(err)
		}
		counting := &repositorytest.CountingRepository{Repository: fixture.store}
		if _, err := MaterializeCandidateTrial(
			t.Context(), counting, fixture.decision.ID, filepath.Join(t.TempDir(), "drifted"),
		); err == nil {
			t.Fatal("source-byte drift materialized")
		}
		if counting.Commits != 0 {
			t.Fatalf("source-byte drift made %d commits", counting.Commits)
		}
	})

	t.Run("stale resource budget", func(t *testing.T) {
		fixture := newCandidateMaterializationFixture(t)
		t.Cleanup(func() { _ = fixture.store.Close() })
		useMaterializationCodeCommit(t, materializationCodeCommit)
		charge, err := runrecord.NewBudgetCharge(
			fixture.resources.ID, 1, fixture.decision.ID, "concurrent resource use",
		)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := charge.Batch("fixture/candidate-materialization/stale-budget")
		if err != nil {
			t.Fatal(err)
		}
		if _, err := artifact.CommitBatch(t.Context(), fixture.store, batch); err != nil {
			t.Fatal(err)
		}
		counting := &repositorytest.CountingRepository{Repository: fixture.store}
		if _, err := MaterializeCandidateTrial(
			t.Context(), counting, fixture.decision.ID, filepath.Join(t.TempDir(), "stale-budget"),
		); err == nil {
			t.Fatal("stale driver budget materialized")
		}
		if counting.Commits != 0 {
			t.Fatalf("stale driver budget made %d commits", counting.Commits)
		}
	})

	t.Run("non-realize decision", func(t *testing.T) {
		fixture := newCandidateMaterializationFixture(t)
		t.Cleanup(func() { _ = fixture.store.Close() })
		useMaterializationCodeCommit(t, materializationCodeCommit)
		stop := candidateTypedDocument(t, artifact.KindEvidence, "materialization operator stop", "materialization-stop")
		head, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
			Key: "fixture/candidate-materialization/operator-stop", Contents: []artifact.Content{stop},
		})
		if err != nil {
			t.Fatal(err)
		}
		facts := fixture.decisionFacts
		facts.Head, facts.OperatorStop = head, stop.Descriptor.ID
		stopped := publishCandidateMaterializationDecision(t, fixture.store, facts)
		if stopped.Action != runrecord.DriverActionStop {
			t.Fatalf("stopped driver action = %q", stopped.Action)
		}
		counting := &repositorytest.CountingRepository{Repository: fixture.store}
		if _, err := MaterializeCandidateTrial(
			t.Context(), counting, stopped.ID, filepath.Join(t.TempDir(), "stopped"),
		); err == nil {
			t.Fatal("non-realize decision materialized")
		}
		if counting.Commits != 0 {
			t.Fatalf("non-realize decision made %d commits", counting.Commits)
		}
	})
}

type candidateMaterializationParentReader struct {
	artifact.Reader
	target  artifact.ID
	rewrite func([]artifact.Lineage) []artifact.Lineage
}

func (reader candidateMaterializationParentReader) Parents(
	ctx context.Context,
	id artifact.ID,
) ([]artifact.Lineage, error) {
	edges, err := reader.Reader.Parents(ctx, id)
	if err != nil || id != reader.target {
		return edges, err
	}
	return reader.rewrite(slices.Clone(edges)), nil
}

type candidateMaterializationFixture struct {
	store         *overgodb.Store
	storeRoot     string
	base          modelrecipe.ResolvedModelDefinition
	delta         modelrecipe.ResolvedModelDefinition
	baseDirectory string
	candidate     modelrecipe.Candidate
	admission     runrecord.CandidateAdmission
	compiled      modelrecipe.CandidateCompilation
	resources     runrecord.Budget
	decisionFacts runrecord.DriverDecisionFacts
	decision      runrecord.DriverDecision
}

func newCandidateMaterializationFixture(t *testing.T) candidateMaterializationFixture {
	t.Helper()
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	base, _, baseDirectory := publishCandidateMaterializationSource(t, store, "base", []float32{2, 4})
	delta, _, _ := publishCandidateMaterializationSource(t, store, "delta", []float32{10, 20})
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "fixture/candidate-materialization/same-base",
		Lineage: []artifact.Lineage{
			{Child: delta.Document.ID, Parent: base.Document.ID, Relation: artifact.RelationDerivedFrom},
			{Child: delta.Document.Model, Parent: base.Document.Model, Relation: artifact.RelationDerivedFrom},
		},
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}

	id := func(kind artifact.Kind, label string) artifact.ID {
		return testutil.ArtifactID(t, kind, "candidate materialization "+label)
	}
	developmentSplit := id(artifact.KindDatasetShard, "development split")
	promotionSplit := id(artifact.KindDatasetShard, "promotion split")
	binding, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Proposer:     runrecord.AuthorityDomain{Name: "materialization-proposer", Identity: id(artifact.KindEvidence, "proposer")},
		Evaluator:    runrecord.AuthorityDomain{Name: "materialization-evaluator", Identity: id(artifact.KindEvidence, "evaluator")},
		Decider:      runrecord.AuthorityDomain{Name: "materialization-decider", Identity: id(artifact.KindEvidence, "decider")},
		SealedInputs: promotionSplit,
		CleanWorker:  id(artifact.KindEvidence, "clean worker"),
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	code, err := runrecord.NewCodeRevision(materializationCodeCommit)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	environment, err := runrecord.CurrentEnvironment(materializationDevice, materializationBackend)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	prediction := recipe.SteeringPrediction{
		Metric: "held-out quality", Benefit: 1, Cost: 1, Unit: "ratio",
	}
	development := candidateBudget(t, developmentSplit, "queries", prediction.Cost, binding.Decider.Identity)
	promotion := candidateBudget(t, promotionSplit, "queries", prediction.Cost, binding.Decider.Identity)
	const tensorBytes = uint64(2 * 4)
	compositionSpec := composition.SameBaseTaskArithmeticSpec{
		Version: artifact.InitialDocumentVersion, BaseDefinition: base.Document.ID,
		Deltas: []composition.SameBaseTaskDelta{{
			Definition: delta.Document.ID, Numerator: 1, Denominator: 2,
		}},
		OutputFormat: modelartifact.TensorFormatSafetensors, Placement: recipe.PlacementHost,
		MaxResidentBytes: tensorBytes * 2, MaxShardBytes: tensorBytes, MaxArtifactBytes: tensorBytes * 2,
		Rationale:     "bounded by the exact source tensor inventories",
		ReopenTrigger: "recompile when an exact source definition changes",
	}
	compositionSpec.ID, err = artifact.JSONID(artifact.KindRecipe, compositionSpec)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if _, err := compositionSpec.Content(); err != nil {
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
	observations := []candidateDecisionFixture{
		candidateObservedDecision(t, binding, base.Document.ID, "materialization baseline", materializationCodeCommit),
		candidateObservedDecision(t, binding, compositionSpec.ID, "materialization gap", materializationCodeCommit),
		candidateObservedDecision(t, binding, compositionSpec.ID, "materialization provenance", materializationCodeCommit),
	}
	candidate, err := modelrecipe.NewCandidate(modelrecipe.CandidateSpec{
		Subject: compositionSpec.ID, Parent: base.Document.ID,
		Components: []modelrecipe.CandidateComponent{{
			Domain: modelrecipe.CandidateComposition, Specification: compositionSpec.ID,
		}},
		Prediction: prediction, CostUnit: "queries", Falsifier: evaluator.ID,
		References: []modelrecipe.CandidateReference{
			{Role: modelrecipe.CandidateReferenceBaseline, Subject: base.Document.ID, Evidence: observations[0].decision.ID},
			{Role: modelrecipe.CandidateReferenceGap, Subject: compositionSpec.ID, Evidence: observations[1].decision.ID},
			{Role: modelrecipe.CandidateReferenceProvenance, Subject: compositionSpec.ID, Evidence: observations[2].decision.ID},
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
		mustCandidateDocumentContent(t, binding), mustCandidateDocumentContent(t, development),
		mustCandidateDocumentContent(t, promotion), mustCandidateDocumentContent(t, environment),
		mustCandidateDocumentContent(t, compositionSpec), mustCandidateDocumentContent(t, evaluator),
		mustCandidateDocumentContent(t, candidate), mustCodeRevisionContent(t, code),
	}
	lineage := append(compositionSpec.Lineage(), evaluator.Lineage()...)
	lineage = append(lineage, candidate.Lineage()...)
	lineage = append(lineage, development.Lineage()...)
	lineage = append(lineage, promotion.Lineage()...)
	for _, observation := range observations {
		contents = append(contents, observation.source, mustCandidateDocumentContent(t, observation.decision))
		lineage = append(lineage, observation.decision.Lineage()...)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "fixture/candidate-materialization/candidate",
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
	admission, err := runrecord.AdmitCandidate(
		t.Context(), store, candidate, binding.ID,
		append(modelrecipe.CandidateAdmissionAdapters(), composition.CandidateAdmissionAdapter())...,
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	admissionContent, err := admission.Content()
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "fixture/candidate-materialization/admission", Contents: []artifact.Content{admissionContent},
		Lineage: admission.Lineage(),
	}); err != nil {
		store.Close()
		t.Fatal(err)
	}
	compilationBatch, err := CompileTransaction(t.Context(), store, Action{
		Version: ActionVersion, Kind: KindCandidateCompilation,
		Compilation: &CandidateCompilationAction{Candidate: candidate.ID(), Admission: admission.ID},
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, compilationBatch); err != nil {
		store.Close()
		t.Fatal(err)
	}
	compiled, err := modelrecipe.RequireStoredCandidateCompilation(
		t.Context(), store, candidate.ID(), admission.ID, composition.SameBaseCandidatePlugin{},
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}

	interactionSplit := id(artifact.KindDatasetShard, "interaction split")
	resourceSplit := id(artifact.KindDatasetShard, "resource split")
	interactionAuthority := id(artifact.KindEvidence, "interaction authority")
	resourceAuthority := id(artifact.KindEvidence, "resource authority")
	interaction := candidateBudget(t, interactionSplit, "queries", 2, interactionAuthority)
	resources := candidateBudget(t, resourceSplit, "queries", 2, resourceAuthority)
	goal := candidateTypedDocument(t, artifact.KindEvidence, "materialization goal", "materialization-goal")
	rootEvidence := candidateTypedDocument(t, artifact.KindEvidence, "materialization causal root", "materialization-root")
	incumbent := candidateTypedDocument(t, artifact.KindRecipe, "materialization incumbent", "materialization-incumbent")
	lifecycle := candidateTypedDocument(t, artifact.KindEvidence, "materialization incumbent lifecycle", "materialization-lifecycle")
	placement := candidateTypedDocument(t, artifact.KindProfile, "materialization incumbent placement", "materialization-placement")
	incumbentEvidence := candidateTypedDocument(t, artifact.KindEvaluation, "materialization incumbent evidence", "materialization-evidence")
	causal, err := runrecord.NewCausalRoot(
		runrecord.TriggerControllerProposal, rootEvidence.Descriptor.ID, goal.Descriptor.ID,
	)
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	head, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "fixture/candidate-materialization/driver-authorities",
		Artifacts: []artifact.Descriptor{
			{ID: interactionSplit}, {ID: resourceSplit},
			{ID: interactionAuthority}, {ID: resourceAuthority},
		},
		Contents: []artifact.Content{
			goal, rootEvidence, incumbent, lifecycle, placement, incumbentEvidence,
			mustCandidateDocumentContent(t, interaction), mustCandidateDocumentContent(t, resources),
		},
		Lineage: append(interaction.Lineage(), resources.Lineage()...),
	})
	if err != nil {
		store.Close()
		t.Fatal(err)
	}
	facts := runrecord.DriverDecisionFacts{
		Goal: goal.Descriptor.ID, Causal: causal, Head: head,
		Options: []runrecord.DriverOption{
			{
				Recipe: incumbent.Descriptor.ID, Lifecycle: lifecycle.Descriptor.ID,
				State: runrecord.DriverIncumbentActive, Placement: placement.Descriptor.ID,
				Evidence: []artifact.ID{incumbentEvidence.Descriptor.ID},
			},
			{
				Candidate: candidate.ID(), Admission: admission.ID, State: runrecord.DriverCandidateEligible,
				Missing: []runrecord.DriverEvidenceGap{{
					Need: runrecord.DriverNeedRealization, Target: candidate.ID(), CostUnits: prediction.Cost,
					CostUnit: "queries", CostAuthority: candidate.ID(),
				}},
			},
		},
		InteractionBudget: runrecord.DriverBudgetReference{Grant: interaction.ID},
		ResourceBudget:    runrecord.DriverBudgetReference{Grant: resources.ID},
	}
	decision := publishCandidateMaterializationDecision(t, store, facts)
	if decision.Action != runrecord.DriverActionRealize || decision.Subject != candidate.ID() ||
		decision.Target != candidate.ID() || decision.Need != runrecord.DriverNeedRealization {
		store.Close()
		t.Fatalf("driver decision = %+v", decision)
	}
	return candidateMaterializationFixture{
		store: store, storeRoot: root, base: base, delta: delta, baseDirectory: baseDirectory,
		candidate: candidate, admission: admission, compiled: compiled,
		resources: resources, decisionFacts: facts, decision: decision,
	}
}

func publishCandidateMaterializationSource(
	t *testing.T,
	store *overgodb.Store,
	label string,
	values []float32,
) (modelrecipe.ResolvedModelDefinition, modelartifact.Inventory, string) {
	t.Helper()
	directory := t.TempDir()
	weightPath := filepath.Join(directory, "model.safetensors")
	if err := safetensors.Save(
		weightPath, map[string][]float32{"weight": values},
		map[string][]int{"weight": {len(values)}}, nil,
	); err != nil {
		t.Fatal(err)
	}
	inventory, err := modelartifact.FromSafetensorsPath(directory)
	if err != nil {
		t.Fatal(err)
	}
	profile, found := model.LookupArchitecture("llama")
	if !found {
		t.Fatal("llama profile is absent")
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		t.Fatal(err)
	}
	spec := model.Spec{
		CommonSpec: model.CommonSpec{
			Architecture: "llama", BlockCount: 1, ContextLength: 128,
			EmbeddingLength: 8, FeedForwardLength: 16, RMSNormEpsilon: 1e-5,
		},
		AttentionSpec: model.AttentionSpec{
			HeadCount: 2, HeadCountKV: 1, KeyLength: 4, ValueLength: 4, RopeFrequencyBase: 10_000,
		},
	}
	document, err := modelrecipe.NewModelDefinitionDocument(profileDocument, inventory.TensorInventory, spec)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := resolved.Batch("fixture/candidate-materialization/source/"+label, inventory)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	storedDirectory, err := artifact.AvailablePath(
		t.Context(), store, inventory.Manifest.ID, artifact.LocationDirectory,
	)
	if err != nil {
		t.Fatal(err)
	}
	storedWeight, err := artifact.AvailablePath(
		t.Context(), store, inventory.Components[0].ID, artifact.LocationFile,
	)
	if err != nil {
		t.Fatal(err)
	}
	absDirectory, _ := filepath.Abs(directory)
	absWeight, _ := filepath.Abs(weightPath)
	if filepath.Clean(storedDirectory) != filepath.Clean(absDirectory) ||
		filepath.Clean(storedWeight) != filepath.Clean(absWeight) {
		t.Fatalf("source locations = (%q, %q), want (%q, %q)", storedDirectory, storedWeight, absDirectory, absWeight)
	}
	return resolved, inventory, directory
}

func publishCandidateMaterializationDecision(
	t *testing.T,
	store *overgodb.Store,
	facts runrecord.DriverDecisionFacts,
) runrecord.DriverDecision {
	t.Helper()
	decision, err := runrecord.NewDriverDecision(t.Context(), store, facts)
	if err != nil {
		t.Fatal(err)
	}
	content, err := decision.Content()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := artifact.NewDocumentBatch(
		"fixture/candidate-materialization/decision/"+decision.ID.DigestHex(),
		[]artifact.Content{content}, decision.Lineage(), nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := runrecord.BindCausality(&batch, decision.ID, &facts.Causal); err != nil {
		t.Fatal(err)
	}
	expected := facts.Head
	batch.ExpectedHead = &expected
	if _, err := artifact.CommitBatch(t.Context(), store, batch); err != nil {
		t.Fatal(err)
	}
	return decision
}

func assertCandidateMaterializationBuildClosure(
	t *testing.T,
	reader artifact.Reader,
	result CandidateMaterializationResult,
	deltaDefinition artifact.ID,
) {
	t.Helper()
	primary, ablation := 0, 0
	for _, arm := range result.Materialization.Arms {
		want := []float32{6, 12}
		if arm.Ablation.Valid() {
			ablation++
			want = []float32{2, 4}
			if arm.Omitted != deltaDefinition {
				t.Fatalf("ablation omitted %s, want %s", arm.Omitted, deltaDefinition)
			}
		} else {
			primary++
			if arm.Omitted.Valid() {
				t.Fatalf("primary arm omitted %s", arm.Omitted)
			}
		}
		directory, found := result.Directories[arm.Realization]
		if !found {
			t.Fatalf("directory for realization %s is absent", arm.Realization)
		}
		locations, err := reader.Locations(t.Context(), arm.Model)
		if err != nil {
			t.Fatal(err)
		}
		locationFound := slices.ContainsFunc(locations, func(location artifact.Location) bool {
			return location.Kind == artifact.LocationDirectory &&
				filepath.Clean(location.Value) == filepath.Clean(directory)
		})
		if !locationFound {
			t.Fatalf("output locations = %+v, want directory %q", locations, directory)
		}
		if got := readCandidateMaterializationTensor(t, directory, "weight"); !slices.Equal(got, want) {
			t.Fatalf("materialized weight = %v, want %v", got, want)
		}
		definition, err := modelrecipe.ResolveModelDefinition(t.Context(), reader, arm.ModelDefinition)
		if err != nil {
			t.Fatal(err)
		}
		if definition.Document.Model != arm.Model || definition.Tensors.ID != arm.TensorInventory ||
			len(definition.Tensors.Tensors) != 1 || definition.Tensors.Tensors[0].Name != "weight" ||
			!slices.Equal(definition.Tensors.Tensors[0].Shape, []uint64{2}) ||
			definition.Tensors.Tensors[0].Bytes != 8 {
			t.Fatalf("materialized model definition = %+v", definition)
		}
		output, err := composition.RequireOfflineArtifactOutput(t.Context(), reader, arm.Output)
		if err != nil {
			t.Fatal(err)
		}
		if output.ExecutionPlan != arm.Realization || output.Model != arm.Model ||
			output.TensorInventory != arm.TensorInventory || output.ModelDefinition != arm.ModelDefinition ||
			output.TensorBytes != arm.TensorBytes || output.StoredBytes != arm.StoredBytes {
			t.Fatalf("materialized output = %+v, arm = %+v", output, arm)
		}
		observation, err := composition.RequireOfflineArtifactObservation(
			t.Context(), reader, arm.Observation,
		)
		if err != nil {
			t.Fatal(err)
		}
		if observation.ExecutionPlan != arm.Realization || observation.Output != arm.Output ||
			observation.Run != arm.Run || observation.Model != arm.Model ||
			observation.PeakResidentBytes != arm.PeakResidentBytes {
			t.Fatalf("materialized observation = %+v, arm = %+v", observation, arm)
		}
		plan, err := composition.LoadOfflineTensorExecutionPlan(t.Context(), reader, arm.Realization)
		if err != nil {
			t.Fatal(err)
		}
		run, err := runrecord.RequireExactRun(t.Context(), reader, arm.Run)
		if err != nil {
			t.Fatal(err)
		}
		if run.Recipe != plan.ArtifactPlan || run.Outcome != runrecord.OutcomeSucceeded ||
			run.CodeCommit != materializationCodeCommit || run.Environment != result.Materialization.Environment ||
			!slices.Contains(run.Inputs, arm.Realization) || !slices.Contains(run.Outputs, arm.Output) ||
			(!arm.Ablation.Valid() && !slices.Contains(run.Outputs, arm.Model)) ||
			(arm.Ablation.Valid() && slices.Contains(run.Outputs, arm.Model)) {
			t.Fatalf("materialized run = %+v", run)
		}
	}
	if primary != 1 || ablation != 1 {
		t.Fatalf("materialized arms = primary %d, ablation %d", primary, ablation)
	}
	chargeContent, found, err := artifact.ReadContent(
		t.Context(), reader, result.Materialization.ResourceCharge,
	)
	if err != nil || !found {
		t.Fatalf("materialization charge content = (%t, %v)", found, err)
	}
	charge, err := runrecord.ParseBudgetCharge(chargeContent.Data)
	if err != nil || charge.Consumer != result.Materialization.Decision || charge.Amount != 1 {
		t.Fatalf("materialization charge = (%+v, %v)", charge, err)
	}
}

func readCandidateMaterializationTensor(t *testing.T, directory, name string) []float32 {
	t.Helper()
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	view, found := source.Tensors[name]
	if !found {
		t.Fatalf("tensor %q is absent", name)
	}
	data := make([]byte, view.Size())
	if _, err := io.ReadFull(view.Reader(), data); err != nil {
		t.Fatal(err)
	}
	values := make([]float32, len(data)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[index*4:]))
	}
	return values
}

func useMaterializationCodeCommit(t *testing.T, commit string) {
	t.Helper()
	prior := executableCodeCommit
	executableCodeCommit = constantMaterializationCodeCommit(commit)
	t.Cleanup(func() { executableCodeCommit = prior })
}

func constantMaterializationCodeCommit(commit string) func(string) (string, error) {
	return func(string) (string, error) { return commit, nil }
}
