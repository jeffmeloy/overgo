package evaluation

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

type knowledgePromotionFixture struct {
	path      string
	store     *overgodb.Store
	proposal  EpisodeKnowledgeProposal
	admission runrecord.CandidateAdmission
	candidate modelrecipe.Candidate
	transform dataset.DatasetTransform
	hygiene   KnowledgeHygieneReport
	baseline  RetrievalEvaluation
	trial     RetrievalEvaluation
	split     artifact.ID
}

func TestEpisodeKnowledgePromotionRequiresHeldOutBenefit(t *testing.T) {
	fixture := newKnowledgePromotionFixture(t)
	evidence, err := PublishEpisodeKnowledgePromotion(t.Context(), fixture.store,
		fixture.proposal.ID, fixture.baseline.ID, fixture.trial.ID)
	if err != nil {
		fixture.store.Close()
		t.Fatal(err)
	}
	if evidence.Admission != fixture.admission.ID || evidence.Split != fixture.split ||
		evidence.Output != fixture.proposal.Output || !slices.Contains(evidence.Benefits, "recall") ||
		evidence.ValidateIdentity() != nil {
		fixture.store.Close()
		t.Fatalf("promotion evidence = %+v", evidence)
	}
	if fixture.hygiene.InputRecords != 2 || fixture.hygiene.OutputRecords != 1 ||
		fixture.hygiene.DuplicatesRemoved != 1 || fixture.hygiene.ConflictsResolved != 1 ||
		fixture.hygiene.PromotionOverlap != 0 {
		fixture.store.Close()
		t.Fatalf("derived hygiene = %+v", fixture.hygiene)
	}
	if err := fixture.store.Close(); err != nil {
		t.Fatal(err)
	}
	reopened, err := overgodb.Open(fixture.path)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	cold, err := RequireKnowledgePromotionEvidence(t.Context(), reopened, evidence.ID)
	if err != nil || cold.ID != evidence.ID {
		t.Fatalf("cold promotion = %+v, %v", cold, err)
	}

	tied := fixture.trial
	tied.ID = testutil.ArtifactID(t, artifact.KindEvaluation, "knowledge-tied-evaluation")
	tied.Receipt = testutil.ArtifactID(t, artifact.KindEvidence, "knowledge-tied-receipt")
	if _, err := retrievalParetoBenefits(fixture.trial, tied); err == nil || !strings.Contains(err.Error(), "no measured benefit") {
		t.Fatalf("held-out tie promoted: %v", err)
	}
}

func TestEpisodeKnowledgePromotionRejectsUncitedOrContaminatedCandidate(t *testing.T) {
	t.Run("contaminated stored hygiene", func(t *testing.T) {
		fixture := newKnowledgePromotionFixtureWithOverlap(t, true)
		defer fixture.store.Close()
		if fixture.hygiene.PromotionOverlap != 1 {
			t.Fatalf("promotion overlap = %d", fixture.hygiene.PromotionOverlap)
		}
		if _, err := EvaluateEpisodeKnowledgePromotion(t.Context(), fixture.store,
			fixture.proposal.ID, fixture.baseline.ID, fixture.trial.ID); err == nil ||
			!strings.Contains(err.Error(), "held-out overlap") {
			t.Fatalf("contaminated candidate promoted: %v", err)
		}
	})

	t.Run("caller cannot add a citation", func(t *testing.T) {
		fixture := newKnowledgePromotionFixture(t)
		defer fixture.store.Close()
		extra := knowledgeTypedDocument(t, artifact.KindEvidence, "forged knowledge citation")
		if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
			Key: "knowledge/forged-citation-source", Contents: []artifact.Content{extra},
		}); err != nil {
			t.Fatal(err)
		}
		forged := fixture.proposal
		forged.ID = artifact.ID{}
		forged.Citations = append(slices.Clone(forged.Citations), extra.Descriptor.ID)
		forged, err := episodeKnowledgeProposalCodec.New(forged)
		if err != nil {
			t.Fatal(err)
		}
		publishKnowledgeDocument(t, fixture.store, "knowledge/forged-proposal", forged)
		if _, err := EvaluateEpisodeKnowledgePromotion(t.Context(), fixture.store,
			forged.ID, fixture.baseline.ID, fixture.trial.ID); err == nil ||
			!strings.Contains(err.Error(), "differs from admitted evidence") {
			t.Fatalf("caller-authored citation promoted: %v", err)
		}
	})

	t.Run("candidate identity is not admission", func(t *testing.T) {
		fixture := newKnowledgePromotionFixture(t)
		defer fixture.store.Close()
		if _, err := PublishEpisodeKnowledgeProposal(t.Context(), fixture.store, fixture.candidate.ID()); err == nil {
			t.Fatal("unadmitted candidate identity was accepted as promotion authority")
		}
	})
}

func TestKnowledgePromotionPublicationRejectsStaleOrForgedEvidence(t *testing.T) {
	t.Run("stale head", func(t *testing.T) {
		fixture := newKnowledgePromotionFixture(t)
		defer fixture.store.Close()
		racing := &knowledgeRacingRepository{
			Repository: fixture.store,
			marker:     testutil.ArtifactID(t, artifact.KindEvidence, "knowledge-publication-race"),
		}
		if _, err := PublishEpisodeKnowledgePromotion(t.Context(), racing,
			fixture.proposal.ID, fixture.baseline.ID, fixture.trial.ID); !errors.Is(err, overgodb.ErrHeadConflict) {
			t.Fatalf("stale promotion publication = %v", err)
		}
	})

	t.Run("replay mismatch", func(t *testing.T) {
		fixture := newKnowledgePromotionFixture(t)
		defer fixture.store.Close()
		legitimate, err := EvaluateEpisodeKnowledgePromotion(t.Context(), fixture.store,
			fixture.proposal.ID, fixture.baseline.ID, fixture.trial.ID)
		if err != nil {
			t.Fatal(err)
		}
		forged := legitimate
		forged.ID = artifact.ID{}
		forged.Benefits = []string{"provider-cost"}
		forged, err = knowledgePromotionCodec.New(forged)
		if err != nil || forged.ID == legitimate.ID {
			t.Fatalf("forged promotion = %+v, %v", forged, err)
		}
		publishKnowledgeDocument(t, fixture.store, "knowledge/forged-promotion", forged)
		if _, err := RequireKnowledgePromotionEvidence(t.Context(), fixture.store, forged.ID); err == nil ||
			!strings.Contains(err.Error(), "differs from replayed") {
			t.Fatalf("forged aggregate survived replay: %v", err)
		}
	})

	t.Run("added lineage", func(t *testing.T) {
		fixture := newKnowledgePromotionFixture(t)
		defer fixture.store.Close()
		promotion, err := PublishEpisodeKnowledgePromotion(t.Context(), fixture.store,
			fixture.proposal.ID, fixture.baseline.ID, fixture.trial.ID)
		if err != nil {
			t.Fatal(err)
		}
		extra := testutil.ArtifactID(t, artifact.KindEvidence, "knowledge-extra-promotion-parent")
		if _, err := artifact.CommitBatch(t.Context(), fixture.store, artifact.Batch{
			Key:       "knowledge/promotion-extra-lineage",
			Artifacts: []artifact.Descriptor{{ID: extra}},
			Lineage: []artifact.Lineage{{
				Child: promotion.ID, Parent: extra, Relation: artifact.RelationDependsOn,
			}},
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := RequireKnowledgePromotionEvidence(t.Context(), fixture.store, promotion.ID); err == nil ||
			!strings.Contains(err.Error(), "stored lineage differs") {
			t.Fatalf("promotion with added authority survived: %v", err)
		}
	})
}

func newKnowledgePromotionFixture(t *testing.T) knowledgePromotionFixture {
	return newKnowledgePromotionFixtureWithOverlap(t, false)
}

func newKnowledgePromotionFixtureWithOverlap(t *testing.T, promotionOverlap bool) knowledgePromotionFixture {
	t.Helper()
	path := t.TempDir()
	store, err := overgodb.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	closeOnError := func(err error) {
		store.Close()
		t.Fatal(err)
	}

	baselineDataset := knowledgeTypedDocument(t, artifact.KindDataset, "knowledge baseline dataset")
	trialDataset := knowledgeTypedDocument(t, artifact.KindDataset, "knowledge trial dataset")
	baselineSearch := publishKnowledgeConsumedRetrieval(t, store, "knowledge-baseline", baselineDataset,
		[]string{"target distractor"}, map[string]float64{"target distractor": 1})
	trialSearch := publishKnowledgeConsumedRetrieval(t, store, "knowledge-trial", trialDataset,
		[]string{"target answer"}, map[string]float64{"target answer": 1})
	result := trialSearch.search.Results[0]
	split := testutil.ArtifactID(t, artifact.KindDatasetShard, "knowledge-promotion-split")
	developmentSplit := testutil.ArtifactID(t, artifact.KindDatasetShard, "knowledge-development-split")
	promotionRecord := testutil.ArtifactID(t, artifact.KindDatasetShard, "knowledge-promotion-record")
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "knowledge/evaluation/splits",
		Artifacts: []artifact.Descriptor{{ID: split}, {ID: developmentSplit}, {ID: promotionRecord}},
	}); err != nil {
		closeOnError(err)
	}
	caseValue, err := NewRetrievalCase(RetrievalCase{
		Query: trialSearch.receipt.Query, Split: split, Group: "knowledge-held-out",
		Judgments: []RetrievalJudgment{{
			Chunk: result.Citation.Chunk, Source: result.Citation.Source,
			StartRune: result.Citation.StartRune, EndRune: result.Citation.EndRune, Relevance: 3,
		}},
	})
	if err != nil {
		closeOnError(err)
	}
	baseline, err := PublishRetrievalEvaluation(t.Context(), store, caseValue, baselineSearch.receipt.ID)
	if err != nil {
		closeOnError(err)
	}
	trial, err := PublishRetrievalEvaluation(t.Context(), store, caseValue, trialSearch.receipt.ID)
	if err != nil {
		closeOnError(err)
	}

	registration := testutil.ArtifactID(t, artifact.KindProfile, "knowledge-transform-registration")
	operation := testutil.ArtifactID(t, artifact.KindRecipe, "knowledge-transform-operation")
	redaction := testutil.ArtifactID(t, artifact.KindRecipe, "knowledge-redaction-policy")
	sourceShard := testutil.ArtifactID(t, artifact.KindDatasetShard, "knowledge-episode-shard")
	replay := knowledgeTypedDocument(t, artifact.KindEvidence, "knowledge-transform-replay")
	conflict := knowledgeTypedDocument(t, artifact.KindEvidence, "knowledge-conflict-resolution")
	code, err := runrecord.NewCodeRevision(strings.Repeat("a", 40))
	if err != nil {
		closeOnError(err)
	}
	environment, err := runrecord.NewEnvironment(runrecord.Environment{
		Host: "knowledge-host", OS: "windows", Arch: "amd64", Device: "cpu",
		Backend: "go", Driver: "process", Runtime: "go-test",
	})
	if err != nil {
		closeOnError(err)
	}
	transform, err := dataset.NewDatasetTransform(dataset.DatasetTransformSpec{
		Registration: registration, Operation: operation, Parameters: redaction,
		Code: code.ID, Environment: environment.ID, Inputs: []artifact.ID{sourceShard},
		Outputs: []artifact.ID{trialDataset.Descriptor.ID}, ReplayEvidence: replay.Descriptor.ID,
	})
	if err != nil {
		closeOnError(err)
	}
	transformContent, err := transform.Content()
	if err != nil {
		closeOnError(err)
	}
	codePublication, err := code.Publication()
	if err != nil || len(codePublication.Contents) != 1 {
		closeOnError(errors.Join(errors.New("knowledge code publication"), err))
	}
	environmentContent, err := environment.Content()
	if err != nil {
		closeOnError(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "knowledge/transform",
		Artifacts: []artifact.Descriptor{
			{ID: registration}, {ID: operation}, {ID: redaction}, {ID: sourceShard},
		},
		Contents: []artifact.Content{
			replay, conflict, codePublication.Contents[0], environmentContent, transformContent,
		},
		Lineage: transform.Lineage(),
	}); err != nil {
		closeOnError(err)
	}
	promotionRecords := []artifact.ID{promotionRecord}
	if promotionOverlap {
		promotionRecords = []artifact.ID{sourceShard}
	}
	hygiene, err := EvaluateKnowledgeHygiene(t.Context(), store, KnowledgeHygieneObservation{
		Transform: transform.ID(), DevelopmentSplit: developmentSplit, PromotionSplit: split,
		InputRecordIDs: []artifact.ID{sourceShard, sourceShard}, OutputRecordIDs: []artifact.ID{sourceShard},
		ConflictEvidence: []artifact.ID{conflict.Descriptor.ID}, PromotionRecordIDs: promotionRecords,
	})
	if err != nil {
		closeOnError(err)
	}
	hygieneContent, err := hygiene.Content()
	if err != nil {
		closeOnError(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "knowledge/hygiene", Contents: []artifact.Content{hygieneContent}, Lineage: hygiene.Lineage(),
	}); err != nil {
		closeOnError(err)
	}

	id := func(kind artifact.Kind, name string) artifact.ID { return testutil.ArtifactID(t, kind, name) }
	binding, err := runrecord.NewAdmissionBinding(runrecord.AdmissionBinding{
		Generation:   0,
		Proposer:     runrecord.AuthorityDomain{Name: "knowledge-proposer", Identity: id(artifact.KindEvidence, "knowledge proposer")},
		Evaluator:    runrecord.AuthorityDomain{Name: "knowledge-evaluator", Identity: id(artifact.KindEvidence, "knowledge evaluator")},
		Decider:      runrecord.AuthorityDomain{Name: "knowledge-decider", Identity: id(artifact.KindEvidence, "knowledge decider")},
		SealedInputs: split, CleanWorker: id(artifact.KindEvidence, "knowledge clean worker"),
	})
	if err != nil {
		closeOnError(err)
	}
	developmentBudget, err := runrecord.NewBudget("provider-calls", developmentSplit, 1, binding.Decider.Identity)
	if err != nil {
		closeOnError(err)
	}
	promotionBudget, err := runrecord.NewBudget("provider-calls", split, 1, binding.Decider.Identity)
	if err != nil {
		closeOnError(err)
	}
	episode := knowledgeTypedDocument(t, artifact.KindEvidence, "knowledge episode source")
	baselineSource := knowledgeTypedDocument(t, artifact.KindEvidence, "knowledge baseline measurement")
	gapSource := knowledgeTypedDocument(t, artifact.KindEvidence, "knowledge gap measurement")
	falsifier := knowledgeTypedDocument(t, artifact.KindRecipe, "knowledge falsifier")
	baselineDecision := knowledgeDecision(t, baselineDataset.Descriptor.ID, binding.Evaluator.Identity, baselineSource.Descriptor.ID)
	gapDecision := knowledgeDecision(t, trialDataset.Descriptor.ID, binding.Evaluator.Identity, gapSource.Descriptor.ID)
	episodeDecision := knowledgeDecision(t, transform.ID(), binding.Evaluator.Identity, episode.Descriptor.ID)
	citationDecision := knowledgeDecision(t, transform.ID(), binding.Evaluator.Identity, trialSearch.receipt.ID)
	hygieneDecision := knowledgeDecision(t, transform.ID(), binding.Evaluator.Identity, hygiene.ID)
	candidate, err := modelrecipe.NewCandidate(modelrecipe.CandidateSpec{
		Subject: trialDataset.Descriptor.ID, Parent: baselineDataset.Descriptor.ID,
		Components: []modelrecipe.CandidateComponent{{
			Domain: modelrecipe.CandidateDatasetTransform, Specification: transform.ID(),
		}},
		Prediction: recipe.SteeringPrediction{
			Metric: "retrieval-recall", Benefit: 1, Cost: 1, Unit: "ratio", Uncertainty: 0,
		},
		CostUnit: "provider-calls", Falsifier: falsifier.Descriptor.ID,
		References: []modelrecipe.CandidateReference{
			{Role: modelrecipe.CandidateReferenceBaseline, Subject: baselineDataset.Descriptor.ID, Evidence: baselineDecision.ID},
			{Role: modelrecipe.CandidateReferenceGap, Subject: trialDataset.Descriptor.ID, Evidence: gapDecision.ID},
			{Role: modelrecipe.CandidateReferenceProvenance, Subject: transform.ID(), Evidence: episodeDecision.ID},
			{Role: modelrecipe.CandidateReferenceProvenance, Subject: transform.ID(), Evidence: citationDecision.ID},
			{Role: modelrecipe.CandidateReferenceMeasurement, Subject: transform.ID(), Evidence: hygieneDecision.ID},
		},
		DevelopmentSplit: developmentSplit, PromotionSplit: split,
		DevelopmentBudget: developmentBudget.ID, PromotionBudget: promotionBudget.ID,
		Code: code.ID, Environment: environment.ID,
	})
	if err != nil {
		closeOnError(err)
	}
	bindingContent, err := binding.Content()
	if err != nil {
		closeOnError(err)
	}
	developmentContent, err := developmentBudget.Content()
	if err != nil {
		closeOnError(err)
	}
	promotionContent, err := promotionBudget.Content()
	if err != nil {
		closeOnError(err)
	}
	candidateContent, err := candidate.Content()
	if err != nil {
		closeOnError(err)
	}
	decisions := []recipe.Decision{baselineDecision, gapDecision, episodeDecision, citationDecision, hygieneDecision}
	contents := []artifact.Content{
		bindingContent, developmentContent, promotionContent, episode, baselineSource, gapSource, falsifier, candidateContent,
	}
	lineage := append(developmentBudget.Lineage(), promotionBudget.Lineage()...)
	lineage = append(lineage, artifact.Lineage{
		Child: falsifier.Descriptor.ID, Parent: trialDataset.Descriptor.ID, Relation: artifact.RelationDependsOn,
	})
	for _, decision := range decisions {
		content, contentErr := decision.Content()
		if contentErr != nil {
			closeOnError(contentErr)
		}
		contents = append(contents, content)
		lineage = append(lineage, decision.Lineage()...)
	}
	lineage = append(lineage, candidate.Lineage()...)
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "knowledge/candidate",
		Artifacts: []artifact.Descriptor{
			{ID: binding.Proposer.Identity}, {ID: binding.Evaluator.Identity},
			{ID: binding.Decider.Identity}, {ID: binding.CleanWorker},
		},
		Contents: contents, Lineage: lineage,
	}); err != nil {
		closeOnError(err)
	}
	admission, err := runrecord.AdmitCandidate(
		t.Context(), store, candidate, binding.ID, modelrecipe.CandidateAdmissionAdapters()...,
	)
	if err != nil {
		closeOnError(err)
	}
	admissionContent, err := admission.Content()
	if err != nil {
		closeOnError(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "knowledge/admission", Contents: []artifact.Content{admissionContent}, Lineage: admission.Lineage(),
	}); err != nil {
		closeOnError(err)
	}
	proposal, err := PublishEpisodeKnowledgeProposal(t.Context(), store, admission.ID)
	if err != nil {
		closeOnError(err)
	}
	return knowledgePromotionFixture{
		path: path, store: store, proposal: proposal, admission: admission, candidate: candidate,
		transform: transform, hygiene: hygiene, baseline: baseline, trial: trial, split: split,
	}
}

func publishKnowledgeConsumedRetrieval(
	t *testing.T,
	store *overgodb.Store,
	tag string,
	datasetContent artifact.Content,
	texts []string,
	scores map[string]float64,
) consumedRetrievalFixture {
	t.Helper()
	datasetID := datasetContent.Descriptor.ID
	modelID := testutil.ArtifactID(t, artifact.KindModel, tag+"-model")
	rerankID := testutil.ArtifactID(t, artifact.KindProfile, tag+"-rerank")
	recipeID := testutil.ArtifactID(t, artifact.KindRecipe, tag+"-recipe")
	taskID := testutil.ArtifactID(t, artifact.KindRecipe, tag+"-task")
	operationID := testutil.ArtifactID(t, artifact.KindEvidence, tag+"-operation")
	strategyID := testutil.ArtifactID(t, artifact.KindProfile, tag+"-strategy")
	descriptors := []artifact.Descriptor{
		{ID: modelID}, {ID: rerankID}, {ID: recipeID}, {ID: taskID}, {ID: operationID}, {ID: strategyID},
	}
	documents := make([]dataset.AgentRetrievalDocument, len(texts))
	for index, sourceText := range texts {
		documents[index] = dataset.AgentRetrievalDocument{
			Text: sourceText, Facet: dataset.AgentRetrievalFacetText,
			Structure: []string{"fixture", tag},
		}
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: "retrieval/knowledge-fixture/" + tag, Artifacts: descriptors, Contents: []artifact.Content{datasetContent},
	}); err != nil {
		t.Fatal(err)
	}
	documents, err := dataset.PublishAgentRetrievalSource(t.Context(), store, artifact.KindFile, documents)
	if err != nil {
		t.Fatal(err)
	}
	embedder := evaluationEmbedder{model: modelID}
	reranker := evaluationReranker{policy: rerankID, scores: scores}
	builder := dataset.AgentRetrievalBuilder{Repository: store}
	projection, err := builder.Build(t.Context(), dataset.AgentRetrievalBuild{
		Dataset: datasetID, Documents: documents,
		Policy:   dataset.AgentRetrievalPolicy{MaximumChunkRunes: 128, CandidateLimit: uint64(len(texts))},
		Embedder: embedder, RerankPolicy: rerankID,
	})
	if err != nil {
		t.Fatal(err)
	}
	consumed, err := runrecord.SearchAndPublishConsumedRetrieval(t.Context(), store, dataset.AgentRetrievalQuery{
		Projection: projection.ID, Text: "target", Limit: 1, AllowedDatasets: []artifact.ID{datasetID},
		Embedder: embedder, Reranker: reranker,
	}, runrecord.RetrievalConsumer{
		Recipe: recipeID, Model: modelID, Operation: operationID, TaskContract: taskID, Strategy: strategyID,
	})
	if err != nil {
		t.Fatal(err)
	}
	return consumedRetrievalFixture{receipt: consumed.Receipt, search: consumed.Search}
}

func knowledgeDecision(
	t *testing.T,
	subject artifact.ID,
	evaluator artifact.ID,
	evidence ...artifact.ID,
) recipe.Decision {
	t.Helper()
	value, err := recipe.NewDecision(
		subject, recipe.DecisionObserved, recipe.EvidenceVerified, "",
		recipe.Decider{CodeCommit: strings.Repeat("b", 40), Derivation: evaluator}, evidence,
	)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func knowledgeTypedDocument(t *testing.T, kind artifact.Kind, label string) artifact.Content {
	t.Helper()
	body := struct {
		Label string `json:"label"`
	}{Label: label}
	id, err := artifact.JSONID(kind, body)
	if err != nil {
		t.Fatal(err)
	}
	contract := artifact.DocumentContract{
		Kind: kind, MediaType: "application/vnd.overgo.test-knowledge+json", Schema: "overgo/test-knowledge/v1",
	}
	content, err := contract.ContentJSON(id, body)
	if err != nil {
		t.Fatal(err)
	}
	return content
}

type knowledgeDocument interface {
	Content() (artifact.Content, error)
	Lineage() []artifact.Lineage
}

func publishKnowledgeDocument(t *testing.T, store *overgodb.Store, key string, value knowledgeDocument) {
	t.Helper()
	content, err := value.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{
		Key: key, Contents: []artifact.Content{content}, Lineage: value.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
}

type knowledgeRacingRepository struct {
	artifact.Repository
	marker artifact.ID
	raced  bool
}

func (repository *knowledgeRacingRepository) Commit(
	ctx context.Context,
	batch artifact.Batch,
) (artifact.CommitID, error) {
	if !repository.raced {
		repository.raced = true
		if _, err := repository.Repository.Commit(ctx, artifact.Batch{
			Key: "knowledge/concurrent-evidence", Artifacts: []artifact.Descriptor{{ID: repository.marker}},
		}); err != nil {
			return artifact.CommitID{}, err
		}
	}
	return repository.Repository.Commit(ctx, batch)
}
