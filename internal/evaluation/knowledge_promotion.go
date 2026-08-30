package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

const (
	episodeKnowledgeProposalMediaType = "application/vnd.overgo.episode-knowledge-proposal+json"
	episodeKnowledgeProposalSchema    = "overgo/episode-knowledge-proposal/v1"
	knowledgeHygieneMediaType         = "application/vnd.overgo.knowledge-hygiene-report+json"
	knowledgeHygieneSchema            = "overgo/knowledge-hygiene-report/v1"
	knowledgePromotionMediaType       = "application/vnd.overgo.knowledge-promotion-evidence+json"
	knowledgePromotionSchema          = "overgo/knowledge-promotion-evidence/v1"
)

// EpisodeKnowledgeProposal is a stored projection of an admitted candidate.
// Its transform, source, citation, and hygiene fields are derived from the
// candidate's typed decisions rather than supplied by the promotion caller.
type EpisodeKnowledgeProposal struct {
	Version         uint16        `json:"version"`
	Admission       artifact.ID   `json:"admission"`
	Candidate       artifact.ID   `json:"candidate"`
	Transform       artifact.ID   `json:"transform"`
	Incumbent       artifact.ID   `json:"incumbent"`
	Output          artifact.ID   `json:"output"`
	EpisodeSources  []artifact.ID `json:"episode_sources"`
	SourceShards    []artifact.ID `json:"source_shards"`
	Citations       []artifact.ID `json:"citations"`
	RedactionPolicy artifact.ID   `json:"redaction_policy"`
	Hygiene         artifact.ID   `json:"hygiene"`
	ID              artifact.ID   `json:"-"`
}

// KnowledgeHygieneObservation names the exact stored record evidence scanned
// by the hygiene evaluator. Aggregate counts are deliberately absent.
type KnowledgeHygieneObservation struct {
	Transform          artifact.ID
	DevelopmentSplit   artifact.ID
	PromotionSplit     artifact.ID
	InputRecordIDs     []artifact.ID
	OutputRecordIDs    []artifact.ID
	ConflictEvidence   []artifact.ID
	PromotionRecordIDs []artifact.ID
}

// KnowledgeHygieneReport stores the record-level observation and the
// canonical aggregates derived from it. Requiring the report replays those
// aggregates against the dataset transform and stored record identities.
type KnowledgeHygieneReport struct {
	Version            uint16        `json:"version"`
	Transform          artifact.ID   `json:"transform"`
	Output             artifact.ID   `json:"output"`
	DevelopmentSplit   artifact.ID   `json:"development_split"`
	PromotionSplit     artifact.ID   `json:"promotion_split"`
	RedactionPolicy    artifact.ID   `json:"redaction_policy"`
	ScanEvidence       artifact.ID   `json:"scan_evidence"`
	InputRecordIDs     []artifact.ID `json:"input_record_ids"`
	OutputRecordIDs    []artifact.ID `json:"output_record_ids"`
	ConflictEvidence   []artifact.ID `json:"conflict_evidence,omitempty"`
	PromotionRecordIDs []artifact.ID `json:"promotion_record_ids"`
	InputRecords       uint64        `json:"input_records"`
	OutputRecords      uint64        `json:"output_records"`
	DuplicatesRemoved  uint64        `json:"duplicates_removed,omitzero"`
	ConflictsResolved  uint64        `json:"conflicts_resolved,omitzero"`
	PromotionOverlap   uint64        `json:"promotion_overlap,omitzero"`
	ID                 artifact.ID   `json:"-"`
}

// KnowledgePromotionEvidence is emitted only for a strict held-out Pareto
// benefit with no quality or work regression.
type KnowledgePromotionEvidence struct {
	Version   uint16      `json:"version"`
	Proposal  artifact.ID `json:"proposal"`
	Admission artifact.ID `json:"admission"`
	Candidate artifact.ID `json:"candidate"`
	Transform artifact.ID `json:"transform"`
	Incumbent artifact.ID `json:"incumbent"`
	Output    artifact.ID `json:"output"`
	Split     artifact.ID `json:"split"`
	Baseline  artifact.ID `json:"baseline"`
	Trial     artifact.ID `json:"trial"`
	Benefits  []string    `json:"benefits"`
	ID        artifact.ID `json:"-"`
}

var episodeKnowledgeProposalCodec = artifact.JSONDocumentCodec(
	"episode knowledge proposal", artifact.KindEvidence,
	episodeKnowledgeProposalMediaType, episodeKnowledgeProposalSchema,
	canonicalizeEpisodeKnowledgeProposal,
	func(value EpisodeKnowledgeProposal) artifact.ID { return value.ID },
	func(value *EpisodeKnowledgeProposal, id artifact.ID) { value.ID = id },
	cloneEpisodeKnowledgeProposal,
)

var knowledgeHygieneCodec = artifact.JSONDocumentCodec(
	"knowledge hygiene report", artifact.KindEvidence,
	knowledgeHygieneMediaType, knowledgeHygieneSchema,
	canonicalizeKnowledgeHygieneReport,
	func(value KnowledgeHygieneReport) artifact.ID { return value.ID },
	func(value *KnowledgeHygieneReport, id artifact.ID) { value.ID = id },
	cloneKnowledgeHygieneReport,
)

var knowledgePromotionCodec = artifact.JSONDocumentCodec(
	"knowledge promotion evidence", artifact.KindEvidence,
	knowledgePromotionMediaType, knowledgePromotionSchema,
	canonicalizeKnowledgePromotionEvidence,
	func(value KnowledgePromotionEvidence) artifact.ID { return value.ID },
	func(value *KnowledgePromotionEvidence, id artifact.ID) { value.ID = id },
	func(value KnowledgePromotionEvidence) KnowledgePromotionEvidence {
		value.Benefits = slices.Clone(value.Benefits)
		return value
	},
)

// EvaluateKnowledgeHygiene derives transform, policy, scan, and aggregate
// facts from the exact stored transform and record-level observation.
func EvaluateKnowledgeHygiene(
	ctx context.Context,
	reader artifact.Reader,
	observation KnowledgeHygieneObservation,
) (KnowledgeHygieneReport, error) {
	if ctx == nil || reader == nil {
		return KnowledgeHygieneReport{}, errors.New("evaluation: knowledge hygiene authority is absent")
	}
	transform, err := dataset.RequireDatasetTransform(ctx, reader, observation.Transform)
	if err != nil {
		return KnowledgeHygieneReport{}, err
	}
	spec := transform.Spec()
	if len(spec.Outputs) != 1 || observation.DevelopmentSplit.Kind() != artifact.KindDatasetShard ||
		observation.PromotionSplit.Kind() != artifact.KindDatasetShard ||
		observation.DevelopmentSplit == observation.PromotionSplit {
		return KnowledgeHygieneReport{}, errors.New("evaluation: invalid knowledge hygiene observation authority")
	}
	required := []artifact.ID{observation.DevelopmentSplit, observation.PromotionSplit}
	required = append(required, observation.InputRecordIDs...)
	required = append(required, observation.OutputRecordIDs...)
	required = append(required, observation.ConflictEvidence...)
	required = append(required, observation.PromotionRecordIDs...)
	if err := requireStoredKnowledgeArtifacts(ctx, reader, required); err != nil {
		return KnowledgeHygieneReport{}, err
	}
	if _, err := artifact.RequireTypedContent(ctx, reader, spec.ReplayEvidence); err != nil {
		return KnowledgeHygieneReport{}, fmt.Errorf("evaluation: hygiene scan evidence: %w", err)
	}
	for _, evidence := range observation.ConflictEvidence {
		if _, err := artifact.RequireTypedContent(ctx, reader, evidence); err != nil {
			return KnowledgeHygieneReport{}, fmt.Errorf("evaluation: conflict evidence: %w", err)
		}
	}
	return knowledgeHygieneCodec.New(KnowledgeHygieneReport{
		Version: artifact.InitialDocumentVersion, Transform: transform.ID(), Output: spec.Outputs[0],
		DevelopmentSplit: observation.DevelopmentSplit, PromotionSplit: observation.PromotionSplit,
		RedactionPolicy: spec.Parameters, ScanEvidence: spec.ReplayEvidence,
		InputRecordIDs: slices.Clone(observation.InputRecordIDs), OutputRecordIDs: slices.Clone(observation.OutputRecordIDs),
		ConflictEvidence:   slices.Clone(observation.ConflictEvidence),
		PromotionRecordIDs: slices.Clone(observation.PromotionRecordIDs),
	})
}

// Content returns canonical hygiene evidence.
func (value KnowledgeHygieneReport) Content() (artifact.Content, error) {
	return knowledgeHygieneCodec.Content(value)
}

// Lineage binds the report to every authority from which its facts derive.
func (value KnowledgeHygieneReport) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Transform, value.Output, value.DevelopmentSplit, value.PromotionSplit,
		value.RedactionPolicy, value.ScanEvidence,
	}
	parents = append(parents, value.InputRecordIDs...)
	parents = append(parents, value.OutputRecordIDs...)
	parents = append(parents, value.ConflictEvidence...)
	parents = append(parents, value.PromotionRecordIDs...)
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

// RequireKnowledgeHygieneReport cold-loads and replays exact hygiene facts.
func RequireKnowledgeHygieneReport(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (KnowledgeHygieneReport, error) {
	value, err := knowledgeHygieneCodec.RequireExactLineage(ctx, reader, id, KnowledgeHygieneReport.Lineage)
	if err != nil {
		return KnowledgeHygieneReport{}, err
	}
	replayed, err := EvaluateKnowledgeHygiene(ctx, reader, KnowledgeHygieneObservation{
		Transform: value.Transform, DevelopmentSplit: value.DevelopmentSplit, PromotionSplit: value.PromotionSplit,
		InputRecordIDs: value.InputRecordIDs, OutputRecordIDs: value.OutputRecordIDs,
		ConflictEvidence: value.ConflictEvidence, PromotionRecordIDs: value.PromotionRecordIDs,
	})
	if err != nil || replayed.ID != value.ID {
		return KnowledgeHygieneReport{}, errors.Join(
			errors.New("evaluation: knowledge hygiene differs from stored observation"), err,
		)
	}
	return value, nil
}

// PublishEpisodeKnowledgeProposal derives and stores the sole proposal facts
// under a store-head compare-and-set.
func PublishEpisodeKnowledgeProposal(
	ctx context.Context,
	repository artifact.Repository,
	admissionID artifact.ID,
) (EpisodeKnowledgeProposal, error) {
	if ctx == nil || repository == nil {
		return EpisodeKnowledgeProposal{}, errors.New("evaluation: knowledge proposal repository is absent")
	}
	head, _ := repository.Head()
	value, err := deriveEpisodeKnowledgeProposal(ctx, repository, admissionID)
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	if _, found, readErr := episodeKnowledgeProposalCodec.Read(ctx, repository, value.ID); readErr != nil {
		return EpisodeKnowledgeProposal{}, readErr
	} else if found {
		return RequireEpisodeKnowledgeProposal(ctx, repository, value.ID)
	}
	content, err := value.Content()
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/knowledge-proposal/"+value.ID.String(), []artifact.Content{content}, value.Lineage(), nil,
	)
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	batch.ExpectedHead = &head
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	return RequireEpisodeKnowledgeProposal(ctx, repository, value.ID)
}

// Content returns the canonical proposal document.
func (value EpisodeKnowledgeProposal) Content() (artifact.Content, error) {
	return episodeKnowledgeProposalCodec.Content(value)
}

// Lineage binds every derived proposal fact and its admission authority.
func (value EpisodeKnowledgeProposal) Lineage() []artifact.Lineage {
	parents := []artifact.ID{
		value.Admission, value.Candidate, value.Transform, value.Incumbent,
		value.Output, value.RedactionPolicy, value.Hygiene,
	}
	parents = append(parents, value.EpisodeSources...)
	parents = append(parents, value.SourceShards...)
	parents = append(parents, value.Citations...)
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(value.ID, parents...)
}

// RequireEpisodeKnowledgeProposal refuses noncanonical lineage and any stored
// projection that no longer equals the admitted candidate's evidence.
func RequireEpisodeKnowledgeProposal(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (EpisodeKnowledgeProposal, error) {
	value, err := episodeKnowledgeProposalCodec.RequireExactLineage(ctx, reader, id, EpisodeKnowledgeProposal.Lineage)
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	replayed, err := deriveEpisodeKnowledgeProposal(ctx, reader, value.Admission)
	if err != nil || replayed.ID != value.ID {
		return EpisodeKnowledgeProposal{}, errors.Join(
			errors.New("evaluation: episode knowledge proposal differs from admitted evidence"), err,
		)
	}
	return value, nil
}

// EvaluateEpisodeKnowledgePromotion uses only stored, independently evaluated
// evidence. It refuses a tie, any regression, or development/promotion leakage.
func EvaluateEpisodeKnowledgePromotion(
	ctx context.Context,
	reader artifact.Reader,
	proposalID, baselineID, trialID artifact.ID,
) (KnowledgePromotionEvidence, error) {
	proposal, err := RequireEpisodeKnowledgeProposal(ctx, reader, proposalID)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	admission, candidate, err := requireAdmittedKnowledgeCandidate(ctx, reader, proposal.Admission)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	if admission.Candidate != proposal.Candidate {
		return KnowledgePromotionEvidence{}, errors.New("evaluation: knowledge proposal names another admitted candidate")
	}
	transform, err := dataset.RequireDatasetTransform(ctx, reader, proposal.Transform)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	hygiene, err := RequireKnowledgeHygieneReport(ctx, reader, proposal.Hygiene)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	baseline, err := RequireRetrievalEvaluation(ctx, reader, baselineID)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	trial, err := RequireRetrievalEvaluation(ctx, reader, trialID)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	baselineReceipt, err := runrecord.RequireConsumedRetrievalReceipt(ctx, reader, baseline.Receipt)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	trialReceipt, err := runrecord.RequireConsumedRetrievalReceipt(ctx, reader, trial.Receipt)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	if baselineReceipt.Dataset != proposal.Incumbent || trialReceipt.Dataset != proposal.Output {
		return KnowledgePromotionEvidence{}, errors.New("evaluation: retrieval evidence addresses another knowledge dataset")
	}
	if err := verifyKnowledgeCandidate(proposal, candidate, transform, hygiene, baseline, trial); err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	benefits, err := retrievalParetoBenefits(baseline, trial)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	return knowledgePromotionCodec.New(KnowledgePromotionEvidence{
		Version: artifact.InitialDocumentVersion, Proposal: proposal.ID, Admission: admission.ID,
		Candidate: candidate.ID(), Transform: transform.ID(), Incumbent: proposal.Incumbent, Output: proposal.Output,
		Split: candidate.Spec().PromotionSplit, Baseline: baseline.ID, Trial: trial.ID, Benefits: benefits,
	})
}

// PublishEpisodeKnowledgePromotion publishes the replayed claim under the
// store head observed before evaluation. Concurrent evidence wins; a stale
// promotion never lands as an apparently current fact.
func PublishEpisodeKnowledgePromotion(
	ctx context.Context,
	repository artifact.Repository,
	proposalID, baselineID, trialID artifact.ID,
) (KnowledgePromotionEvidence, error) {
	if ctx == nil || repository == nil {
		return KnowledgePromotionEvidence{}, errors.New("evaluation: knowledge promotion repository is absent")
	}
	head, _ := repository.Head()
	value, err := EvaluateEpisodeKnowledgePromotion(ctx, repository, proposalID, baselineID, trialID)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	content, err := value.Content()
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/knowledge-promotion/"+value.ID.String(), []artifact.Content{content}, value.Lineage(), nil,
	)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	batch.ExpectedHead = &head
	if _, err := artifact.CommitBatch(ctx, repository, batch); err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	return RequireKnowledgePromotionEvidence(ctx, repository, value.ID)
}

// Content returns canonical promotion evidence.
func (value KnowledgePromotionEvidence) Content() (artifact.Content, error) {
	return knowledgePromotionCodec.Content(value)
}

// ValidateIdentity proves the promotion claim did not change.
func (value KnowledgePromotionEvidence) ValidateIdentity() error {
	return knowledgePromotionCodec.ValidateIdentity(value)
}

// Lineage binds the promotion claim to admission, proposal, evaluations, and datasets.
func (value KnowledgePromotionEvidence) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Proposal, value.Admission, value.Candidate, value.Transform,
		value.Incumbent, value.Output, value.Split, value.Baseline, value.Trial)
}

// RequireKnowledgePromotionEvidence cold-loads exact lineage and replays the
// held-out comparison. A valid-looking caller-authored aggregate is refused.
func RequireKnowledgePromotionEvidence(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (KnowledgePromotionEvidence, error) {
	value, err := knowledgePromotionCodec.RequireExactLineage(ctx, reader, id, KnowledgePromotionEvidence.Lineage)
	if err != nil {
		return KnowledgePromotionEvidence{}, err
	}
	replayed, err := EvaluateEpisodeKnowledgePromotion(ctx, reader, value.Proposal, value.Baseline, value.Trial)
	if err != nil || replayed.ID != value.ID {
		return KnowledgePromotionEvidence{}, errors.Join(
			errors.New("evaluation: knowledge promotion differs from replayed held-out evidence"), err,
		)
	}
	return value, nil
}

func deriveEpisodeKnowledgeProposal(
	ctx context.Context,
	reader artifact.Reader,
	admissionID artifact.ID,
) (EpisodeKnowledgeProposal, error) {
	admission, candidate, err := requireAdmittedKnowledgeCandidate(ctx, reader, admissionID)
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	spec := candidate.Spec()
	var transformID artifact.ID
	for _, component := range spec.Components {
		if component.Domain != modelrecipe.CandidateDatasetTransform {
			continue
		}
		if transformID.Valid() {
			return EpisodeKnowledgeProposal{}, errors.New("evaluation: knowledge candidate has multiple dataset transforms")
		}
		transformID = component.Specification
	}
	if !transformID.Valid() {
		return EpisodeKnowledgeProposal{}, errors.New("evaluation: knowledge candidate has no dataset transform")
	}
	transform, err := dataset.RequireDatasetTransform(ctx, reader, transformID)
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	episodes, citations, hygieneIDs := []artifact.ID{}, []artifact.ID{}, []artifact.ID{}
	for _, reference := range spec.References {
		decision, decisionErr := recipe.RequireDecision(ctx, reader, reference.Evidence)
		if decisionErr != nil {
			return EpisodeKnowledgeProposal{}, decisionErr
		}
		switch {
		case reference.Role == modelrecipe.CandidateReferenceProvenance && reference.Subject == transformID:
			for _, source := range decision.Evidence {
				if _, receiptErr := runrecord.RequireConsumedRetrievalReceipt(ctx, reader, source); receiptErr == nil {
					citations = append(citations, source)
				} else {
					episodes = append(episodes, source)
				}
			}
		case reference.Role == modelrecipe.CandidateReferenceMeasurement && reference.Subject == transformID:
			for _, source := range decision.Evidence {
				if _, hygieneErr := RequireKnowledgeHygieneReport(ctx, reader, source); hygieneErr == nil {
					hygieneIDs = append(hygieneIDs, source)
				}
			}
		}
	}
	episodes, err = canonicalKnowledgeIDs(episodes, func(id artifact.ID) bool { return id.Kind() == artifact.KindEvidence })
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	citations, err = canonicalKnowledgeIDs(citations, func(id artifact.ID) bool { return id.Kind() == artifact.KindEvidence })
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	hygieneIDs, err = canonicalKnowledgeIDs(hygieneIDs, func(id artifact.ID) bool { return id.Kind() == artifact.KindEvidence })
	if err != nil || len(episodes) == 0 || len(citations) == 0 || len(hygieneIDs) != 1 {
		return EpisodeKnowledgeProposal{}, errors.Join(
			errors.New("evaluation: admitted knowledge evidence lacks exact episode, citation, or hygiene authority"), err,
		)
	}
	transformSpec := transform.Spec()
	if slices.Contains(transformSpec.Inputs, spec.PromotionSplit) {
		return EpisodeKnowledgeProposal{}, errors.New("evaluation: admitted knowledge transform consumes held-out promotion data")
	}
	hygiene, err := RequireKnowledgeHygieneReport(ctx, reader, hygieneIDs[0])
	if err != nil {
		return EpisodeKnowledgeProposal{}, err
	}
	if hygiene.Transform != transform.ID() || hygiene.Output != spec.Subject ||
		hygiene.DevelopmentSplit != spec.DevelopmentSplit || hygiene.PromotionSplit != spec.PromotionSplit ||
		hygiene.RedactionPolicy != transformSpec.Parameters {
		return EpisodeKnowledgeProposal{}, errors.New("evaluation: admitted hygiene evidence differs from candidate authority")
	}
	return episodeKnowledgeProposalCodec.New(EpisodeKnowledgeProposal{
		Version: artifact.InitialDocumentVersion, Admission: admission.ID, Candidate: candidate.ID(), Transform: transform.ID(),
		Incumbent: spec.Parent, Output: spec.Subject, EpisodeSources: episodes,
		SourceShards: transformSpec.Inputs, Citations: citations, RedactionPolicy: transformSpec.Parameters,
		Hygiene: hygiene.ID,
	})
}

func requireAdmittedKnowledgeCandidate(
	ctx context.Context,
	reader artifact.Reader,
	admissionID artifact.ID,
) (runrecord.CandidateAdmission, modelrecipe.Candidate, error) {
	admission, err := runrecord.RequireCandidateAdmission(ctx, reader, admissionID)
	if err != nil {
		return runrecord.CandidateAdmission{}, modelrecipe.Candidate{}, err
	}
	candidate, err := modelrecipe.RequireCandidate(ctx, reader, admission.Candidate)
	if err != nil {
		return runrecord.CandidateAdmission{}, modelrecipe.Candidate{}, err
	}
	replayed, err := runrecord.RequireReplayedCandidateAdmission(
		ctx, reader, admission.ID, candidate, modelrecipe.CandidateAdmissionAdapters()...,
	)
	if err != nil || replayed.ID != admission.ID {
		return runrecord.CandidateAdmission{}, modelrecipe.Candidate{}, errors.Join(
			errors.New("evaluation: candidate admission does not replay"), err,
		)
	}
	return admission, candidate, nil
}

func verifyKnowledgeCandidate(
	proposal EpisodeKnowledgeProposal,
	candidate modelrecipe.Candidate,
	transform dataset.DatasetTransform,
	hygiene KnowledgeHygieneReport,
	baseline, trial RetrievalEvaluation,
) error {
	spec := candidate.Spec()
	transformSpec := transform.Spec()
	datasetComponents := 0
	for _, component := range spec.Components {
		if component.Domain == modelrecipe.CandidateDatasetTransform {
			datasetComponents++
			if component.Specification != transform.ID() {
				return errors.New("evaluation: candidate names another dataset transform")
			}
		}
	}
	if datasetComponents != 1 || spec.Subject != proposal.Output || spec.Parent != proposal.Incumbent ||
		spec.DevelopmentSplit == spec.PromotionSplit || baseline.Split != spec.PromotionSplit || trial.Split != spec.PromotionSplit ||
		baseline.Case != trial.Case || baseline.Receipt == trial.Receipt || transformSpec.Parameters != proposal.RedactionPolicy ||
		transformSpec.Code != spec.Code || transformSpec.Environment != spec.Environment ||
		!slices.Contains(transformSpec.Outputs, proposal.Output) || !slices.Equal(transformSpec.Inputs, proposal.SourceShards) {
		return errors.New("evaluation: knowledge candidate authorities differ")
	}
	if hygiene.Transform != transform.ID() || hygiene.Output != proposal.Output ||
		hygiene.DevelopmentSplit != spec.DevelopmentSplit || hygiene.PromotionSplit != spec.PromotionSplit ||
		hygiene.RedactionPolicy != proposal.RedactionPolicy || hygiene.ScanEvidence != transformSpec.ReplayEvidence ||
		hygiene.PromotionOverlap != 0 {
		return errors.New("evaluation: knowledge candidate hygiene or held-out overlap differs")
	}
	return nil
}

func retrievalParetoBenefits(baseline, trial RetrievalEvaluation) ([]string, error) {
	left, right := baseline.Metrics, trial.Metrics
	if right.Recall < left.Recall || right.ReciprocalRank < left.ReciprocalRank || right.NormalizedDCG < left.NormalizedDCG ||
		right.CitationPrecision < left.CitationPrecision || right.CitationRecall < left.CitationRecall ||
		(left.AbstentionCorrect && !right.AbstentionCorrect) || right.RelevantExpected != left.RelevantExpected ||
		trial.Work.InspectedFacts > baseline.Work.InspectedFacts || trial.Work.LoadedFacts > baseline.Work.LoadedFacts ||
		trial.Work.LoadedBytes > baseline.Work.LoadedBytes || trial.Work.ContextBytes > baseline.Work.ContextBytes ||
		trial.Work.ProviderCost > baseline.Work.ProviderCost {
		return nil, errors.New("evaluation: held-out knowledge candidate regresses")
	}
	var benefits []string
	appendBenefit := func(name string, improved bool) {
		if improved {
			benefits = append(benefits, name)
		}
	}
	appendBenefit("recall", right.Recall > left.Recall)
	appendBenefit("reciprocal-rank", right.ReciprocalRank > left.ReciprocalRank)
	appendBenefit("normalized-dcg", right.NormalizedDCG > left.NormalizedDCG)
	appendBenefit("citation-precision", right.CitationPrecision > left.CitationPrecision)
	appendBenefit("citation-recall", right.CitationRecall > left.CitationRecall)
	appendBenefit("abstention", !left.AbstentionCorrect && right.AbstentionCorrect)
	appendBenefit("inspected-facts", trial.Work.InspectedFacts < baseline.Work.InspectedFacts)
	appendBenefit("loaded-facts", trial.Work.LoadedFacts < baseline.Work.LoadedFacts)
	appendBenefit("loaded-bytes", trial.Work.LoadedBytes < baseline.Work.LoadedBytes)
	appendBenefit("context-bytes", trial.Work.ContextBytes < baseline.Work.ContextBytes)
	appendBenefit("provider-cost", trial.Work.ProviderCost < baseline.Work.ProviderCost)
	if len(benefits) == 0 {
		return nil, errors.New("evaluation: held-out knowledge candidate has no measured benefit")
	}
	slices.Sort(benefits)
	return benefits, nil
}

func canonicalizeEpisodeKnowledgeProposal(value *EpisodeKnowledgeProposal) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Admission.Kind() != artifact.KindEvidence ||
		value.Candidate.Kind() != artifact.KindRecipe || value.Transform.Kind() != artifact.KindEvidence ||
		value.Incumbent.Kind() != artifact.KindDataset ||
		(value.Output.Kind() != artifact.KindDataset && value.Output.Kind() != artifact.KindDatasetShard) ||
		value.Output == value.Incumbent || value.RedactionPolicy.Kind() != artifact.KindRecipe ||
		value.Hygiene.Kind() != artifact.KindEvidence || len(value.EpisodeSources) == 0 ||
		len(value.SourceShards) == 0 || len(value.Citations) == 0 {
		return errors.New("evaluation: invalid episode knowledge proposal")
	}
	var err error
	if value.EpisodeSources, err = canonicalKnowledgeIDs(value.EpisodeSources, func(id artifact.ID) bool {
		return id.Kind() == artifact.KindEvidence
	}); err != nil {
		return err
	}
	if value.SourceShards, err = canonicalKnowledgeIDs(value.SourceShards, validKnowledgeRecordID); err != nil {
		return err
	}
	value.Citations, err = canonicalKnowledgeIDs(value.Citations, func(id artifact.ID) bool {
		return id.Kind() == artifact.KindEvidence
	})
	return err
}

func canonicalizeKnowledgeHygieneReport(value *KnowledgeHygieneReport) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Transform.Kind() != artifact.KindEvidence ||
		(value.Output.Kind() != artifact.KindDataset && value.Output.Kind() != artifact.KindDatasetShard) ||
		value.DevelopmentSplit.Kind() != artifact.KindDatasetShard || value.PromotionSplit.Kind() != artifact.KindDatasetShard ||
		value.DevelopmentSplit == value.PromotionSplit || value.RedactionPolicy.Kind() != artifact.KindRecipe ||
		value.ScanEvidence.Kind() != artifact.KindEvidence {
		return errors.New("evaluation: invalid knowledge hygiene report")
	}
	var err error
	value.InputRecordIDs, err = canonicalKnowledgeRecordIDs(value.InputRecordIDs, true)
	if err != nil {
		return err
	}
	value.OutputRecordIDs, err = canonicalKnowledgeRecordIDs(value.OutputRecordIDs, false)
	if err != nil {
		return err
	}
	value.PromotionRecordIDs, err = canonicalKnowledgeRecordIDs(value.PromotionRecordIDs, false)
	if err != nil {
		return err
	}
	value.ConflictEvidence, err = canonicalKnowledgeIDs(value.ConflictEvidence, func(id artifact.ID) bool {
		return id.Kind() == artifact.KindEvidence
	})
	if err != nil {
		return err
	}
	uniqueInputs := slices.Compact(slices.Clone(value.InputRecordIDs))
	if len(value.InputRecordIDs) == 0 || len(value.OutputRecordIDs) == 0 || len(value.PromotionRecordIDs) == 0 ||
		len(value.OutputRecordIDs) > len(uniqueInputs) {
		return errors.New("evaluation: invalid knowledge hygiene record observation")
	}
	value.InputRecords = uint64(len(value.InputRecordIDs))
	value.OutputRecords = uint64(len(value.OutputRecordIDs))
	value.DuplicatesRemoved = uint64(len(value.InputRecordIDs) - len(uniqueInputs))
	value.ConflictsResolved = uint64(len(value.ConflictEvidence))
	promotion := make(map[artifact.ID]bool, len(value.PromotionRecordIDs))
	for _, id := range value.PromotionRecordIDs {
		promotion[id] = true
	}
	var promotionOverlap uint64
	for _, id := range value.OutputRecordIDs {
		if promotion[id] {
			promotionOverlap++
		}
	}
	value.PromotionOverlap = promotionOverlap
	return nil
}

func canonicalKnowledgeRecordIDs(values []artifact.ID, allowDuplicates bool) ([]artifact.ID, error) {
	values = slices.Clone(values)
	slices.SortFunc(values, artifact.CompareID)
	for index, id := range values {
		if !validKnowledgeRecordID(id) || !allowDuplicates && index > 0 && values[index-1] == id {
			return nil, errors.New("evaluation: invalid or duplicate knowledge record evidence")
		}
	}
	return values, nil
}

func canonicalKnowledgeIDs(values []artifact.ID, valid func(artifact.ID) bool) ([]artifact.ID, error) {
	values = slices.Clone(values)
	slices.SortFunc(values, artifact.CompareID)
	for index, id := range values {
		if !valid(id) || index > 0 && values[index-1] == id {
			return nil, errors.New("evaluation: invalid or duplicate knowledge evidence")
		}
	}
	return values, nil
}

func validKnowledgeRecordID(id artifact.ID) bool {
	switch id.Kind() {
	case artifact.KindDataset, artifact.KindDatasetShard, artifact.KindFile:
		return true
	default:
		return false
	}
}

func canonicalizeKnowledgePromotionEvidence(value *KnowledgePromotionEvidence) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Proposal.Kind() != artifact.KindEvidence ||
		value.Admission.Kind() != artifact.KindEvidence || value.Candidate.Kind() != artifact.KindRecipe ||
		value.Transform.Kind() != artifact.KindEvidence || value.Incumbent.Kind() != artifact.KindDataset ||
		(value.Output.Kind() != artifact.KindDataset && value.Output.Kind() != artifact.KindDatasetShard) ||
		value.Split.Kind() != artifact.KindDatasetShard || value.Baseline.Kind() != artifact.KindEvaluation ||
		value.Trial.Kind() != artifact.KindEvaluation || value.Baseline == value.Trial || len(value.Benefits) == 0 {
		return errors.New("evaluation: invalid knowledge promotion evidence")
	}
	value.Benefits = slices.Clone(value.Benefits)
	slices.Sort(value.Benefits)
	for index, benefit := range value.Benefits {
		if !validKnowledgeBenefit(benefit) || index > 0 && value.Benefits[index-1] == benefit {
			return errors.New("evaluation: invalid knowledge promotion benefit")
		}
	}
	return nil
}

func validKnowledgeBenefit(value string) bool {
	switch value {
	case "recall", "reciprocal-rank", "normalized-dcg", "citation-precision", "citation-recall", "abstention",
		"inspected-facts", "loaded-facts", "loaded-bytes", "context-bytes", "provider-cost":
		return true
	default:
		return false
	}
}

func requireStoredKnowledgeArtifacts(ctx context.Context, reader artifact.Reader, ids []artifact.ID) error {
	for _, id := range ids {
		if _, present, err := reader.Artifact(ctx, id); err != nil || !present {
			return errors.Join(fmt.Errorf("evaluation: stored knowledge artifact %s is absent", id), err)
		}
	}
	return nil
}

func cloneEpisodeKnowledgeProposal(value EpisodeKnowledgeProposal) EpisodeKnowledgeProposal {
	value.EpisodeSources = slices.Clone(value.EpisodeSources)
	value.SourceShards = slices.Clone(value.SourceShards)
	value.Citations = slices.Clone(value.Citations)
	return value
}

func cloneKnowledgeHygieneReport(value KnowledgeHygieneReport) KnowledgeHygieneReport {
	value.InputRecordIDs = slices.Clone(value.InputRecordIDs)
	value.OutputRecordIDs = slices.Clone(value.OutputRecordIDs)
	value.ConflictEvidence = slices.Clone(value.ConflictEvidence)
	value.PromotionRecordIDs = slices.Clone(value.PromotionRecordIDs)
	return value
}
