package runrecord

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	// CandidateAdmissionMediaType identifies one cross-domain eligibility decision.
	CandidateAdmissionMediaType = "application/vnd.overgo.candidate-admission+json"
	// CandidateAdmissionSchema identifies the cross-domain eligibility contract.
	CandidateAdmissionSchema = "overgo/candidate-admission/v1"
)

// CandidateAdmissionComponent is the admission owner's package-neutral view of
// one compiled-Go component specification.
type CandidateAdmissionComponent struct {
	Domain        string
	Specification artifact.ID
}

// CandidateAdmissionReference is the admission owner's package-neutral view
// of one role- and subject-bound decision.
type CandidateAdmissionReference struct {
	Role     string
	Subject  artifact.ID
	Evidence artifact.ID
}

// CandidateAdmissionFacts is the immutable projection a candidate supplies to
// the sole admission owner. It deliberately carries no execution or lifecycle state.
type CandidateAdmissionFacts struct {
	Candidate         artifact.ID
	Subject           artifact.ID
	Parent            artifact.ID
	Components        []CandidateAdmissionComponent
	PredictionMetric  string
	PredictedBenefit  float64
	PredictedCost     uint64
	BenefitUnit       string
	CostUnit          string
	Falsifier         artifact.ID
	References        []CandidateAdmissionReference
	DevelopmentSplit  artifact.ID
	PromotionSplit    artifact.ID
	DevelopmentBudget artifact.ID
	PromotionBudget   artifact.ID
	Code              artifact.ID
	Environment       artifact.ID
}

// CandidateAdmissionSubject exposes immutable candidate facts without making
// runrecord depend on the candidate-owning modelrecipe package.
type CandidateAdmissionSubject interface {
	ValidateIdentity() error
	AdmissionFacts() CandidateAdmissionFacts
}

// CandidateComponentAdmissionAdapter validates one domain specification
// through its existing typed owner. Adapters supply facts; only AdmitCandidate
// can issue the eligibility decision.
type CandidateComponentAdmissionAdapter interface {
	CandidateDomain() string
	ValidateCandidateComponent(context.Context, artifact.Reader, CandidateAdmissionFacts, CandidateAdmissionComponent) error
}

// CandidateAdmission grants eligibility only. It cannot construct, execute,
// select, activate, promote, or mutate an alias.
type CandidateAdmission struct {
	Version   uint16      `json:"version"`
	Candidate artifact.ID `json:"candidate"`
	Authority artifact.ID `json:"authority"`
	ID        artifact.ID `json:"-"`
}

var candidateAdmissionCodec = artifact.JSONDocumentCodec(
	"cross-domain candidate admission", artifact.KindEvidence,
	CandidateAdmissionMediaType, CandidateAdmissionSchema,
	canonicalizeCandidateAdmission,
	func(value CandidateAdmission) artifact.ID { return value.ID },
	func(value *CandidateAdmission, id artifact.ID) { value.ID = id }, nil,
)

// AdmitCandidate is the single cross-domain eligibility decision owner.
func AdmitCandidate(
	ctx context.Context,
	reader artifact.Reader,
	candidate CandidateAdmissionSubject,
	authority artifact.ID,
	adapters ...CandidateComponentAdmissionAdapter,
) (CandidateAdmission, error) {
	return admitCandidate(ctx, reader, candidate, authority, adapters...)
}

func admitCandidate(
	ctx context.Context,
	reader artifact.Reader,
	candidate CandidateAdmissionSubject,
	authority artifact.ID,
	adapters ...CandidateComponentAdmissionAdapter,
) (CandidateAdmission, error) {
	if ctx == nil || reader == nil || candidate == nil {
		return CandidateAdmission{}, errors.New("run record: candidate admission authority is absent")
	}
	if err := candidate.ValidateIdentity(); err != nil {
		return CandidateAdmission{}, fmt.Errorf("run record: candidate identity: %w", err)
	}
	facts := candidate.AdmissionFacts()
	if err := validateCandidateAdmissionFacts(facts); err != nil {
		return CandidateAdmission{}, err
	}
	binding, err := RequireAdmissionBinding(ctx, reader, authority)
	if err != nil {
		return CandidateAdmission{}, fmt.Errorf("run record: candidate authority: %w", err)
	}
	development, err := RequireBudget(ctx, reader, facts.DevelopmentBudget)
	if err != nil {
		return CandidateAdmission{}, fmt.Errorf("run record: development budget: %w", err)
	}
	promotion, err := RequireBudget(ctx, reader, facts.PromotionBudget)
	if err != nil {
		return CandidateAdmission{}, fmt.Errorf("run record: promotion budget: %w", err)
	}
	if err := validateCandidateBudgets(facts, binding, development, promotion); err != nil {
		return CandidateAdmission{}, err
	}
	if _, err := RequireCodeRevision(ctx, reader, facts.Code); err != nil {
		return CandidateAdmission{}, fmt.Errorf("run record: candidate code: %w", err)
	}
	if _, err := RequireEnvironment(ctx, reader, facts.Environment); err != nil {
		return CandidateAdmission{}, fmt.Errorf("run record: candidate environment: %w", err)
	}
	for _, id := range []artifact.ID{facts.Subject, facts.Parent} {
		if err := requireCandidateArtifactContent(ctx, reader, id); err != nil {
			return CandidateAdmission{}, fmt.Errorf("run record: candidate subject %s: %w", id, err)
		}
	}
	if _, err := artifact.RequireTypedContent(ctx, reader, facts.Falsifier); err != nil {
		return CandidateAdmission{}, fmt.Errorf("run record: candidate falsifier: %w", err)
	}
	if err := requireCandidateLineage(ctx, reader, facts.Falsifier, facts.Subject); err != nil {
		return CandidateAdmission{}, fmt.Errorf("run record: candidate falsifier: %w", err)
	}
	if err := validateCandidateComponents(ctx, reader, facts, adapters); err != nil {
		return CandidateAdmission{}, err
	}
	if err := validateCandidateReferences(ctx, reader, facts, binding); err != nil {
		return CandidateAdmission{}, err
	}
	return candidateAdmissionCodec.New(CandidateAdmission{
		Version: artifact.InitialDocumentVersion, Candidate: facts.Candidate, Authority: authority,
	})
}

// Content returns the canonical eligibility decision.
func (value CandidateAdmission) Content() (artifact.Content, error) {
	return candidateAdmissionCodec.Content(value)
}

// Lineage binds eligibility to the exact candidate and independent authority.
func (value CandidateAdmission) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Candidate, value.Authority)
}

// RequireCandidateAdmission loads one exact eligibility decision. Callers
// that consume the decision as authority must additionally replay admission
// with the candidate owner's closed adapter set.
func RequireCandidateAdmission(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (CandidateAdmission, error) {
	return candidateAdmissionCodec.RequireExactLineage(ctx, reader, id, CandidateAdmission.Lineage)
}

// RequireReplayedCandidateAdmission keeps admission replay inside the sole
// admission owner while allowing consumers to supply the candidate owner's
// package-neutral subject and closed adapter set.
func RequireReplayedCandidateAdmission(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	candidate CandidateAdmissionSubject,
	adapters ...CandidateComponentAdmissionAdapter,
) (CandidateAdmission, error) {
	stored, err := RequireCandidateAdmission(ctx, reader, id)
	if err != nil {
		return CandidateAdmission{}, err
	}
	replayed, err := admitCandidate(ctx, reader, candidate, stored.Authority, adapters...)
	if err != nil || replayed.ID != stored.ID {
		return CandidateAdmission{}, errors.Join(errors.New("run record: candidate admission does not replay"), err)
	}
	return stored, nil
}

func canonicalizeCandidateAdmission(value *CandidateAdmission) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Candidate.Kind() != artifact.KindRecipe || value.Authority.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid candidate admission")
	}
	return nil
}

func validateCandidateAdmissionFacts(value CandidateAdmissionFacts) error {
	if value.Candidate.Kind() != artifact.KindRecipe || !value.Subject.Valid() || !value.Parent.Valid() || value.Subject == value.Parent ||
		len(value.Components) == 0 || strings.TrimSpace(value.PredictionMetric) == "" ||
		math.IsNaN(value.PredictedBenefit) || math.IsInf(value.PredictedBenefit, 0) ||
		value.PredictedBenefit <= 0 || value.PredictedCost == 0 || strings.TrimSpace(value.BenefitUnit) == "" ||
		strings.TrimSpace(value.CostUnit) == "" || value.Falsifier.Kind() != artifact.KindRecipe || len(value.References) == 0 ||
		value.DevelopmentSplit.Kind() != artifact.KindDatasetShard || value.PromotionSplit.Kind() != artifact.KindDatasetShard ||
		value.DevelopmentSplit == value.PromotionSplit || value.DevelopmentBudget.Kind() != artifact.KindEvidence ||
		value.PromotionBudget.Kind() != artifact.KindEvidence || value.Code.Kind() != artifact.KindEvidence ||
		value.Environment.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid candidate admission facts")
	}
	return nil
}

func requireCandidateArtifactContent(ctx context.Context, reader artifact.Reader, id artifact.ID) error {
	_, stream, present, err := reader.OpenContent(ctx, id)
	if err != nil {
		return err
	}
	if closer, ok := stream.(interface{ Close() error }); ok {
		if closeErr := closer.Close(); closeErr != nil {
			return closeErr
		}
	}
	if !present {
		return errors.New("content is absent")
	}
	return nil
}

func validateCandidateBudgets(
	facts CandidateAdmissionFacts,
	binding AdmissionBinding,
	development, promotion Budget,
) error {
	if development.Split != facts.DevelopmentSplit || promotion.Split != facts.PromotionSplit {
		return errors.New("run record: candidate budget split differs")
	}
	if development.Unit != facts.CostUnit || promotion.Unit != facts.CostUnit {
		return errors.New("run record: candidate budget unit differs")
	}
	if development.Issued < facts.PredictedCost || promotion.Issued < facts.PredictedCost {
		return errors.New("run record: candidate predicted cost exceeds its split grant")
	}
	for _, budget := range []Budget{development, promotion} {
		if budget.Authority != binding.Decider.Identity {
			return errors.New("run record: candidate budget grant was not issued by its admission authority")
		}
	}
	if binding.SealedInputs != facts.PromotionSplit {
		return errors.New("run record: candidate promotion split is not the sealed admission input")
	}
	return nil
}

func validateCandidateComponents(
	ctx context.Context,
	reader artifact.Reader,
	facts CandidateAdmissionFacts,
	adapters []CandidateComponentAdmissionAdapter,
) error {
	byDomain := make(map[string]CandidateComponentAdmissionAdapter, len(adapters))
	for _, adapter := range adapters {
		if adapter == nil || strings.TrimSpace(adapter.CandidateDomain()) == "" {
			return errors.New("run record: invalid candidate admission adapter")
		}
		if _, duplicate := byDomain[adapter.CandidateDomain()]; duplicate {
			return fmt.Errorf("run record: duplicate candidate admission adapter %q", adapter.CandidateDomain())
		}
		byDomain[adapter.CandidateDomain()] = adapter
	}
	for _, component := range facts.Components {
		adapter, present := byDomain[component.Domain]
		if !present {
			return fmt.Errorf("run record: candidate domain %q has no compiled Go admission adapter", component.Domain)
		}
		if err := adapter.ValidateCandidateComponent(ctx, reader, facts, component); err != nil {
			return fmt.Errorf("run record: candidate %s component: %w", component.Domain, err)
		}
	}
	return nil
}

func validateCandidateReferences(
	ctx context.Context,
	reader artifact.Reader,
	facts CandidateAdmissionFacts,
	binding AdmissionBinding,
) error {
	baseline, gap := false, false
	provenance := make(map[artifact.ID]bool, len(facts.Components))
	seen := make(map[artifact.ID]bool, len(facts.References))
	for _, reference := range facts.References {
		if !candidateAdmissionReferenceRole(reference.Role) || seen[reference.Evidence] {
			return errors.New("run record: duplicate or unknown candidate reference")
		}
		seen[reference.Evidence] = true
		decision, err := recipe.RequireDecision(ctx, reader, reference.Evidence)
		if err != nil {
			return fmt.Errorf("run record: candidate reference: %w", err)
		}
		if decision.Subject != reference.Subject || decision.Decider.Derivation != binding.Evaluator.Identity ||
			(decision.Outcome != recipe.DecisionObserved && decision.Outcome != recipe.DecisionAccepted) || len(decision.Evidence) == 0 {
			return errors.New("run record: candidate reference is not relevant typed evaluator evidence")
		}
		if err := requireCandidateLineage(ctx, reader, decision.ID, reference.Subject); err != nil {
			return fmt.Errorf("run record: candidate reference: %w", err)
		}
		for _, evidence := range decision.Evidence {
			if evidence == binding.Proposer.Identity || evidence == binding.Evaluator.Identity || evidence == binding.Decider.Identity {
				return errors.New("run record: candidate reference evidence is not independent")
			}
			if _, err := artifact.RequireTypedContent(ctx, reader, evidence); err != nil {
				return fmt.Errorf("run record: candidate reference source: %w", err)
			}
		}
		switch reference.Role {
		case "baseline":
			if reference.Subject != facts.Parent {
				return errors.New("run record: baseline reference does not describe the exact parent")
			}
			baseline = true
		case "gap":
			if reference.Subject != facts.Subject {
				return errors.New("run record: gap reference does not describe the exact candidate subject")
			}
			gap = true
		case "measurement":
			if !candidateMeasurementSubject(facts, reference.Subject) {
				return errors.New("run record: measurement reference is outside the candidate scope")
			}
		case "provenance":
			if reference.Subject != facts.Subject && !candidateComponentSubject(facts, reference.Subject) {
				return errors.New("run record: provenance reference is outside the candidate scope")
			}
			provenance[reference.Subject] = true
		}
	}
	if !baseline || !gap {
		return errors.New("run record: candidate lacks exact parent baseline or subject gap evidence")
	}
	for _, component := range facts.Components {
		if !provenance[component.Specification] {
			return fmt.Errorf("run record: candidate component %s lacks provenance evidence", component.Specification)
		}
	}
	return nil
}

func candidateMeasurementSubject(facts CandidateAdmissionFacts, subject artifact.ID) bool {
	return subject == facts.Subject || subject == facts.Parent || candidateComponentSubject(facts, subject)
}

func candidateComponentSubject(facts CandidateAdmissionFacts, subject artifact.ID) bool {
	for _, component := range facts.Components {
		if component.Specification == subject {
			return true
		}
	}
	return false
}

func candidateAdmissionReferenceRole(role string) bool {
	switch role {
	case "baseline", "gap", "measurement", "provenance":
		return true
	default:
		return false
	}
}

func requireCandidateLineage(ctx context.Context, reader artifact.Reader, child, parent artifact.ID) error {
	edges, err := reader.Parents(ctx, child)
	if err != nil {
		return err
	}
	for _, edge := range edges {
		if edge.Child == child && edge.Parent == parent && edge.Relation == artifact.RelationDependsOn {
			return nil
		}
	}
	return fmt.Errorf("typed document %s is not relevant to subject %s", child, parent)
}
