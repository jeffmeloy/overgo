package modelrecipe

import (
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	// CandidateMediaType identifies the one cross-domain candidate envelope.
	CandidateMediaType = "application/vnd.overgo.cross-domain-candidate+json"
	// CandidateSchema identifies the cross-domain candidate contract version.
	CandidateSchema = "overgo/cross-domain-candidate/v1"
)

// CandidateDomain names one compiled-Go plugin domain without granting it execution authority.
type CandidateDomain string

const (
	// CandidateModelPrototype binds a registered model prototype specification.
	CandidateModelPrototype CandidateDomain = "model-prototype"
	// CandidateComposition binds a composition specification or blocked proposal.
	CandidateComposition CandidateDomain = "composition"
	// CandidateMechanism binds a research mechanism specification.
	CandidateMechanism CandidateDomain = "mechanism"
	// CandidateDatasetTransform binds a content-addressed dataset transform specification.
	CandidateDatasetTransform CandidateDomain = "dataset-transform"
	// CandidateRouting binds a routing objective specification.
	CandidateRouting CandidateDomain = "routing"
	// CandidateSteering binds an inference-time steering specification.
	CandidateSteering CandidateDomain = "steering"
	// CandidateCode binds a proposed compiled-Go code change.
	CandidateCode CandidateDomain = "code"
)

// CandidateComponent binds one immutable domain specification. The specification's
// own lineage remains the sole owner of dependencies within that domain.
type CandidateComponent struct {
	Domain        CandidateDomain `json:"domain"`
	Specification artifact.ID     `json:"specification"`
}

// CandidateReferenceRole names why evidence is relevant to a candidate subject.
type CandidateReferenceRole string

const (
	// CandidateReferenceBaseline identifies incumbent or parent evidence.
	CandidateReferenceBaseline CandidateReferenceRole = "baseline"
	// CandidateReferenceGap identifies evidence of the measured capability gap.
	CandidateReferenceGap CandidateReferenceRole = "gap"
	// CandidateReferenceMeasurement identifies a direct measurement of the subject.
	CandidateReferenceMeasurement CandidateReferenceRole = "measurement"
	// CandidateReferenceProvenance identifies exact external or derivation provenance.
	CandidateReferenceProvenance CandidateReferenceRole = "provenance"
)

// CandidateReference binds evidence to the exact candidate, parent, or component
// subject it describes. Admission resolves the evidence's typed semantic claim.
type CandidateReference struct {
	Role     CandidateReferenceRole `json:"role"`
	Subject  artifact.ID            `json:"subject"`
	Evidence artifact.ID            `json:"evidence"`
}

// CandidateSpec is the complete non-authorizing cross-domain hypothesis input.
// Budget IDs bind runrecord grants; Code and Environment bind exact evidence.
type CandidateSpec struct {
	Subject           artifact.ID               `json:"subject"`
	Parent            artifact.ID               `json:"parent"`
	Components        []CandidateComponent      `json:"components"`
	Prediction        recipe.SteeringPrediction `json:"prediction"`
	Falsifier         artifact.ID               `json:"falsifier"`
	References        []CandidateReference      `json:"references"`
	DevelopmentSplit  artifact.ID               `json:"development_split"`
	PromotionSplit    artifact.ID               `json:"promotion_split"`
	DevelopmentBudget artifact.ID               `json:"development_budget"`
	PromotionBudget   artifact.ID               `json:"promotion_budget"`
	Code              artifact.ID               `json:"code"`
	Environment       artifact.ID               `json:"environment"`
}

type candidateDocument struct {
	Version uint16 `json:"version"`
	CandidateSpec
	ID artifact.ID `json:"-"`
}

// Candidate is one immutable cross-domain hypothesis. It carries no counters,
// callbacks, active recipes, weights, lifecycle state, or capability grants.
type Candidate struct {
	document candidateDocument
}

var candidateCodec = artifact.JSONDocumentCodec(
	"cross-domain candidate", artifact.KindRecipe, CandidateMediaType, CandidateSchema,
	canonicalizeCandidateDocument,
	func(value candidateDocument) artifact.ID { return value.ID },
	func(value *candidateDocument, id artifact.ID) { value.ID = id },
	cloneCandidateDocument,
)

// NewCandidate canonicalizes and identifies one cross-domain hypothesis.
func NewCandidate(spec CandidateSpec) (Candidate, error) {
	document, err := candidateCodec.New(candidateDocument{
		Version: artifact.InitialDocumentVersion, CandidateSpec: cloneCandidateSpec(spec),
	})
	return Candidate{document: document}, err
}

// ID returns the candidate's content identity.
func (value Candidate) ID() artifact.ID { return value.document.ID }

// Spec returns an isolated copy of the candidate hypothesis.
func (value Candidate) Spec() CandidateSpec { return cloneCandidateSpec(value.document.CandidateSpec) }

// ValidateIdentity verifies the candidate's canonical content identity.
func (value Candidate) ValidateIdentity() error {
	return candidateCodec.ValidateIdentity(value.document)
}

// Content returns the canonical candidate document.
func (value Candidate) Content() (artifact.Content, error) {
	return candidateCodec.Content(value.document)
}

// Lineage binds every component, reference, budget, split, and execution-evidence authority.
func (value Candidate) Lineage() []artifact.Lineage {
	document := value.document
	parents := []artifact.ID{
		document.Subject, document.Parent, document.Falsifier,
		document.DevelopmentSplit, document.PromotionSplit,
		document.DevelopmentBudget, document.PromotionBudget,
		document.Code, document.Environment,
	}
	for _, component := range document.Components {
		parents = append(parents, component.Specification)
	}
	for _, reference := range document.References {
		parents = append(parents, reference.Subject, reference.Evidence)
	}
	slices.SortFunc(parents, artifact.CompareID)
	parents = slices.Compact(parents)
	return artifact.DependencyLineage(document.ID, parents...)
}

func canonicalizeCandidateDocument(value *candidateDocument) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		!value.Subject.Valid() || !value.Parent.Valid() || value.Parent == value.Subject ||
		len(value.Components) == 0 || value.Falsifier.Kind() != artifact.KindRecipe || len(value.References) == 0 ||
		value.DevelopmentSplit.Kind() != artifact.KindDatasetShard || value.PromotionSplit.Kind() != artifact.KindDatasetShard ||
		value.DevelopmentSplit == value.PromotionSplit || value.DevelopmentBudget.Kind() != artifact.KindEvidence ||
		value.PromotionBudget.Kind() != artifact.KindEvidence || value.Code.Kind() != artifact.KindEvidence ||
		value.Environment.Kind() != artifact.KindEvidence ||
		!candidateDistinct(value.DevelopmentBudget, value.PromotionBudget, value.Code, value.Environment) {
		return errors.New("model recipe: invalid cross-domain candidate envelope")
	}
	if err := value.Prediction.Validate(); err != nil {
		return err
	}
	if err := canonicalizeCandidateComponents(value); err != nil {
		return err
	}
	return canonicalizeCandidateReferences(value)
}

func canonicalizeCandidateComponents(value *candidateDocument) error {
	value.Components = slices.Clone(value.Components)
	slices.SortFunc(value.Components, func(left, right CandidateComponent) int {
		if order := strings.Compare(string(left.Domain), string(right.Domain)); order != 0 {
			return order
		}
		return artifact.CompareID(left.Specification, right.Specification)
	})
	seen := make(map[artifact.ID]bool, len(value.Components))
	for _, component := range value.Components {
		if !candidateSpecificationKind(component.Domain, component.Specification.Kind()) || seen[component.Specification] ||
			component.Specification == value.Falsifier {
			return errors.New("model recipe: invalid or duplicate candidate component")
		}
		seen[component.Specification] = true
	}
	return nil
}

func canonicalizeCandidateReferences(value *candidateDocument) error {
	relevant := map[artifact.ID]bool{value.Subject: true, value.Parent: true}
	for _, component := range value.Components {
		relevant[component.Specification] = true
	}
	value.References = slices.Clone(value.References)
	slices.SortFunc(value.References, func(left, right CandidateReference) int {
		if order := strings.Compare(string(left.Role), string(right.Role)); order != 0 {
			return order
		}
		if order := artifact.CompareID(left.Subject, right.Subject); order != 0 {
			return order
		}
		return artifact.CompareID(left.Evidence, right.Evidence)
	})
	for index, reference := range value.References {
		if !reference.Role.valid() || !relevant[reference.Subject] || reference.Evidence.Kind() != artifact.KindEvidence ||
			index > 0 && reference == value.References[index-1] {
			return errors.New("model recipe: candidate reference is duplicate, untyped, or irrelevant")
		}
	}
	return nil
}

func cloneCandidateDocument(value candidateDocument) candidateDocument {
	value.CandidateSpec = cloneCandidateSpec(value.CandidateSpec)
	return value
}

func cloneCandidateSpec(value CandidateSpec) CandidateSpec {
	value.Components = slices.Clone(value.Components)
	value.References = slices.Clone(value.References)
	return value
}

func candidateSpecificationKind(domain CandidateDomain, kind artifact.Kind) bool {
	switch domain {
	case CandidateModelPrototype, CandidateMechanism, CandidateRouting, CandidateSteering:
		return kind == artifact.KindRecipe
	case CandidateComposition:
		return kind == artifact.KindRecipe || kind == artifact.KindEvidence
	case CandidateDatasetTransform, CandidateCode:
		return kind == artifact.KindEvidence
	default:
		return false
	}
}

func (role CandidateReferenceRole) valid() bool {
	switch role {
	case CandidateReferenceBaseline, CandidateReferenceGap, CandidateReferenceMeasurement, CandidateReferenceProvenance:
		return true
	default:
		return false
	}
}

func candidateDistinct(values ...artifact.ID) bool {
	seen := make(map[artifact.ID]bool, len(values))
	for _, value := range values {
		if !value.Valid() || seen[value] {
			return false
		}
		seen[value] = true
	}
	return true
}
