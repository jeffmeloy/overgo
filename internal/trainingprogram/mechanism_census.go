package trainingprogram

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
)

const (
	// MechanismCensusMediaType identifies a mechanism census document.
	MechanismCensusMediaType = "application/vnd.overgo.mechanism-census+json"
	// MechanismCensusSchema identifies the mechanism census contract version.
	MechanismCensusSchema = "overgo/mechanism-census/v1"
)

var mechanismCensusContract = artifact.DocumentContract{
	Kind: artifact.KindRecipe, MediaType: MechanismCensusMediaType, Schema: MechanismCensusSchema,
}

// MechanismGapState says whether direct evidence leaves a capability gap.
type MechanismGapState string

const (
	// MechanismGapOpen says direct evidence leaves a capability gap.
	MechanismGapOpen MechanismGapState = "open"
	// MechanismGapClosed says the nearest owner covers the mechanism.
	MechanismGapClosed MechanismGapState = "closed"
)

// MechanismAssessment grounds one proposed mechanism against an existing owner.
type MechanismAssessment struct {
	Mechanism artifact.ID       `json:"mechanism"`
	Owner     artifact.ID       `json:"owner"`
	Gap       artifact.ID       `json:"gap"`
	Falsifier artifact.ID       `json:"falsifier"`
	GapState  MechanismGapState `json:"gap_state"`
}

type mechanismCensusBody struct {
	Assessments []MechanismAssessment `json:"assessments"`
}

// MechanismCensus is an immutable ownership and capability-gap assessment.
type MechanismCensus struct {
	id          artifact.ID
	assessments []MechanismAssessment
}

// CompileMechanismCensus validates and identifies a mechanism census.
func CompileMechanismCensus(assessments []MechanismAssessment) (MechanismCensus, error) {
	assessments = slices.Clone(assessments)
	slices.SortFunc(assessments, func(left, right MechanismAssessment) int {
		return cmp.Compare(left.Mechanism.String(), right.Mechanism.String())
	})
	if err := validateMechanismAssessments(assessments); err != nil {
		return MechanismCensus{}, err
	}
	body := mechanismCensusBody{Assessments: assessments}
	id, err := artifact.JSONID(artifact.KindRecipe, body)
	if err != nil {
		return MechanismCensus{}, err
	}
	return MechanismCensus{id: id, assessments: assessments}, nil
}

// ID returns the census content identity.
func (value MechanismCensus) ID() artifact.ID { return value.id }

// Assessments returns the canonical mechanism order.
func (value MechanismCensus) Assessments() []MechanismAssessment {
	return slices.Clone(value.assessments)
}

// Content returns the canonical census document.
func (value MechanismCensus) Content() (artifact.Content, error) {
	return mechanismCensusContract.ContentJSON(value.id, mechanismCensusBody{Assessments: value.assessments})
}

// Lineage binds every mechanism to its nearest owner, gap evidence, and falsifier.
func (value MechanismCensus) Lineage() []artifact.Lineage {
	parents := make([]artifact.ID, 0, len(value.assessments)*4)
	for _, assessment := range value.assessments {
		parents = append(parents, assessment.Mechanism, assessment.Owner, assessment.Gap, assessment.Falsifier)
	}
	return artifact.DependencyLineage(value.id, parents...)
}

func validateMechanismAssessments(assessments []MechanismAssessment) error {
	if len(assessments) == 0 {
		return errors.New("training program: mechanism census is empty")
	}
	for index, assessment := range assessments {
		if assessment.Mechanism.Kind() != artifact.KindRecipe || !assessment.Owner.Valid() ||
			assessment.Gap.Kind() != artifact.KindEvidence || assessment.Falsifier.Kind() != artifact.KindRecipe ||
			assessment.Mechanism == assessment.Owner || assessment.Mechanism == assessment.Falsifier ||
			assessment.Owner == assessment.Gap || assessment.Owner == assessment.Falsifier || assessment.Gap == assessment.Falsifier ||
			assessment.GapState != MechanismGapOpen && assessment.GapState != MechanismGapClosed {
			return errors.New("training program: invalid mechanism assessment")
		}
		if index > 0 && assessments[index-1].Mechanism == assessment.Mechanism {
			return errors.New("training program: duplicate mechanism assessment")
		}
	}
	return nil
}
