package trainingprogram

import (
	"errors"

	"overgo/internal/artifact"
)

const (
	// MechanismCandidateMediaType identifies an evidence-gated mechanism candidate.
	MechanismCandidateMediaType = "application/vnd.overgo.mechanism-candidate+json"
	// MechanismCandidateSchema identifies the mechanism candidate contract version.
	MechanismCandidateSchema = "overgo/mechanism-candidate/v1"
)

var mechanismCandidateContract = artifact.DocumentContract{
	Kind: artifact.KindRecipe, MediaType: MechanismCandidateMediaType, Schema: MechanismCandidateSchema,
}

type mechanismCandidateBody struct {
	Census     artifact.ID `json:"census"`
	Provenance artifact.ID `json:"provenance"`
	Mechanism  artifact.ID `json:"mechanism"`
	Owner      artifact.ID `json:"owner"`
	Gap        artifact.ID `json:"gap"`
	CostBound  artifact.ID `json:"cost_bound"`
	Benefit    artifact.ID `json:"benefit"`
	Falsifier  artifact.ID `json:"falsifier"`
}

// MechanismCandidate is a non-authorizing recipe backed by direct gap, cost, benefit, and falsifier evidence.
type MechanismCandidate struct {
	id         artifact.ID
	census     artifact.ID
	provenance artifact.ID
	mechanism  artifact.ID
	owner      artifact.ID
	gap        artifact.ID
	costBound  artifact.ID
	benefit    artifact.ID
	falsifier  artifact.ID
}

// CompileMechanismCandidate derives a candidate from one open census assessment.
func CompileMechanismCandidate(
	census MechanismCensus,
	provenance, mechanism, costBound, benefit artifact.ID,
) (MechanismCandidate, error) {
	var selected *MechanismAssessment
	for _, assessment := range census.assessments {
		if assessment.Mechanism == mechanism {
			copy := assessment
			selected = &copy
			break
		}
	}
	if selected == nil || selected.GapState != MechanismGapOpen || provenance.Kind() != artifact.KindEvidence ||
		costBound.Kind() != artifact.KindEvidence || benefit.Kind() != artifact.KindEvidence ||
		provenance == selected.Gap || provenance == costBound || provenance == benefit ||
		selected.Gap == costBound || selected.Gap == benefit || costBound == benefit {
		return MechanismCandidate{}, errors.New("training program: mechanism candidate lacks independent gap, cost, benefit, or provenance evidence")
	}
	body := mechanismCandidateBody{
		Census: census.id, Provenance: provenance, Mechanism: mechanism, Owner: selected.Owner,
		Gap: selected.Gap, CostBound: costBound, Benefit: benefit, Falsifier: selected.Falsifier,
	}
	id, err := artifact.JSONID(artifact.KindRecipe, body)
	if err != nil {
		return MechanismCandidate{}, err
	}
	return MechanismCandidate{
		id: id, census: body.Census, provenance: provenance, mechanism: mechanism, owner: body.Owner,
		gap: body.Gap, costBound: costBound, benefit: benefit, falsifier: body.Falsifier,
	}, nil
}

// ID returns the candidate content identity.
func (value MechanismCandidate) ID() artifact.ID { return value.id }

// Census returns the ownership and gap census authority.
func (value MechanismCandidate) Census() artifact.ID { return value.census }

// Provenance returns the exact external source authority.
func (value MechanismCandidate) Provenance() artifact.ID { return value.provenance }

// Mechanism returns the exact research mechanism subject.
func (value MechanismCandidate) Mechanism() artifact.ID { return value.mechanism }

// Owner returns the existing implementation owner named by the census.
func (value MechanismCandidate) Owner() artifact.ID { return value.owner }

// Gap returns typed evidence for the open capability gap.
func (value MechanismCandidate) Gap() artifact.ID { return value.gap }

// CostBound returns typed evidence for the candidate's resource bound.
func (value MechanismCandidate) CostBound() artifact.ID { return value.costBound }

// Benefit returns typed evidence for the candidate's expected benefit.
func (value MechanismCandidate) Benefit() artifact.ID { return value.benefit }

// Falsifier returns the typed recipe that can reject the mechanism claim.
func (value MechanismCandidate) Falsifier() artifact.ID { return value.falsifier }

// Evidence returns the candidate's census, provenance, owner, gap, cost, benefit, and falsifier authorities.
func (value MechanismCandidate) Evidence() []artifact.ID {
	return []artifact.ID{value.census, value.provenance, value.owner, value.gap, value.costBound, value.benefit, value.falsifier}
}

// Content returns the canonical candidate recipe.
func (value MechanismCandidate) Content() (artifact.Content, error) {
	return mechanismCandidateContract.ContentJSON(value.id, value.body())
}

// Lineage binds the candidate to its mechanism and every required authority.
func (value MechanismCandidate) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.id, append([]artifact.ID{value.mechanism}, value.Evidence()...)...)
}

func (value MechanismCandidate) body() mechanismCandidateBody {
	return mechanismCandidateBody{
		Census: value.census, Provenance: value.provenance, Mechanism: value.mechanism, Owner: value.owner,
		Gap: value.gap, CostBound: value.costBound, Benefit: value.benefit, Falsifier: value.falsifier,
	}
}
