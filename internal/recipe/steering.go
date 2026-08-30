package recipe

import (
	"errors"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	// SteeringProposalMediaType identifies steering proposals.
	SteeringProposalMediaType = "application/vnd.overgo.steering-proposal+json"
	// SteeringProposalSchema identifies the steering proposal schema.
	SteeringProposalSchema = "overgo/steering-proposal/v1"
)

// SteeringPrediction is the falsifiable benefit, cost, and uncertainty claim
// behind a proposal, measured in a named metric and unit.
type SteeringPrediction struct {
	Metric      string  `json:"metric"`
	Benefit     float64 `json:"benefit"`
	Cost        uint64  `json:"cost"`
	Unit        string  `json:"unit"`
	Uncertainty float64 `json:"uncertainty"`
}

// Validate checks the shared quantified benefit, cost, and uncertainty claim.
func (prediction SteeringPrediction) Validate() error {
	if !boundedStatement(prediction.Metric) || !boundedStatement(prediction.Unit) ||
		!checked.Finite64(prediction.Benefit) || prediction.Benefit <= 0 || prediction.Cost == 0 ||
		!checked.Finite64(prediction.Uncertainty) || prediction.Uncertainty < 0 ||
		prediction.Uncertainty > float64(artifact.InitialDocumentVersion) {
		return errors.New("recipe: invalid steering prediction")
	}
	return nil
}

// SteeringPlanRow is one verifier-bound plan row the proposal would admit.
type SteeringPlanRow struct {
	Item         string      `json:"item"`
	Step         string      `json:"step"`
	Title        string      `json:"title"`
	Verifier     artifact.ID `json:"verifier"`
	Capabilities []string    `json:"capabilities,omitempty"`
}

// SteeringProposal is an immutable request to extend the plan: a goal, a
// measurable prediction, affected authorities, measurements, and the
// verifier-bound rows admission would add.
type SteeringProposal struct {
	Version             uint16             `json:"version"`
	ID                  artifact.ID        `json:"-"`
	Goal                string             `json:"goal"`
	Prediction          SteeringPrediction `json:"prediction"`
	Evaluation          artifact.ID        `json:"evaluation"`
	AffectedAuthorities []artifact.ID      `json:"affected_authorities"`
	Measurements        []artifact.ID      `json:"measurements"`
	Rows                []SteeringPlanRow  `json:"rows"`
}

var steeringProposalCodec = artifact.JSONDocumentCodec(
	"steering proposal", artifact.KindRecipe, SteeringProposalMediaType, SteeringProposalSchema,
	canonicalizeSteeringProposal, func(v SteeringProposal) artifact.ID { return v.ID }, func(v *SteeringProposal, id artifact.ID) { v.ID = id },
	func(v SteeringProposal) SteeringProposal {
		v.AffectedAuthorities = slices.Clone(v.AffectedAuthorities)
		v.Measurements = slices.Clone(v.Measurements)
		v.Rows = slices.Clone(v.Rows)
		for i := range v.Rows {
			v.Rows[i].Capabilities = slices.Clone(v.Rows[i].Capabilities)
		}
		return v
	},
)

// NewSteeringProposal canonicalizes and identifies one immutable steering proposal.
func NewSteeringProposal(v SteeringProposal) (SteeringProposal, error) {
	v.Version, v.ID = artifact.InitialDocumentVersion, artifact.ID{}
	return steeringProposalCodec.New(v)
}

// Content returns the canonical committed bytes of the proposal.
func (v SteeringProposal) Content() (artifact.Content, error) {
	return steeringProposalCodec.Content(v)
}

// ValidateIdentity checks the proposal's canonical form and content-addressed identity.
func (v SteeringProposal) ValidateIdentity() error { return steeringProposalCodec.ValidateIdentity(v) }

// Lineage links the proposal to its evaluation, affected authorities,
// measurements, and row verifiers.
func (v SteeringProposal) Lineage() []artifact.Lineage {
	parents := []artifact.ID{v.Evaluation}
	parents = append(parents, v.AffectedAuthorities...)
	parents = append(parents, v.Measurements...)
	for _, row := range v.Rows {
		parents = append(parents, row.Verifier)
	}
	return artifact.DependencyLineage(v.ID, parents...)
}

func canonicalizeSteeringProposal(v *SteeringProposal) error {
	if v == nil || v.Version != artifact.InitialDocumentVersion || !boundedStatement(v.Goal) ||
		v.Evaluation.Kind() != artifact.KindProfile {
		return errors.New("recipe: invalid steering proposal")
	}
	if err := v.Prediction.Validate(); err != nil {
		return err
	}
	if !checked.Nonempty(v.AffectedAuthorities) {
		return errors.New("recipe: steering proposal has no affected authority")
	}
	if !checked.Nonempty(v.Measurements) {
		return errors.New("recipe: steering proposal has no measurement")
	}
	if !checked.Nonempty(v.Rows) {
		return errors.New("recipe: steering proposal has no plan row")
	}
	var err error
	if v.AffectedAuthorities, err = canonicalSteeringIDs(v.AffectedAuthorities); err != nil {
		return err
	}
	if v.Measurements, err = canonicalSteeringIDs(v.Measurements); err != nil {
		return err
	}
	seenRows := map[string]bool{}
	rows := make([]SteeringPlanRow, 0, len(v.Rows))
	for _, source := range v.Rows {
		row := source
		if !steeringID(row.Item) || !steeringID(row.Step) || !boundedStatement(row.Title) || row.Verifier.Kind() != artifact.KindRecipe {
			return errors.New("recipe: invalid steering plan row")
		}
		slices.Sort(row.Capabilities)
		row.Capabilities = slices.Compact(row.Capabilities)
		for _, capability := range row.Capabilities {
			if !steeringID(capability) {
				return errors.New("recipe: invalid steering capability")
			}
		}
		key := row.Item + "\x00" + row.Step
		if seenRows[key] {
			return errors.New("recipe: duplicate steering row")
		}
		seenRows[key] = true
		rows = append(rows, row)
	}
	v.Rows = rows
	slices.SortFunc(v.Rows, func(left, right SteeringPlanRow) int {
		if order := strings.Compare(left.Item, right.Item); order != 0 {
			return order
		}
		return strings.Compare(left.Step, right.Step)
	})
	return nil
}
func canonicalSteeringIDs(values []artifact.ID) ([]artifact.ID, error) {
	for _, id := range values {
		if !id.Valid() {
			return nil, errors.New("recipe: invalid steering reference")
		}
	}
	result := slices.Clone(values)
	slices.SortFunc(result, artifact.CompareID)
	return slices.Compact(result), nil
}
func steeringID(value string) bool {
	return value != "" && !strings.ContainsAny(value, "/\\ \t\x00\r\n") && boundedStatement(value)
}
