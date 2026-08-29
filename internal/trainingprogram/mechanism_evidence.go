package trainingprogram

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

const (
	mechanismGapEvidenceMediaType       = "application/vnd.overgo.mechanism-gap-evidence+json"
	mechanismGapEvidenceSchema          = "overgo/mechanism-gap-evidence/v1"
	mechanismFalsifierMediaType         = "application/vnd.overgo.mechanism-falsifier+json"
	mechanismFalsifierSchema            = "overgo/mechanism-falsifier/v1"
	mechanismCostBoundEvidenceMediaType = "application/vnd.overgo.mechanism-cost-bound-evidence+json"
	mechanismCostBoundEvidenceSchema    = "overgo/mechanism-cost-bound-evidence/v1"
	mechanismBenefitEvidenceMediaType   = "application/vnd.overgo.mechanism-benefit-evidence+json"
	mechanismBenefitEvidenceSchema      = "overgo/mechanism-benefit-evidence/v1"
)

// MechanismEvidenceRole names the semantic claim made by one typed mechanism reference.
type MechanismEvidenceRole string

const (
	// MechanismEvidenceGap identifies direct evidence that an implementation gap exists.
	MechanismEvidenceGap MechanismEvidenceRole = "gap"
	// MechanismEvidenceFalsifier identifies the recipe that can refute the mechanism claim.
	MechanismEvidenceFalsifier MechanismEvidenceRole = "falsifier"
	// MechanismEvidenceCostBound identifies direct evidence for the mechanism's bounded cost.
	MechanismEvidenceCostBound MechanismEvidenceRole = "cost-bound"
	// MechanismEvidenceBenefit identifies direct evidence for the mechanism's expected benefit.
	MechanismEvidenceBenefit MechanismEvidenceRole = "benefit"
)

// MechanismEvidence binds one typed claim to the exact mechanism it concerns
// and to the direct source that supports or can falsify that claim. It carries
// no activation or promotion authority.
type MechanismEvidence struct {
	Version   uint16                `json:"version"`
	Role      MechanismEvidenceRole `json:"role"`
	Mechanism artifact.ID           `json:"mechanism"`
	Source    artifact.ID           `json:"source"`
	ID        artifact.ID           `json:"-"`
}

var (
	mechanismGapEvidenceCodec = newMechanismEvidenceCodec(
		"mechanism gap evidence", artifact.KindEvidence,
		mechanismGapEvidenceMediaType, mechanismGapEvidenceSchema, MechanismEvidenceGap,
	)
	mechanismFalsifierCodec = newMechanismEvidenceCodec(
		"mechanism falsifier", artifact.KindRecipe,
		mechanismFalsifierMediaType, mechanismFalsifierSchema, MechanismEvidenceFalsifier,
	)
	mechanismCostBoundEvidenceCodec = newMechanismEvidenceCodec(
		"mechanism cost-bound evidence", artifact.KindEvidence,
		mechanismCostBoundEvidenceMediaType, mechanismCostBoundEvidenceSchema, MechanismEvidenceCostBound,
	)
	mechanismBenefitEvidenceCodec = newMechanismEvidenceCodec(
		"mechanism benefit evidence", artifact.KindEvidence,
		mechanismBenefitEvidenceMediaType, mechanismBenefitEvidenceSchema, MechanismEvidenceBenefit,
	)
)

// NewMechanismEvidence identifies one role-specific mechanism claim.
func NewMechanismEvidence(
	role MechanismEvidenceRole,
	mechanism, source artifact.ID,
) (MechanismEvidence, error) {
	codec, ok := mechanismEvidenceCodecFor(role)
	if !ok {
		return MechanismEvidence{}, errors.New("training program: unknown mechanism evidence role")
	}
	return codec.New(MechanismEvidence{
		Version: artifact.InitialDocumentVersion, Role: role, Mechanism: mechanism, Source: source,
	})
}

// RequireMechanismEvidence loads the exact role-specific document through its
// owning contract. A same-kind document for another evidence role is refused.
func RequireMechanismEvidence(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	role MechanismEvidenceRole,
) (MechanismEvidence, error) {
	codec, ok := mechanismEvidenceCodecFor(role)
	if !ok {
		return MechanismEvidence{}, errors.New("training program: unknown mechanism evidence role")
	}
	return codec.Require(ctx, reader, id)
}

// Content returns the canonical role-specific document.
func (value MechanismEvidence) Content() (artifact.Content, error) {
	codec, ok := mechanismEvidenceCodecFor(value.Role)
	if !ok {
		return artifact.Content{}, errors.New("training program: unknown mechanism evidence role")
	}
	return codec.Content(value)
}

// Lineage binds the claim to its exact mechanism subject and direct source.
func (value MechanismEvidence) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Mechanism, value.Source)
}

func newMechanismEvidenceCodec(
	name string,
	kind artifact.Kind,
	mediaType, schema string,
	role MechanismEvidenceRole,
) artifact.DocumentCodec[MechanismEvidence] {
	return artifact.JSONDocumentCodec(
		name, kind, mediaType, schema,
		func(value *MechanismEvidence) error { return canonicalizeMechanismEvidence(value, role) },
		func(value MechanismEvidence) artifact.ID { return value.ID },
		func(value *MechanismEvidence, id artifact.ID) { value.ID = id }, nil,
	)
}

func mechanismEvidenceCodecFor(
	role MechanismEvidenceRole,
) (artifact.DocumentCodec[MechanismEvidence], bool) {
	switch role {
	case MechanismEvidenceGap:
		return mechanismGapEvidenceCodec, true
	case MechanismEvidenceFalsifier:
		return mechanismFalsifierCodec, true
	case MechanismEvidenceCostBound:
		return mechanismCostBoundEvidenceCodec, true
	case MechanismEvidenceBenefit:
		return mechanismBenefitEvidenceCodec, true
	default:
		return artifact.DocumentCodec[MechanismEvidence]{}, false
	}
}

func canonicalizeMechanismEvidence(value *MechanismEvidence, role MechanismEvidenceRole) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Role != role ||
		value.Mechanism.Kind() != artifact.KindRecipe || value.Source == value.Mechanism {
		return errors.New("training program: invalid mechanism evidence")
	}
	wantSourceKind := artifact.KindEvidence
	if role == MechanismEvidenceFalsifier {
		wantSourceKind = artifact.KindRecipe
	}
	if value.Source.Kind() != wantSourceKind {
		return errors.New("training program: invalid mechanism evidence source")
	}
	return nil
}
