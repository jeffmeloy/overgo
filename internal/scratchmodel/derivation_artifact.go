package scratchmodel

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	DerivationProfileDocumentVersion   uint16 = 1
	DerivationProfileDocumentMediaType        = "application/vnd.overgo.derivation-profile-document+json"
	DerivationProfileDocumentSchema           = "overgo/derivation-profile-document/v1"
)

// DerivationProfileDocument makes the corpus-to-topology derivation profile a
// content-addressed artifact: the policy that turns corpora into models is
// itself proposable, comparable and promotable -- and it is judged solely on
// the models it produces, never on its own prose.
type DerivationProfileDocument struct {
	Version uint16            `json:"version"`
	Profile DerivationProfile `json:"profile"`
	ID      artifact.ID       `json:"-"`
}

var derivationProfileDocumentCodec = artifact.JSONDocumentCodec(
	"derivation profile document", artifact.KindProfile, DerivationProfileDocumentMediaType, DerivationProfileDocumentSchema,
	canonicalizeDerivationProfile,
	func(value DerivationProfileDocument) artifact.ID { return value.ID },
	func(value *DerivationProfileDocument, id artifact.ID) { value.ID = id }, nil,
)

// NewDerivationProfileDocument identifies one derivation policy.
func NewDerivationProfileDocument(profile DerivationProfile) (DerivationProfileDocument, error) {
	profile, err := NewDerivationProfile(profile)
	if err != nil {
		return DerivationProfileDocument{}, err
	}
	return derivationProfileDocumentCodec.New(DerivationProfileDocument{
		Version: DerivationProfileDocumentVersion, Profile: profile,
	})
}

func ParseDerivationProfileDocument(content []byte) (DerivationProfileDocument, error) {
	return derivationProfileDocumentCodec.Parse(content)
}

func (d DerivationProfileDocument) Content() (artifact.Content, error) {
	return derivationProfileDocumentCodec.Content(d)
}

func canonicalizeDerivationProfile(value *DerivationProfileDocument) error {
	if value == nil || value.Version != DerivationProfileDocumentVersion {
		return errors.New("scratch model: invalid derivation profile version")
	}
	return value.Profile.validate()
}

// DerivationPromotion is the typed verdict on a candidate derivation profile:
// promoted only when the models it produced beat the incumbent's models under
// the full descendant metric contract, with lineage to both profile
// documents and the judging evidence.
type DerivationPromotion struct {
	Version   uint16                   `json:"version"`
	Candidate artifact.ID              `json:"candidate"`
	Incumbent artifact.ID              `json:"incumbent"`
	Contract  runrecord.MetricContract `json:"contract"`
	Promoted  bool                     `json:"promoted"`
	Reason    string                   `json:"reason"`
	ID        artifact.ID              `json:"-"`
}

var derivationPromotionCodec = artifact.JSONDocumentCodec(
	"derivation promotion", artifact.KindEvidence, DerivationPromotionMediaType, DerivationPromotionSchema,
	canonicalizeDerivationPromotion,
	func(value DerivationPromotion) artifact.ID { return value.ID },
	func(value *DerivationPromotion, id artifact.ID) { value.ID = id }, nil,
)

const (
	DerivationPromotionMediaType = "application/vnd.overgo.derivation-promotion+json"
	DerivationPromotionSchema    = "overgo/derivation-promotion/v1"
)

// PromoteDerivationProfile judges a candidate profile against the incumbent
// by the models each produced: the metric contract's comparisons carry, per
// seed, the incumbent's model quality as Parent and the candidate's as
// Child. Promotion requires the full contract; anything else is a recorded
// refusal with the measured reason.
func PromoteDerivationProfile(
	candidate, incumbent DerivationProfileDocument,
	contract runrecord.MetricContract,
) (DerivationPromotion, error) {
	if err := candidate.ValidateIdentity(); err != nil {
		return DerivationPromotion{}, fmt.Errorf("scratch model: candidate profile: %w", err)
	}
	if err := incumbent.ValidateIdentity(); err != nil {
		return DerivationPromotion{}, fmt.Errorf("scratch model: incumbent profile: %w", err)
	}
	if candidate.ID == incumbent.ID {
		return DerivationPromotion{}, errors.New("scratch model: a profile cannot challenge itself")
	}
	promotion := DerivationPromotion{
		Version: DerivationPromotionVersion, Candidate: candidate.ID, Incumbent: incumbent.ID,
		Contract: contract,
	}
	if err := runrecord.ValidateDescendantImprovement(contract); err != nil {
		promotion.Reason = "refusal: " + err.Error()
	} else {
		promotion.Promoted = true
		promotion.Reason = fmt.Sprintf(
			"candidate models beat incumbent models on all %d seeds under the descendant metric contract",
			len(contract.Comparisons))
	}
	return derivationPromotionCodec.New(promotion)
}

const DerivationPromotionVersion uint16 = 1

func ParseDerivationPromotion(content []byte) (DerivationPromotion, error) {
	return derivationPromotionCodec.Parse(content)
}

func (p DerivationPromotion) Content() (artifact.Content, error) {
	return derivationPromotionCodec.Content(p)
}

// Batch commits the promotion with lineage to both profile documents.
func (p DerivationPromotion) Batch(key string) (artifact.Batch, error) {
	return derivationPromotionCodec.Batch(key, p, []artifact.Lineage{
		{Child: p.ID, Parent: p.Candidate, Relation: artifact.RelationDependsOn},
		{Child: p.ID, Parent: p.Incumbent, Relation: artifact.RelationDependsOn},
	}, nil)
}

func (d DerivationProfileDocument) ValidateIdentity() error {
	return derivationProfileDocumentCodec.ValidateIdentity(d)
}

func canonicalizeDerivationPromotion(value *DerivationPromotion) error {
	if value == nil || value.Version != DerivationPromotionVersion {
		return errors.New("scratch model: invalid derivation promotion version")
	}
	if !value.Candidate.Valid() || !value.Incumbent.Valid() || value.Candidate == value.Incumbent {
		return errors.New("scratch model: promotion requires distinct candidate and incumbent profiles")
	}
	if value.Reason == "" {
		return errors.New("scratch model: promotion requires its reason")
	}
	return nil
}
