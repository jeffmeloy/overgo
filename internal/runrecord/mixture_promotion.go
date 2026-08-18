package runrecord

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

const (
	MixturePromotionVersion   uint16 = 1
	MixturePromotionMediaType        = "application/vnd.overgo.mixture-promotion+json"
	MixturePromotionSchema           = "overgo/mixture-promotion/v1"
)

// MixturePromotion is the typed verdict on a composed objective mixture:
// promoted only when the models trained under the mixture beat the incumbent
// objective's models under the descendant metric contract. It lives beside
// the contract it consumes; the mixture document itself is trainingprogram's.
type MixturePromotion struct {
	Version   uint16         `json:"version"`
	Mixture   artifact.ID    `json:"mixture"`
	Incumbent artifact.ID    `json:"incumbent"`
	Contract  MetricContract `json:"contract"`
	Promoted  bool           `json:"promoted"`
	Reason    string         `json:"reason"`
	ID        artifact.ID    `json:"-"`
}

var mixturePromotionCodec = artifact.JSONDocumentCodec(
	"mixture promotion", artifact.KindEvidence, MixturePromotionMediaType, MixturePromotionSchema,
	canonicalizeMixturePromotion,
	func(value MixturePromotion) artifact.ID { return value.ID },
	func(value *MixturePromotion, id artifact.ID) { value.ID = id }, nil,
)

// PromoteObjectiveMixture judges a composed objective against the incumbent
// by the models each produced. The verdict is faithful to the contract:
// promotion on full separation, a measured refusal otherwise.
func PromoteObjectiveMixture(
	mixture, incumbent artifact.ID,
	contract MetricContract,
) (MixturePromotion, error) {
	if !mixture.Valid() || !incumbent.Valid() || mixture == incumbent {
		return MixturePromotion{}, errors.New("run record: mixture promotion requires distinct mixture and incumbent identities")
	}
	promotion := MixturePromotion{
		Version: MixturePromotionVersion, Mixture: mixture, Incumbent: incumbent,
		Contract: contract,
	}
	if err := ValidateDescendantImprovement(contract); err != nil {
		promotion.Reason = "refusal: " + err.Error()
	} else {
		promotion.Promoted = true
		promotion.Reason = fmt.Sprintf(
			"mixture descendants beat incumbent descendants on all %d seeds under the metric contract",
			len(contract.Comparisons))
	}
	return mixturePromotionCodec.New(promotion)
}

func ParseMixturePromotion(content []byte) (MixturePromotion, error) {
	return mixturePromotionCodec.Parse(content)
}

func (p MixturePromotion) Content() (artifact.Content, error) {
	return mixturePromotionCodec.Content(p)
}

// Batch commits the promotion with lineage to the mixture and the incumbent.
func (p MixturePromotion) Batch(key string) (artifact.Batch, error) {
	return mixturePromotionCodec.Batch(key, p, []artifact.Lineage{
		{Child: p.ID, Parent: p.Mixture, Relation: artifact.RelationDependsOn},
		{Child: p.ID, Parent: p.Incumbent, Relation: artifact.RelationDependsOn},
	}, nil)
}

func canonicalizeMixturePromotion(value *MixturePromotion) error {
	if value == nil || value.Version != MixturePromotionVersion {
		return errors.New("run record: invalid mixture promotion version")
	}
	if !value.Mixture.Valid() || !value.Incumbent.Valid() || value.Mixture == value.Incumbent {
		return errors.New("run record: promotion requires distinct mixture and incumbent")
	}
	if value.Reason == "" {
		return errors.New("run record: mixture promotion requires its reason")
	}
	return nil
}
