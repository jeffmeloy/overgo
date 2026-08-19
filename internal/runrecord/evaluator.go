package runrecord

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	EvaluatorPromotionVersion   uint16 = 1
	EvaluatorPromotionMediaType        = "application/vnd.overgo.evaluator-promotion+json"
	EvaluatorPromotionSchema           = "overgo/evaluator-promotion/v1"
)

// EvaluatorCase is one held-out oracle model with a KNOWN outcome and the
// candidate evaluator's score for it (lower is better, matching every loss
// surface in the repo).
type EvaluatorCase struct {
	Model     artifact.ID `json:"model"`
	Score     float64     `json:"score"`
	KnownGood bool        `json:"known_good"`
}

// EvaluatorPromotion is the typed verdict on a candidate evaluator: promoted
// only when it ranks EVERY known-good oracle model strictly better than
// every known-bad one, under an approval decided by the prior generation's
// authority. An evaluator that cannot separate known outcomes judges nothing.
type EvaluatorPromotion struct {
	Version   uint16          `json:"version"`
	Evaluator artifact.ID     `json:"evaluator"`
	Cases     []EvaluatorCase `json:"cases"`
	Approval  artifact.ID     `json:"approval"`
	Promoted  bool            `json:"promoted"`
	Reason    string          `json:"reason"`
	ID        artifact.ID     `json:"-"`
}

var evaluatorPromotionCodec = artifact.JSONDocumentCodec(
	"evaluator promotion", artifact.KindEvidence, EvaluatorPromotionMediaType, EvaluatorPromotionSchema,
	canonicalizeEvaluatorPromotion,
	func(value EvaluatorPromotion) artifact.ID { return value.ID },
	func(value *EvaluatorPromotion, id artifact.ID) { value.ID = id }, nil,
)

// PromoteEvaluator judges a candidate evaluator against the sealed oracle
// cases under prior-generation authority. The approval must be decided by
// the prior decider identity, must name the candidate evaluator as its
// subject, and must accept -- the same succession bar admission bindings
// meet. Ranking failure or authority failure is a recorded refusal.
func PromoteEvaluator(
	evaluator artifact.ID,
	cases []EvaluatorCase,
	approval recipe.Decision,
	priorAuthority artifact.ID,
) (EvaluatorPromotion, error) {
	promotion := EvaluatorPromotion{
		Version: EvaluatorPromotionVersion, Evaluator: evaluator,
		Cases: append([]EvaluatorCase(nil), cases...), Approval: approval.ID,
	}
	good, bad := 0, 0
	worstGood, bestBad := 0.0, 0.0
	for _, oracle := range cases {
		if oracle.KnownGood {
			if good == 0 || oracle.Score > worstGood {
				worstGood = oracle.Score
			}
			good++
		} else {
			if bad == 0 || oracle.Score < bestBad {
				bestBad = oracle.Score
			}
			bad++
		}
	}
	switch {
	case good == 0 || bad == 0:
		return EvaluatorPromotion{}, fmt.Errorf(
			"run record: evaluator oracle set needs known-good and known-bad models, have %d/%d", good, bad)
	case approval.Decider.Derivation != priorAuthority:
		promotion.Reason = "refusal: approval was not decided by the prior generation's authority"
	case approval.Subject != evaluator:
		promotion.Reason = "refusal: approval subject is not the candidate evaluator"
	case approval.Outcome != recipe.DecisionAccepted:
		promotion.Reason = fmt.Sprintf("refusal: approval outcome %q does not admit the evaluator", approval.Outcome)
	case worstGood >= bestBad:
		promotion.Reason = fmt.Sprintf(
			"refusal: evaluator misranks the sealed oracles: worst known-good %.6f does not beat best known-bad %.6f",
			worstGood, bestBad)
	default:
		promotion.Promoted = true
		promotion.Reason = fmt.Sprintf(
			"evaluator separates %d known-good from %d known-bad oracles (margin %.6f) under prior-generation approval",
			good, bad, bestBad-worstGood)
	}
	return evaluatorPromotionCodec.New(promotion)
}

func ParseEvaluatorPromotion(content []byte) (EvaluatorPromotion, error) {
	return evaluatorPromotionCodec.Parse(content)
}

func (p EvaluatorPromotion) Content() (artifact.Content, error) {
	return evaluatorPromotionCodec.Content(p)
}

// Batch commits the promotion with lineage to the evaluator, the approval and
// every oracle model.
func (p EvaluatorPromotion) Batch(key string) (artifact.Batch, error) {
	lineage := []artifact.Lineage{
		{Child: p.ID, Parent: p.Evaluator, Relation: artifact.RelationDependsOn},
		{Child: p.ID, Parent: p.Approval, Relation: artifact.RelationDependsOn},
	}
	for _, oracle := range p.Cases {
		lineage = append(lineage, artifact.Lineage{
			Child: p.ID, Parent: oracle.Model, Relation: artifact.RelationDependsOn,
		})
	}
	return evaluatorPromotionCodec.Batch(key, p, lineage, nil)
}

func canonicalizeEvaluatorPromotion(value *EvaluatorPromotion) error {
	if value == nil || value.Version != EvaluatorPromotionVersion {
		return errors.New("run record: invalid evaluator promotion version")
	}
	if !value.Evaluator.Valid() || !value.Approval.Valid() {
		return errors.New("run record: evaluator promotion requires evaluator and approval identities")
	}
	if len(value.Cases) < 2 {
		return errors.New("run record: evaluator promotion requires oracle cases")
	}
	seen := map[artifact.ID]bool{}
	for _, oracle := range value.Cases {
		if oracle.Model.Kind() != artifact.KindModel || seen[oracle.Model] {
			return errors.New("run record: evaluator oracle cases must be distinct models")
		}
		seen[oracle.Model] = true
	}
	if value.Reason == "" {
		return errors.New("run record: evaluator promotion requires its reason")
	}
	return nil
}
