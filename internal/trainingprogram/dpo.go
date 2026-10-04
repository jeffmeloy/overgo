package trainingprogram

import (
	"errors"
	"fmt"
	"math"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
)

// PreferenceScores are sequence log probabilities for one preference pair.
type PreferenceScores struct {
	PolicyChosen      sequencescore.Score
	PolicyRejected    sequencescore.Score
	ReferenceChosen   sequencescore.Score
	ReferenceRejected sequencescore.Score
}

// DPOResult carries loss and policy score gradients. Reference scores are frozen.
type DPOResult struct {
	Loss             float64
	ChosenGradient   float64
	RejectedGradient float64
}

type DPOObservation struct {
	Step              uint64  `json:"step"`
	Loss              float64 `json:"loss"`
	PolicyChosen      float64 `json:"policy_chosen"`
	PolicyRejected    float64 `json:"policy_rejected"`
	ReferenceChosen   float64 `json:"reference_chosen"`
	ReferenceRejected float64 `json:"reference_rejected"`
	PolicyMargin      float64 `json:"policy_margin"`
	ReferenceMargin   float64 `json:"reference_margin"`
	RelativeMargin    float64 `json:"relative_margin"`
	ChosenTokens      uint64  `json:"chosen_tokens"`
	RejectedTokens    uint64  `json:"rejected_tokens"`
	LearningRate      float64 `json:"learning_rate"`
	GradientL2        float64 `json:"gradient_l2"`
	UpdateL2          float64 `json:"update_l2"`
}

func (observation DPOObservation) Valid() bool {
	values := [...]float64{
		observation.Loss, observation.PolicyChosen, observation.PolicyRejected,
		observation.ReferenceChosen, observation.ReferenceRejected,
		observation.PolicyMargin, observation.ReferenceMargin, observation.RelativeMargin,
		observation.LearningRate, observation.GradientL2, observation.UpdateL2,
	}
	if observation.Step == 0 || observation.ChosenTokens == 0 || observation.RejectedTokens == 0 ||
		observation.Loss < 0 || observation.LearningRate <= 0 || observation.GradientL2 < 0 || observation.UpdateL2 < 0 {
		return false
	}
	for _, value := range values {
		if !finite(value) {
			return false
		}
	}
	return observation.PolicyMargin == observation.PolicyChosen-observation.PolicyRejected &&
		observation.ReferenceMargin == observation.ReferenceChosen-observation.ReferenceRejected &&
		observation.RelativeMargin == observation.PolicyMargin-observation.ReferenceMargin
}

func DPOLoss(scores PreferenceScores, scale float64) (DPOResult, error) {
	if scale <= 0 || !finite(scale) ||
		!scores.PolicyChosen.Valid() || !scores.PolicyRejected.Valid() ||
		!scores.ReferenceChosen.Valid() || !scores.ReferenceRejected.Valid() {
		return DPOResult{}, errors.New("training program: invalid DPO scores or scale")
	}
	policy, err := sequencescore.Select(
		[]sequencescore.Score{scores.PolicyChosen, scores.PolicyRejected}, sequencescore.NormalizationSum,
	)
	if err != nil {
		return DPOResult{}, err
	}
	reference, err := sequencescore.Select(
		[]sequencescore.Score{scores.ReferenceChosen, scores.ReferenceRejected}, sequencescore.NormalizationSum,
	)
	if err != nil {
		return DPOResult{}, err
	}
	margin := scale * ((policy.Values[0] - policy.Values[1]) - (reference.Values[0] - reference.Values[1]))
	loss := softplus(-margin)
	down := sigmoid(-margin)
	return DPOResult{
		Loss: loss, ChosenGradient: -scale * down, RejectedGradient: scale * down,
	}, nil
}

// PreferenceMode names the loss the DPO objective takes over a chosen and
// rejected pair, or, for KTO, over unpaired completions. The zero mode is
// DPO's own sigmoid loss against the frozen reference (Rafailov et al.
// 2023); every pair mode is judged on held-out pairs by that same loss, and
// KTO on held-out unpaired completions by its own.
type PreferenceMode string

const (
	// PreferenceSimPO is SimPO's reference-free, length-normalized loss.
	PreferenceSimPO PreferenceMode = "simpo"
	// PreferenceORPO is ORPO's reference-free SFT plus odds-ratio loss.
	PreferenceORPO PreferenceMode = "orpo"
	// PreferenceKTO is KTO's loss over unpaired completions, each labelled
	// desirable or undesirable, against the frozen reference.
	PreferenceKTO PreferenceMode = "kto"
)

// preferenceModes are the declared modes, DPO's own first.
var preferenceModes = []PreferenceMode{"", PreferenceSimPO, PreferenceORPO, PreferenceKTO}

// Loss is the pair's loss in the policy's mode, with its gradients with
// respect to the policy's summed chosen and rejected log probabilities. The
// reference model's identity is the program's, not the loss's.
func (policy PreferencePolicy) Loss(scores PreferenceScores) (DPOResult, error) {
	if err := policy.validLoss(); err != nil {
		return DPOResult{}, err
	}
	if policy.Mode == PreferenceKTO {
		return DPOResult{}, errors.New("training program: KTO scores unpaired completions, not pairs")
	}
	if policy.Mode == "" {
		return DPOLoss(scores, policy.Scale)
	}
	// The reference-free modes read each completion's mean token log
	// probability; a slope is the loss's gradient with respect to one.
	mean, err := sequencescore.Select(
		[]sequencescore.Score{scores.PolicyChosen, scores.PolicyRejected}, sequencescore.NormalizationMean,
	)
	if err != nil {
		return DPOResult{}, err
	}
	chosen, rejected := mean.Values[0], mean.Values[1]
	var loss, chosenSlope, rejectedSlope float64
	switch policy.Mode {
	case PreferenceSimPO:
		// SimPO (Meng, Xia and Chen 2024): -log σ(β·(chosen - rejected) - γ).
		reward := policy.Scale*(chosen-rejected) - policy.Margin
		down := policy.Scale * sigmoid(-reward)
		loss, chosenSlope, rejectedSlope = softplus(-reward), -down, down
	case PreferenceORPO:
		// ORPO (Hong, Lee and Thorne 2024): the chosen completion's NLL,
		// -chosen, plus λ·(-log σ(log(odds(yw|x)/odds(yl|x)))), where
		// odds = P/(1-P) and log(1-P) = log(-expm1(log P)) is finite only
		// below probability one; d log odds/d log P = -1/expm1(log P).
		chosenRest, rejectedRest := math.Log(-math.Expm1(chosen)), math.Log(-math.Expm1(rejected))
		if !finite(chosenRest) || !finite(rejectedRest) {
			return DPOResult{}, errors.New("training program: ORPO needs completion likelihoods below one")
		}
		logOddsRatio := (chosen - chosenRest) - (rejected - rejectedRest)
		down := policy.Scale * sigmoid(-logOddsRatio)
		loss, chosenSlope, rejectedSlope = policy.Scale*softplus(-logOddsRatio)-chosen, down/math.Expm1(chosen)-1, -down/math.Expm1(rejected)
	}
	return DPOResult{
		Loss: loss, ChosenGradient: chosenSlope / float64(scores.PolicyChosen.Tokens),
		RejectedGradient: rejectedSlope / float64(scores.PolicyRejected.Tokens),
	}, nil
}

// UnpairedScores are one unpaired completion's label and its sequence log
// probabilities under the policy and the frozen reference: after its own
// prompt, and, for KTO's KL estimate, another example's completion after
// this example's prompt.
type UnpairedScores struct {
	Desirable                             bool
	Policy, Reference                     sequencescore.Score
	MismatchedPolicy, MismatchedReference sequencescore.Score
}

// KTOSummary is KTO's loss over a microbatch, its KL estimate z0, and each
// class's count and mean implied reward r, zero for a class it lacks.
type KTOSummary struct {
	Loss, KL                           float64
	Desirable, Undesirable             uint64
	DesirableReward, UndesirableReward float64
}

// KTOResult is a microbatch's summary and the loss's gradient with respect
// to each example's summed policy log probability.
type KTOResult struct {
	KTOSummary
	Gradients []float64
}

// KTOObservation records one KTO update: its microbatch's summary and the
// update's step, learning rate and norms.
type KTOObservation struct {
	KTOSummary
	Step                               uint64
	LearningRate, GradientL2, UpdateL2 float64
}

// UnpairedLoss is KTO's loss (Ethayarajh et al. 2024) over a microbatch of
// unpaired completions, the mean of λ_y − v(x, y), where the implied reward
// is r = log πθ(y|x) − log πref(y|x), a desirable completion's value is
// λ_D·σ(β(r − z0)) and an undesirable one's λ_U·σ(β(z0 − r)). The KL
// estimate z0 = max(0, mean of r over the microbatch's mismatched pairs)
// is held fixed: no gradient flows through it, so the mismatched scores
// carry none. This is the paper's estimate, not TRL's kto_pair, whose
// matched-pair term carries gradient.
func (policy PreferencePolicy) UnpairedLoss(examples []UnpairedScores) (KTOResult, error) {
	if err := policy.validLoss(); err != nil {
		return KTOResult{}, err
	}
	// A lone completion has no other to mismatch its prompt with.
	if policy.Mode != PreferenceKTO || len(examples) <= 1 {
		return KTOResult{}, errors.New("training program: KTO needs its mode and two or more unpaired completions")
	}
	var mismatched float64
	for _, example := range examples {
		if !example.Policy.Valid() || !example.Reference.Valid() || !example.MismatchedPolicy.Valid() || !example.MismatchedReference.Valid() {
			return KTOResult{}, errors.New("training program: invalid KTO scores")
		}
		mismatched += example.MismatchedPolicy.LogProbability - example.MismatchedReference.LogProbability
	}
	count := float64(len(examples))
	result := KTOResult{KTOSummary: KTOSummary{KL: max(0, mismatched/count)}, Gradients: make([]float64, len(examples))}
	for index, example := range examples {
		reward := example.Policy.LogProbability - example.Reference.LogProbability
		// λ_y − v is λ_y·σ(−z), z the completion's signed margin: β(r − z0)
		// when desirable, β(z0 − r) when not.
		weight, sign := policy.UndesirableWeight, float64(-1)
		if example.Desirable {
			weight, sign = policy.DesirableWeight, 1
			result.Desirable++
			result.DesirableReward += reward
		} else {
			result.Undesirable++
			result.UndesirableReward += reward
		}
		margin := sign * policy.Scale * (reward - result.KL)
		result.Loss += weight * sigmoid(-margin) / count
		result.Gradients[index] = -sign * policy.Scale * weight * sigmoid(margin) * sigmoid(-margin) / count
	}
	if result.Desirable != 0 {
		result.DesirableReward /= float64(result.Desirable)
	}
	if result.Undesirable != 0 {
		result.UndesirableReward /= float64(result.Undesirable)
	}
	return result, nil
}

// KTOPolicy is the KTO preference policy against reference at scale β,
// with the class weights its desirable and undesirable completion counts
// derive: λ_U = 1 and λ_D = n_U/n_D, so λ_D·n_D/(λ_U·n_U) = 1, the low end
// of the range [1, 4/3] the paper advises, holding λ_U at one as its
// example does. Both classes must be present.
func KTOPolicy(reference artifact.ID, scale float64, desirable, undesirable int) (PreferencePolicy, error) {
	if desirable <= 0 || undesirable <= 0 {
		return PreferencePolicy{}, fmt.Errorf("training program: KTO weighs desirable against undesirable completions; %d and %d given", desirable, undesirable)
	}
	policy := PreferencePolicy{
		Reference: reference, Scale: scale, Mode: PreferenceKTO,
		DesirableWeight: float64(undesirable) / float64(desirable), UndesirableWeight: 1,
	}
	return policy, policy.Validate()
}

func softplus(value float64) float64 {
	if value > 0 {
		return value + math.Log1p(math.Exp(-value))
	}
	return math.Log1p(math.Exp(value))
}

func sigmoid(value float64) float64 {
	if value >= 0 {
		inverse := math.Exp(-value)
		return 1 / (1 + inverse)
	}
	exponential := math.Exp(value)
	return exponential / (1 + exponential)
}
