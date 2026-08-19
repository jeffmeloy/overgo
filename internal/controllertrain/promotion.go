package controllertrain

import (
	"errors"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

const (
	DecisionVersion   uint16 = 1
	DecisionMediaType        = "application/vnd.overgo.controller-promotion+json"
	DecisionSchema           = "overgo/controller-promotion/v1"
)

type DecisionState string

const (
	DecisionPromote DecisionState = "promote"
	DecisionRefuse  DecisionState = "refuse"
)

// SuiteMetrics: common held-out controller suite.
type SuiteMetrics struct {
	Loss             float64 `json:"loss"`
	ActionAccuracy   float64 `json:"action_accuracy"`
	ModalityAccuracy float64 `json:"modality_accuracy"`
	ValidActionRate  float64 `json:"valid_action_rate"`
}

// SeedEvidence: one independent scratch initialization.
type SeedEvidence struct {
	Seed       int64        `json:"seed"`
	Model      artifact.ID  `json:"model"`
	Run        artifact.ID  `json:"run"`
	Evaluation artifact.ID  `json:"evaluation"`
	Initial    SuiteMetrics `json:"initial"`
	Final      SuiteMetrics `json:"final"`
	Cost       ResourceCost `json:"cost"`
}

type ResourceCost struct {
	WallNS          uint64 `json:"wall_ns"`
	PeakDeviceBytes uint64 `json:"peak_device_bytes,omitempty"`
	PeakHostBytes   uint64 `json:"peak_host_bytes,omitempty"`
}

// Challenger: same-suite result or explicit eligibility refusal.
type Challenger struct {
	Name       string        `json:"name"`
	Pretrained bool          `json:"pretrained"`
	Model      artifact.ID   `json:"model"`
	Evaluation *artifact.ID  `json:"evaluation,omitempty"`
	Eligible   bool          `json:"eligible"`
	Refusal    string        `json:"refusal,omitempty"`
	Metrics    *SuiteMetrics `json:"metrics,omitempty"`
	Evaluator  artifact.ID   `json:"evaluator,omitzero"`
	Split      artifact.ID   `json:"split,omitzero"`
}

type DecisionSpec struct {
	Dataset         artifact.ID
	Split           artifact.ID
	CurrentChampion artifact.ID
	Candidate       artifact.ID
	Seeds           []SeedEvidence
	Challengers     []Challenger
	Evaluator       artifact.ID
	ObservedNoise   SuiteMetrics
	Stochastic      bool
	Budget          runrecord.Budget
	Charges         []runrecord.BudgetCharge
}

// Decision: external champion/challenger authority.
type Decision struct {
	Version         uint16         `json:"version"`
	State           DecisionState  `json:"state"`
	Dataset         artifact.ID    `json:"dataset"`
	Split           artifact.ID    `json:"split"`
	CurrentChampion artifact.ID    `json:"current_champion"`
	Candidate       artifact.ID    `json:"candidate"`
	Rollback        artifact.ID    `json:"rollback"`
	Seeds           []SeedEvidence `json:"seeds"`
	Challengers     []Challenger   `json:"challengers"`
	Evaluator       artifact.ID    `json:"evaluator"`
	ObservedNoise   SuiteMetrics   `json:"observed_noise"`
	Stochastic      bool           `json:"stochastic"`
	Budget          artifact.ID    `json:"budget"`
	Charges         []artifact.ID  `json:"charges"`
	BudgetRemaining uint64         `json:"budget_remaining"`
	ID              artifact.ID    `json:"-"`
}

var decisionCodec = artifact.JSONDocumentCodec(
	"controller promotion", artifact.KindEvidence, DecisionMediaType, DecisionSchema,
	canonicalizeDecision,
	func(value Decision) artifact.ID { return value.ID },
	func(value *Decision, id artifact.ID) { value.ID = id },
	cloneDecision,
)

func Decide(spec DecisionSpec) (Decision, error) {
	if spec.Budget.Split != spec.Split || spec.Evaluator.Kind() != artifact.KindEvidence {
		return Decision{}, errors.New("controller training: evaluation authority differs")
	}
	remaining, err := runrecord.BudgetBalance(spec.Budget, spec.Charges)
	if err != nil || len(spec.Charges) == 0 {
		return Decision{}, errors.Join(err, errors.New("controller training: held-out query budget is absent"))
	}
	chargeIDs := make([]artifact.ID, len(spec.Charges))
	for index, charge := range spec.Charges {
		if charge.Consumer != spec.Evaluator {
			return Decision{}, errors.New("controller training: query charge evaluator differs")
		}
		chargeIDs[index] = charge.ID
	}
	decision := Decision{
		Version: DecisionVersion, State: DecisionPromote,
		Dataset: spec.Dataset, Split: spec.Split,
		CurrentChampion: spec.CurrentChampion, Candidate: spec.Candidate,
		Rollback: spec.CurrentChampion, Seeds: slices.Clone(spec.Seeds),
		Challengers: cloneChallengers(spec.Challengers),
		Evaluator:   spec.Evaluator, ObservedNoise: spec.ObservedNoise, Stochastic: spec.Stochastic,
		Budget: spec.Budget.ID, Charges: chargeIDs, BudgetRemaining: remaining,
	}
	if !promotionProved(decision) {
		decision.State = DecisionRefuse
	}
	return decisionCodec.New(decision)
}

func ParseDecision(data []byte) (Decision, error)     { return decisionCodec.Parse(data) }
func (d Decision) ValidateIdentity() error            { return decisionCodec.ValidateIdentity(d) }
func (d Decision) Content() (artifact.Content, error) { return decisionCodec.Content(d) }

func (d Decision) Lineage() []artifact.Lineage {
	parents := []artifact.ID{d.Dataset, d.Split, d.CurrentChampion, d.Candidate, d.Evaluator, d.Budget}
	parents = append(parents, d.Charges...)
	for _, seed := range d.Seeds {
		parents = append(parents, seed.Model, seed.Run, seed.Evaluation)
	}
	for _, challenger := range d.Challengers {
		parents = append(parents, challenger.Model)
		if challenger.Evaluation != nil {
			parents = append(parents, *challenger.Evaluation)
		}
	}
	seen := map[artifact.ID]bool{}
	result := make([]artifact.Lineage, 0, len(parents))
	for _, parent := range parents {
		if parent.Valid() && parent != d.ID && !seen[parent] {
			seen[parent] = true
			result = append(result, artifact.Lineage{Child: d.ID, Parent: parent, Relation: artifact.RelationDependsOn})
		}
	}
	return result
}

func (d Decision) Batch(key string) (artifact.Batch, error) {
	return decisionCodec.Batch(key, d, d.Lineage(), nil)
}

func promotionProved(decision Decision) bool {
	if len(decision.Seeds) < 2 {
		return false
	}
	comparisons := make([]runrecord.SeedComparison, 0, len(decision.Seeds))
	var costNS uint64
	for _, seed := range decision.Seeds {
		comparisons = append(comparisons, runrecord.SeedComparison{
			Seed: uint64(seed.Seed), Parent: seed.Initial.Loss, Child: seed.Final.Loss,
		})
		costNS += seed.Cost.WallNS
	}
	if err := runrecord.ValidateDescendantImprovement(runrecord.MetricContract{
		Evaluator: decision.Evaluator, Split: decision.Split,
		Comparisons:    comparisons,
		CostNS:         costNS,
		HoldoutQueries: uint64(len(decision.Charges)),
		HoldoutBudget:  uint64(len(decision.Charges)) + decision.BudgetRemaining,
	}); err != nil {
		return false
	}
	selected, selectedFound := SuiteMetrics{}, false
	for _, seed := range decision.Seeds {
		if seed.Initial.Loss-seed.Final.Loss <= decision.ObservedNoise.Loss ||
			seed.Final.ActionAccuracy+decision.ObservedNoise.ActionAccuracy < seed.Initial.ActionAccuracy ||
			seed.Final.ModalityAccuracy+decision.ObservedNoise.ModalityAccuracy < seed.Initial.ModalityAccuracy ||
			seed.Final.ValidActionRate+decision.ObservedNoise.ValidActionRate < seed.Initial.ValidActionRate {
			return false
		}
		if seed.Model == decision.Candidate {
			selected, selectedFound = seed.Final, true
		}
	}
	if !selectedFound {
		return false
	}
	eligible := 0
	for _, challenger := range decision.Challengers {
		if !challenger.Eligible {
			continue
		}
		if challenger.Metrics == nil {
			return false
		}
		eligible++
		metrics := *challenger.Metrics
		if metrics.Loss-selected.Loss <= decision.ObservedNoise.Loss ||
			selected.ActionAccuracy+decision.ObservedNoise.ActionAccuracy < metrics.ActionAccuracy ||
			selected.ModalityAccuracy+decision.ObservedNoise.ModalityAccuracy < metrics.ModalityAccuracy ||
			selected.ValidActionRate+decision.ObservedNoise.ValidActionRate < metrics.ValidActionRate {
			return false
		}
	}
	return eligible > 0
}

func canonicalizeDecision(decision *Decision) error {
	if decision == nil || decision.Version != DecisionVersion ||
		decision.State != DecisionPromote && decision.State != DecisionRefuse ||
		decision.Dataset.Kind() != artifact.KindDataset || decision.Split.Kind() != artifact.KindDatasetShard ||
		decision.CurrentChampion.Kind() != artifact.KindModel || decision.Candidate.Kind() != artifact.KindModel ||
		decision.Rollback != decision.CurrentChampion || len(decision.Seeds) == 0 || len(decision.Challengers) == 0 ||
		decision.Evaluator.Kind() != artifact.KindEvidence || decision.Budget.Kind() != artifact.KindEvidence ||
		len(decision.Charges) == 0 || !validMetrics(decision.ObservedNoise) {
		return errors.New("controller training: invalid promotion decision")
	}
	sort.Slice(decision.Seeds, func(i, j int) bool { return decision.Seeds[i].Seed < decision.Seeds[j].Seed })
	for index, seed := range decision.Seeds {
		if index > 0 && decision.Seeds[index-1].Seed == seed.Seed || seed.Model.Kind() != artifact.KindModel ||
			seed.Run.Kind() != artifact.KindRun || seed.Evaluation.Kind() != artifact.KindEvaluation ||
			!validMetrics(seed.Initial) || !validMetrics(seed.Final) || !seed.Cost.valid() {
			return errors.New("controller training: invalid seed evidence")
		}
	}
	if decision.Stochastic && len(decision.Seeds) < 2 {
		return errors.New("controller training: stochastic promotion requires multiple seeds")
	}
	sort.Slice(decision.Charges, func(i, j int) bool { return decision.Charges[i].String() < decision.Charges[j].String() })
	for index, charge := range decision.Charges {
		if charge.Kind() != artifact.KindEvidence || index > 0 && decision.Charges[index-1] == charge {
			return errors.New("controller training: invalid query charge identity")
		}
	}
	sort.Slice(decision.Challengers, func(i, j int) bool { return decision.Challengers[i].Name < decision.Challengers[j].Name })
	for index := range decision.Challengers {
		challenger := &decision.Challengers[index]
		if !validToken(challenger.Name) || challenger.Model.Kind() != artifact.KindModel ||
			index > 0 && decision.Challengers[index-1].Name == challenger.Name {
			return errors.New("controller training: invalid challenger")
		}
		if challenger.Eligible {
			if challenger.Evaluation == nil || challenger.Evaluation.Kind() != artifact.KindEvaluation ||
				challenger.Metrics == nil || !validMetrics(*challenger.Metrics) || challenger.Refusal != "" ||
				challenger.Evaluator != decision.Evaluator || challenger.Split != decision.Split {
				return errors.New("controller training: eligible challenger lacks suite evidence")
			}
		} else if challenger.Evaluation != nil || challenger.Metrics != nil || challenger.Evaluator.Valid() || challenger.Split.Valid() ||
			!validToken(challenger.Refusal) {
			return errors.New("controller training: refused challenger carries results or lacks reason")
		}
	}
	if (decision.State == DecisionPromote) != promotionProved(*decision) {
		return errors.New("controller training: promotion verdict differs from evidence")
	}
	return nil
}

func validMetrics(metrics SuiteMetrics) bool {
	if math.IsNaN(metrics.Loss) || math.IsInf(metrics.Loss, 0) || metrics.Loss < 0 {
		return false
	}
	for _, value := range []float64{metrics.ActionAccuracy, metrics.ModalityAccuracy, metrics.ValidActionRate} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
			return false
		}
	}
	return true
}

func (cost ResourceCost) valid() bool {
	return cost.WallNS > 0 && (cost.PeakDeviceBytes > 0 || cost.PeakHostBytes > 0)
}

func ObservedNoise(values []SuiteMetrics) (SuiteMetrics, error) {
	if len(values) < 2 {
		return SuiteMetrics{}, errors.New("controller training: repeated parent measurements required")
	}
	minimum, maximum := values[0], values[0]
	if !validMetrics(values[0]) {
		return SuiteMetrics{}, errors.New("controller training: invalid parent measurement")
	}
	for _, value := range values[1:] {
		if !validMetrics(value) {
			return SuiteMetrics{}, errors.New("controller training: invalid parent measurement")
		}
		minimum.Loss, maximum.Loss = min(minimum.Loss, value.Loss), max(maximum.Loss, value.Loss)
		minimum.ActionAccuracy, maximum.ActionAccuracy = min(minimum.ActionAccuracy, value.ActionAccuracy), max(maximum.ActionAccuracy, value.ActionAccuracy)
		minimum.ModalityAccuracy, maximum.ModalityAccuracy = min(minimum.ModalityAccuracy, value.ModalityAccuracy), max(maximum.ModalityAccuracy, value.ModalityAccuracy)
		minimum.ValidActionRate, maximum.ValidActionRate = min(minimum.ValidActionRate, value.ValidActionRate), max(maximum.ValidActionRate, value.ValidActionRate)
	}
	return SuiteMetrics{
		Loss: maximum.Loss - minimum.Loss, ActionAccuracy: maximum.ActionAccuracy - minimum.ActionAccuracy,
		ModalityAccuracy: maximum.ModalityAccuracy - minimum.ModalityAccuracy,
		ValidActionRate:  maximum.ValidActionRate - minimum.ValidActionRate,
	}, nil
}

func cloneDecision(value Decision) Decision {
	value.Seeds = slices.Clone(value.Seeds)
	value.Challengers = cloneChallengers(value.Challengers)
	value.Charges = slices.Clone(value.Charges)
	return value
}

func cloneChallengers(values []Challenger) []Challenger {
	result := slices.Clone(values)
	for index := range result {
		result[index].Evaluation = artifact.CloneID(result[index].Evaluation)
		if result[index].Metrics != nil {
			metrics := *result[index].Metrics
			result[index].Metrics = &metrics
		}
		result[index].Refusal = strings.TrimSpace(result[index].Refusal)
	}
	return result
}
