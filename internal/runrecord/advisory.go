package runrecord

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"maps"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
)

const (
	AdvisoryMediaType = "application/vnd.overgo.regression-advisory+json"
	AdvisorySchema    = "overgo/regression-advisory/v1"
	// AdvisoryAliasRoot scopes the latest advisory per series.
	AdvisoryAliasRoot = "advisory/active/"

	// MinAdvisoryWindow: open small-sample evidence floor.
	MinAdvisoryWindow = 3
)

var advisoryContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: AdvisoryMediaType, Schema: AdvisorySchema,
}

var advisoryCodec = artifact.JSONDocumentCodec(
	"run record advisory", advisoryContract.Kind, advisoryContract.MediaType, advisoryContract.Schema,
	canonicalizeAdvisory, func(value Advisory) artifact.ID { return value.ID },
	func(value *Advisory, id artifact.ID) { value.ID = id }, cloneAdvisory,
)

type Observation struct {
	Sequence   uint64
	Run        Run
	Evaluation Evaluation
}

type PhaseDelta struct {
	Phase            Phase  `json:"phase"`
	LatestNS         uint64 `json:"latest_ns"`
	BaselineMedianNS uint64 `json:"baseline_median_ns"`
	DeltaNS          int64  `json:"delta_ns"`
}

// Advisory defines immutable robust regression verdict.
type Advisory struct {
	Version           uint16        `json:"version"`
	Recipe            artifact.ID   `json:"recipe"`
	Environment       artifact.ID   `json:"environment"`
	Metric            string        `json:"metric"`
	Unit              string        `json:"unit,omitzero"`
	Direction         Direction     `json:"direction"`
	WindowStart       uint64        `json:"window_start"`
	WindowEnd         uint64        `json:"window_end"`
	LatestSequence    uint64        `json:"latest_sequence"`
	LatestRun         artifact.ID   `json:"latest_run"`
	LatestValue       float64       `json:"latest_value"`
	BaselineMedian    float64       `json:"baseline_median"`
	MAD               float64       `json:"mad"`
	SurpriseNumerator float64       `json:"surprise_numerator"`
	Threshold         float64       `json:"threshold"`
	SourceRuns        []artifact.ID `json:"source_runs"`
	SourceEvaluations []artifact.ID `json:"source_evaluations"`
	PhaseDeltas       []PhaseDelta  `json:"phase_deltas,omitempty"`
	ID                artifact.ID   `json:"-"`
}

func DetectRegression(
	observations []Observation,
	metricName string,
	window int,
	threshold float64,
) (Advisory, bool, error) {
	if window < MinAdvisoryWindow ||
		!checked.PositiveFinite64(threshold) || !validLabel(metricName) {
		return Advisory{}, false, errors.New("run record: invalid advisory configuration")
	}
	ordered := slices.Clone(observations)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Sequence < ordered[j].Sequence })
	if len(ordered) < window+1 {
		return Advisory{}, false, errors.New("run record: insufficient advisory observations")
	}
	ordered = ordered[len(ordered)-window-1:]
	metricValues := make([]float64, len(ordered))
	var recipeID, environmentID artifact.ID
	var unit string
	var direction Direction
	for index, observation := range ordered {
		metric, err := validateObservation(observation, metricName)
		if err != nil {
			return Advisory{}, false, err
		}
		if index == 0 {
			recipeID, environmentID = observation.Run.Recipe, observation.Run.Environment
			unit, direction = metric.Unit, metric.Direction
		} else if observation.Sequence == ordered[index-1].Sequence ||
			observation.Run.Recipe != recipeID || observation.Run.Environment != environmentID ||
			metric.Unit != unit || metric.Direction != direction {
			return Advisory{}, false, errors.New("run record: advisory series key differs")
		}
		metricValues[index] = metric.Value
	}
	baseline := slices.Clone(metricValues[:window])
	baselineMedian := medianFloat(baseline)
	deviations := make([]float64, len(baseline))
	for index, value := range baseline {
		deviations[index] = math.Abs(value - baselineMedian)
	}
	mad := medianFloat(deviations)
	latestValue := metricValues[window]
	numerator := regressionDistance(latestValue, baselineMedian, direction)
	if numerator == 0 || mad > 0 && numerator/mad < threshold {
		return Advisory{}, false, nil
	}
	advisory := Advisory{
		Version: artifact.InitialDocumentVersion, Recipe: recipeID, Environment: environmentID,
		Metric: metricName, Unit: unit, Direction: direction,
		WindowStart: ordered[0].Sequence, WindowEnd: ordered[window-1].Sequence,
		LatestSequence: ordered[window].Sequence, LatestRun: ordered[window].Run.ID,
		LatestValue: latestValue, BaselineMedian: baselineMedian, MAD: mad,
		SurpriseNumerator: numerator, Threshold: threshold,
		SourceRuns:        make([]artifact.ID, len(ordered)),
		SourceEvaluations: make([]artifact.ID, len(ordered)),
		PhaseDeltas:       phaseDeltas(ordered[:window], ordered[window]),
	}
	for index, observation := range ordered {
		advisory.SourceRuns[index] = observation.Run.ID
		advisory.SourceEvaluations[index] = observation.Evaluation.ID
	}
	advisory, err := advisoryCodec.New(advisory)
	return advisory, err == nil, err
}

func ParseAdvisory(content []byte) (Advisory, error) {
	return advisoryCodec.Parse(content)
}

func (a Advisory) Surprise() float64 {
	if a.MAD == 0 && a.SurpriseNumerator == 0 {
		return a.SurpriseNumerator
	}
	return a.SurpriseNumerator / a.MAD
}

func (a Advisory) ValidateIdentity() error {
	return advisoryCodec.ValidateIdentity(a)
}

func (a Advisory) Content() (artifact.Content, error) {
	return advisoryCodec.Content(a)
}

func (a Advisory) Lineage() []artifact.Lineage {
	parents := make([]artifact.ID, 0, 2+len(a.SourceRuns)+len(a.SourceEvaluations))
	parents = append(parents, a.Recipe, a.Environment)
	parents = append(parents, a.SourceRuns...)
	parents = append(parents, a.SourceEvaluations...)
	edges := make([]artifact.Lineage, len(parents))
	for index, parent := range parents {
		edges[index] = artifact.Lineage{
			Child: a.ID, Parent: parent, Relation: artifact.RelationDependsOn,
		}
	}
	return edges
}

func (a Advisory) Batch(key string) (artifact.Batch, error) {
	return advisoryCodec.Batch(key, a, a.Lineage(), nil)
}

// AdvisoryAlias identifies one recipe, environment, and metric series.
func AdvisoryAlias(recipe, environment artifact.ID, metric string) string {
	digest := sha256.Sum256([]byte(recipe.String() + "\x00" + environment.String() + "\x00" + metric))
	return AdvisoryAliasRoot + hex.EncodeToString(digest[:])
}

func validateObservation(observation Observation, metricName string) (Metric, error) {
	if observation.Sequence == 0 || observation.Run.Version != artifact.SecondDocumentVersion ||
		observation.Run.Outcome != OutcomeSucceeded || observation.Evaluation.Run != observation.Run.ID ||
		observation.Evaluation.Recipe != observation.Run.Recipe {
		return Metric{}, errors.New("run record: invalid advisory observation")
	}
	if err := observation.Run.ValidateIdentity(); err != nil {
		return Metric{}, err
	}
	if err := observation.Evaluation.ValidateIdentity(); err != nil {
		return Metric{}, err
	}
	for _, metric := range observation.Evaluation.Metrics {
		if metric.Name == metricName {
			return metric, nil
		}
	}
	return Metric{}, errors.New("run record: advisory metric is absent")
}

func regressionDistance(latest, baseline float64, direction Direction) float64 {
	if advantage, ok := direction.Advantage(latest, baseline); ok {
		var noRegression float64
		return max(noRegression, -advantage)
	}
	return math.Abs(latest - baseline)
}

func phaseDeltas(baseline []Observation, latest Observation) []PhaseDelta {
	values := make(map[Phase][]uint64)
	for _, observation := range baseline {
		seen := make(map[Phase]struct{}, len(observation.Run.Phases))
		for _, metric := range observation.Run.Phases {
			values[metric.Phase] = append(values[metric.Phase], metric.DurationNS)
			seen[metric.Phase] = struct{}{}
		}
		maps.DeleteFunc(values, func(phase Phase, _ []uint64) bool {
			_, ok := seen[phase]
			return !ok
		})
	}
	var deltas []PhaseDelta
	for _, metric := range latest.Run.Phases {
		phaseValues, ok := values[metric.Phase]
		if !ok || len(phaseValues) != len(baseline) {
			continue
		}
		median := medianUint(phaseValues)
		deltas = append(deltas, PhaseDelta{
			Phase: metric.Phase, LatestNS: metric.DurationNS,
			BaselineMedianNS: median, DeltaNS: int64(metric.DurationNS) - int64(median),
		})
	}
	return deltas
}

func medianFloat(values []float64) float64 {
	slices.Sort(values)
	middle := len(values) / 2
	if len(values)%2 != 0 {
		return values[middle]
	}
	return values[middle-1]/2 + values[middle]/2
}

func medianUint(values []uint64) uint64 {
	values = slices.Clone(values)
	slices.Sort(values)
	middle := len(values) / 2
	if len(values)%2 != 0 {
		return values[middle]
	}
	left, right := values[middle-1], values[middle]
	return left/2 + right/2 + (left%2+right%2)/2
}

func canonicalizeAdvisory(advisory *Advisory) error {
	if advisory == nil || advisory.Version != artifact.InitialDocumentVersion ||
		advisory.Recipe.Kind() != artifact.KindRecipe || advisory.Environment.Kind() != artifact.KindEvidence ||
		advisory.LatestRun.Kind() != artifact.KindRun || !validLabel(advisory.Metric) || !validUnit(advisory.Unit) ||
		advisory.WindowStart == 0 ||
		advisory.WindowEnd < advisory.WindowStart || advisory.LatestSequence <= advisory.WindowEnd ||
		advisory.Direction != DirectionNeutral && advisory.Direction != DirectionMinimize &&
			advisory.Direction != DirectionMaximize ||
		!checked.Finite64(advisory.LatestValue) || !checked.Finite64(advisory.BaselineMedian) ||
		!checked.NonNegativeFinite64(advisory.MAD) || !checked.PositiveFinite64(advisory.SurpriseNumerator) ||
		!checked.PositiveFinite64(advisory.Threshold) ||
		len(advisory.SourceRuns) < MinAdvisoryWindow+1 ||
		len(advisory.SourceRuns) != len(advisory.SourceEvaluations) {
		return errors.New("run record: invalid advisory")
	}
	if err := canonicalIDs(advisory.SourceRuns); err != nil {
		return err
	}
	if err := canonicalIDs(advisory.SourceEvaluations); err != nil {
		return err
	}
	for _, id := range advisory.SourceRuns {
		if id.Kind() != artifact.KindRun {
			return errors.New("run record: advisory source is not a run")
		}
	}
	for _, id := range advisory.SourceEvaluations {
		if id.Kind() != artifact.KindEvaluation {
			return errors.New("run record: advisory source is not an evaluation")
		}
	}
	if !slices.Contains(advisory.SourceRuns, advisory.LatestRun) {
		return errors.New("run record: advisory omits latest run")
	}
	for index, delta := range advisory.PhaseDeltas {
		if !validPhase(delta.Phase) || delta.LatestNS > math.MaxInt64 ||
			delta.BaselineMedianNS > math.MaxInt64 ||
			delta.DeltaNS != int64(delta.LatestNS)-int64(delta.BaselineMedianNS) ||
			index > 0 && advisory.PhaseDeltas[index-1].Phase >= delta.Phase {
			return errors.New("run record: invalid advisory phase delta")
		}
	}
	return nil
}

func cloneAdvisory(advisory Advisory) Advisory {
	advisory.SourceRuns = slices.Clone(advisory.SourceRuns)
	advisory.SourceEvaluations = slices.Clone(advisory.SourceEvaluations)
	advisory.PhaseDeltas = slices.Clone(advisory.PhaseDeltas)
	return advisory
}
