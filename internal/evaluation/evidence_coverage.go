package evaluation

import (
	"cmp"
	"context"
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/executionfailure"
	"overgo/internal/overgodb"
	runrecord "overgo/internal/runrecord"
)

// EvidenceCoverageProjectionVersion identifies the deterministic evidence
// coverage projection contract.
const EvidenceCoverageProjectionVersion uint16 = artifact.InitialDocumentVersion

// CoverageAxis is one independently reported evidence dimension.
type CoverageAxis string

const (
	// CoverageMeasurement requires exact resource measurements.
	CoverageMeasurement CoverageAxis = "measurement"
	// CoverageEvaluation requires a typed evaluation bound to the unit.
	CoverageEvaluation CoverageAxis = "evaluation"
	// CoverageHardware requires hardware observation samples.
	CoverageHardware CoverageAxis = "hardware"
	// CoverageCausal requires a typed causal projection input.
	CoverageCausal CoverageAxis = "causal"
)

// CoverageRequirement makes one unit's evidence denominator explicit.
// ExpectedSamples applies only to measurement and hardware. Metrics apply only
// to measurement; an empty metric list accepts any measured sample.
type CoverageRequirement struct {
	Axis            CoverageAxis               `json:"axis"`
	ExpectedSamples uint32                     `json:"expected_samples,omitempty"`
	Metrics         []runrecord.ResourceMetric `json:"metrics,omitempty"`
}

// CoverageUnit is one exact attempt and its required evidence axes.
type CoverageUnit struct {
	Attempt artifact.ID `json:"attempt"`
	// ObservationHead optionally pins the exact immutable summary at the end
	// of Attempt's observation stream. A zero identity preserves live alias
	// resolution for exploratory coverage queries; persisted fitness evidence
	// always supplies the head so later alias advances cannot rewrite history.
	ObservationHead artifact.ID `json:"observation_head,omitzero"`
	// Terminal names the exact immutable terminal fact used for outcome
	// coverage. A zero identity makes no terminal claim. Run and AttemptRecord
	// terminals are self-identifying; a TerminalAttemptReceipt names Attempt as
	// its operation.
	Terminal artifact.ID `json:"terminal,omitzero"`
	// EvaluationEvidence names exact evidence bundles for this run. They are
	// the only run-level causal and comparison facts this projection reads.
	EvaluationEvidence []artifact.ID `json:"evaluation_evidence,omitempty"`
	// FailureObservations names the exact failure evidence counted for a
	// terminal Run. Receipts carry their own exact failure tuple.
	FailureObservations []artifact.ID         `json:"failure_observations,omitempty"`
	Required            []CoverageRequirement `json:"required"`
}

// CoveragePair names one baseline/candidate comparison already admitted by
// its comparison owner. This projector reports whether both exact endpoints
// have complete evidence; it does not infer comparability from optional
// observation fields.
type CoveragePair struct {
	Baseline  artifact.ID `json:"baseline"`
	Candidate artifact.ID `json:"candidate"`
}

// AxisCoverage keeps missing and degraded evidence distinct from observations.
type AxisCoverage struct {
	Observed uint64 `json:"observed"`
	Missing  uint64 `json:"missing"`
	Degraded uint64 `json:"degraded"`
}

// CoverageBounds limits all indexed facts and raw observation chunks read by
// one projection. The budgets are shared across its complete denominator.
type CoverageBounds struct {
	MaxFacts    int                               `json:"max_facts"`
	Observation runrecord.ObservationStreamBounds `json:"observation"`
}

// CoverageQuery is a canonical denominator plus its bounded traversal budget.
// Construct it with NewEvidenceCoverageQuery before calling Project.
type CoverageQuery struct {
	Units  []CoverageUnit `json:"units"`
	Pairs  []CoveragePair `json:"pairs,omitempty"`
	Bounds CoverageBounds `json:"bounds"`
	ID     artifact.ID    `json:"-"`
}

// CoverageProjection reports evidence presence only. It deliberately carries
// no quality or capability scores.
type CoverageProjection struct {
	Version                    uint16            `json:"version"`
	FailureClassifierVersion   uint16            `json:"failure_classifier_version"`
	CausalityProjectionVersion uint16            `json:"causality_projection_version"`
	Query                      artifact.ID       `json:"query"`
	Head                       artifact.CommitID `json:"head"`
	Sequence                   uint64            `json:"sequence"`
	Units                      uint64            `json:"units"`
	SourceChunks               []artifact.ID     `json:"source_chunks,omitempty"`
	Measurement                AxisCoverage      `json:"measurement"`
	Evaluation                 AxisCoverage      `json:"evaluation"`
	Hardware                   AxisCoverage      `json:"hardware"`
	Causal                     AxisCoverage      `json:"causal"`
	Failures                   uint64            `json:"failures"`
	FailuresClassified         uint64            `json:"failures_classified"`
	FailuresUnclassified       uint64            `json:"failures_unclassified"`
	Costed                     uint64            `json:"costed"`
	Uncosted                   uint64            `json:"uncosted"`
	Recovered                  uint64            `json:"recovered"`
	Pairs                      uint64            `json:"pairs"`
	Paired                     uint64            `json:"paired"`
	Unpaired                   uint64            `json:"unpaired"`
}

// NewEvidenceCoverageQuery canonicalizes and identifies an explicit bounded
// evidence denominator.
func NewEvidenceCoverageQuery(
	units []CoverageUnit,
	pairs []CoveragePair,
	bounds CoverageBounds,
) (CoverageQuery, error) {
	query := CoverageQuery{
		Units: cloneCoverageUnits(units), Pairs: slices.Clone(pairs), Bounds: bounds,
	}
	if err := canonicalizeCoverageQuery(&query); err != nil {
		return CoverageQuery{}, err
	}
	id, err := artifact.JSONID(artifact.KindProfile, query)
	if err != nil {
		return CoverageQuery{}, err
	}
	query.ID = id
	return query, nil
}

// Project reads only bounded indexes for the query's explicit units and binds
// the result to one exact journal head. A concurrent head change is refused.
func (query CoverageQuery) Project(ctx context.Context, store *overgodb.Store) (CoverageProjection, error) {
	return query.project(ctx, store, nil)
}

type coverageObservationKey struct {
	attempt artifact.ID
	head    artifact.ID
}

type coverageObservationCache map[coverageObservationKey]runrecord.ObservationStream

// project accepts exact streams already replayed by a stronger containing
// proof. The cache is private so public coverage callers cannot substitute
// unchecked materialized aggregates for owner-verified observation content.
func (query CoverageQuery) project(
	ctx context.Context,
	store *overgodb.Store,
	observations coverageObservationCache,
) (CoverageProjection, error) {
	if ctx == nil || store == nil {
		return CoverageProjection{}, errors.New("evaluation: evidence coverage store is absent")
	}
	canonical, err := NewEvidenceCoverageQuery(
		query.Units, query.Pairs, query.Bounds,
	)
	if err != nil || query.ID != canonical.ID || !slices.EqualFunc(query.Units, canonical.Units, equalCoverageUnit) ||
		!slices.Equal(query.Pairs, canonical.Pairs) {
		return CoverageProjection{}, errors.Join(err, errors.New("evaluation: evidence coverage query is not canonical"))
	}
	return projectEvidenceCoverage(ctx, store, canonical, observations)
}

func canonicalizeCoverageQuery(query *CoverageQuery) error {
	if query == nil || query.Bounds.MaxFacts <= 0 || query.Bounds.Observation.MaxChunks <= 0 ||
		query.Bounds.MaxFacts > runrecord.MaximumAttemptPopulation ||
		query.Bounds.Observation.MaxChunks > runrecord.MaximumAttemptPopulation ||
		query.Bounds.Observation.MaxRawBytes == 0 ||
		query.Bounds.Observation.MaxRawBytes > artifact.MaxContentBytes || len(query.Units) == 0 ||
		len(query.Units) > runrecord.MaximumAttemptPopulation ||
		len(query.Pairs) > runrecord.MaximumAttemptPopulation {
		return errors.New("evaluation: invalid evidence coverage denominator")
	}
	declaredFacts := 0
	for unitIndex := range query.Units {
		unit := &query.Units[unitIndex]
		var ok bool
		declaredFacts, ok = checked.AddInt(
			declaredFacts, len(unit.EvaluationEvidence), len(unit.FailureObservations),
		)
		if !ok || declaredFacts > query.Bounds.MaxFacts {
			return errors.New("evaluation: evidence coverage sources exceed fact budget")
		}
		if unit.Attempt.Kind() != artifact.KindRun && unit.Attempt.Kind() != artifact.KindEvidence {
			return errors.New("evaluation: invalid evidence coverage unit")
		}
		if unit.ObservationHead.Valid() && unit.ObservationHead.Kind() != artifact.KindEvidence {
			return errors.New("evaluation: invalid evidence coverage observation head")
		}
		if unit.Terminal.Valid() && unit.Terminal.Kind() != artifact.KindRun &&
			unit.Terminal.Kind() != artifact.KindEvidence {
			return errors.New("evaluation: invalid evidence coverage terminal")
		}
		if unit.Terminal.Valid() && (unit.Attempt.Kind() == artifact.KindRun && unit.Terminal != unit.Attempt ||
			unit.Attempt.Kind() == artifact.KindEvidence && unit.Terminal.Kind() == artifact.KindRun) {
			return errors.New("evaluation: evidence coverage terminal kind differs from its unit")
		}
		evaluationEvidenceCount, failureObservationCount := len(unit.EvaluationEvidence), len(unit.FailureObservations)
		unit.EvaluationEvidence = canonicalCoverageEvidenceIDs(unit.EvaluationEvidence)
		unit.FailureObservations = canonicalCoverageEvidenceIDs(unit.FailureObservations)
		if len(unit.EvaluationEvidence) != evaluationEvidenceCount ||
			len(unit.FailureObservations) != failureObservationCount {
			return errors.New("evaluation: duplicate evidence coverage source")
		}
		for _, evidence := range append(slices.Clone(unit.EvaluationEvidence), unit.FailureObservations...) {
			if evidence.Kind() != artifact.KindEvidence {
				return errors.New("evaluation: invalid evidence coverage source")
			}
		}
		if unit.Attempt.Kind() == artifact.KindEvidence && len(unit.EvaluationEvidence) != 0 ||
			unit.Attempt.Kind() != artifact.KindRun && len(unit.FailureObservations) != 0 ||
			len(unit.FailureObservations) != 0 && unit.Terminal != unit.Attempt {
			return errors.New("evaluation: evidence coverage sources differ from terminal contract")
		}
		for requirementIndex := range unit.Required {
			requirement := &unit.Required[requirementIndex]
			requirement.Metrics = slices.Clone(requirement.Metrics)
			sort.Slice(requirement.Metrics, func(i, j int) bool { return requirement.Metrics[i] < requirement.Metrics[j] })
			if !validCoverageAxis(requirement.Axis) ||
				(requirement.Axis == CoverageMeasurement || requirement.Axis == CoverageHardware) != (requirement.ExpectedSamples > 0) ||
				requirement.Axis != CoverageMeasurement && requirement.Axis != CoverageHardware && len(requirement.Metrics) != 0 {
				return errors.New("evaluation: invalid evidence coverage requirement")
			}
			for _, metric := range requirement.Metrics {
				if !metric.Valid() || requirement.Axis == CoverageHardware && !metric.Hardware() {
					return errors.New("evaluation: foreign evidence coverage metric")
				}
			}
			if len(slices.Compact(slices.Clone(requirement.Metrics))) != len(requirement.Metrics) {
				return errors.New("evaluation: duplicate evidence coverage metric")
			}
		}
		if len(unit.Required) == 0 {
			unit.Required = nil
		}
		sort.Slice(unit.Required, func(i, j int) bool { return unit.Required[i].Axis < unit.Required[j].Axis })
		if len(slices.CompactFunc(slices.Clone(unit.Required), func(left, right CoverageRequirement) bool {
			return left.Axis == right.Axis
		})) != len(unit.Required) {
			return errors.New("evaluation: duplicate evidence coverage axis")
		}
	}
	sort.Slice(query.Units, func(i, j int) bool { return artifact.CompareID(query.Units[i].Attempt, query.Units[j].Attempt) < 0 })
	if len(slices.CompactFunc(slices.Clone(query.Units), func(left, right CoverageUnit) bool {
		return left.Attempt == right.Attempt
	})) != len(query.Units) {
		return errors.New("evaluation: duplicate evidence coverage unit")
	}
	unitIDs := make([]artifact.ID, len(query.Units))
	for index, unit := range query.Units {
		unitIDs[index] = unit.Attempt
	}
	for _, pair := range query.Pairs {
		if pair.Baseline == pair.Candidate ||
			!slices.Contains(unitIDs, pair.Baseline) || !slices.Contains(unitIDs, pair.Candidate) {
			return errors.New("evaluation: invalid evidence coverage pair")
		}
		baseline := query.Units[slices.Index(unitIDs, pair.Baseline)]
		candidate := query.Units[slices.Index(unitIDs, pair.Candidate)]
		if len(baseline.Required) == 0 || len(candidate.Required) == 0 {
			return errors.New("evaluation: evidence coverage pair has no required axes")
		}
		if !equalCoverageRequirements(baseline.Required, candidate.Required) {
			return errors.New("evaluation: evidence coverage pair requirements differ")
		}
	}
	sort.Slice(query.Pairs, func(i, j int) bool {
		if order := artifact.CompareID(query.Pairs[i].Baseline, query.Pairs[j].Baseline); order != 0 {
			return order < 0
		}
		return artifact.CompareID(query.Pairs[i].Candidate, query.Pairs[j].Candidate) < 0
	})
	if len(slices.Compact(slices.Clone(query.Pairs))) != len(query.Pairs) {
		return errors.New("evaluation: duplicate evidence coverage pair")
	}
	return nil
}

func validCoverageAxis(axis CoverageAxis) bool {
	return slices.Contains([]CoverageAxis{
		CoverageMeasurement,
		CoverageEvaluation,
		CoverageHardware,
		CoverageCausal,
	}, axis)
}

func cloneCoverageUnits(units []CoverageUnit) []CoverageUnit {
	result := slices.Clone(units)
	for index := range result {
		result[index].EvaluationEvidence = slices.Clone(result[index].EvaluationEvidence)
		result[index].FailureObservations = slices.Clone(result[index].FailureObservations)
		result[index].Required = slices.Clone(result[index].Required)
		for requirement := range result[index].Required {
			result[index].Required[requirement].Metrics = slices.Clone(result[index].Required[requirement].Metrics)
		}
	}
	return result
}

func equalCoverageUnit(left, right CoverageUnit) bool {
	return left.Attempt == right.Attempt && left.ObservationHead == right.ObservationHead && left.Terminal == right.Terminal &&
		slices.Equal(left.EvaluationEvidence, right.EvaluationEvidence) &&
		slices.Equal(left.FailureObservations, right.FailureObservations) &&
		equalCoverageRequirements(left.Required, right.Required)
}

func canonicalCoverageEvidenceIDs(ids []artifact.ID) []artifact.ID {
	ids = slices.Clone(ids)
	sort.Slice(ids, func(i, j int) bool { return artifact.CompareID(ids[i], ids[j]) < 0 })
	return slices.Compact(ids)
}

func equalCoverageRequirements(left, right []CoverageRequirement) bool {
	return slices.EqualFunc(left, right, func(a, b CoverageRequirement) bool {
		return a.Axis == b.Axis && a.ExpectedSamples == b.ExpectedSamples && slices.Equal(a.Metrics, b.Metrics)
	})
}

type coverageState uint8

const (
	coverageMissing coverageState = iota
	coverageDegraded
	coverageObserved
)

type projectedCoverageUnit struct {
	axes       map[CoverageAxis]coverageState
	pair       coveragePairContext
	pairValid  bool
	failed     bool
	classified bool
	costed     bool
	recovered  bool
}

type coverageBudget struct {
	facts       int
	observation runrecord.ObservationStreamBounds
	plans       map[artifact.ID]coveragePlanContext
}

func newCoverageBudget(bounds CoverageBounds) coverageBudget {
	return coverageBudget{
		facts: bounds.MaxFacts, observation: bounds.Observation,
		plans: make(map[artifact.ID]coveragePlanContext),
	}
}

func (budget *coverageBudget) takeFacts(count int) error {
	if budget == nil || count < 0 || count > budget.facts {
		return errors.New("evaluation: evidence coverage exceeded its fact budget")
	}
	budget.facts -= count
	return nil
}

func (budget *coverageBudget) takeFact() error {
	if budget == nil || budget.facts == 0 {
		return errors.New("evaluation: evidence coverage exceeded its fact budget")
	}
	budget.facts--
	return nil
}

func (budget *coverageBudget) takeObservation(stream runrecord.ObservationStream) error {
	if budget == nil || len(stream.ChunkIDs) > budget.observation.MaxChunks ||
		stream.Coverage.RawBytes > budget.observation.MaxRawBytes {
		return errors.New("evaluation: evidence coverage exceeded its observation budget")
	}
	budget.observation.MaxChunks -= len(stream.ChunkIDs)
	budget.observation.MaxRawBytes -= stream.Coverage.RawBytes
	return nil
}

type coverageCausalExecution struct {
	id            artifact.ID
	typed         *runrecord.CausalContext
	requiresTyped bool
}

type coveragePairDomain uint8

const (
	coveragePairAttempt coveragePairDomain = iota + 1
	coveragePairEvaluation
)

type coveragePairContext struct {
	domain      coveragePairDomain
	task        artifact.ID
	dataset     artifact.ID
	split       artifact.ID
	environment artifact.ID
}

type coveragePlanContext struct {
	pair            coveragePairContext
	modelDefinition artifact.ID
	recipe          artifact.ID
	codeCommit      string
}

func (context coveragePlanContext) matches(evidence coverageEvaluationAuthority) bool {
	return context.modelDefinition == evidence.ModelDefinition && context.recipe == evidence.Recipe &&
		context.pair.dataset == evidence.Dataset && context.pair.split == evidence.Split &&
		context.pair.environment == evidence.Environment && context.codeCommit == evidence.CodeCommit
}

func (context coveragePairContext) valid() bool {
	switch context.domain {
	case coveragePairAttempt:
		return context.task.Valid() && context.environment.Valid() &&
			!context.dataset.Valid() && !context.split.Valid()
	case coveragePairEvaluation:
		return context.task.Valid() && context.dataset.Kind() == artifact.KindDataset &&
			context.split.Kind() == artifact.KindDatasetShard && context.environment.Valid()
	default:
		return false
	}
}

type coverageEvaluationFacts struct {
	observed         bool
	causalExecutions []coverageCausalExecution
	pair             coveragePairContext
}

type coverageEvaluationAuthority struct {
	ID              artifact.ID
	Plan            artifact.ID
	ModelDefinition artifact.ID
	Recipe          artifact.ID
	Dataset         artifact.ID
	Split           artifact.ID
	Environment     artifact.ID
	CodeCommit      string
	Causal          *runrecord.CausalContext
}

type coverageEvaluationRelevance struct {
	run        runrecord.Run
	evaluation runrecord.Evaluation
}

func (relevance coverageEvaluationRelevance) verify(
	_ context.Context,
	_ artifact.Reader,
	evidence EvaluationEvidence,
) error {
	if relevance.run.ID != evidence.Run || relevance.run.Outcome != runrecord.OutcomeSucceeded ||
		relevance.evaluation.ID != evidence.Evaluation || relevance.evaluation.Run != relevance.run.ID ||
		relevance.run.Recipe != evidence.Recipe || relevance.evaluation.Recipe != evidence.Recipe ||
		relevance.evaluation.Dataset != evidence.Dataset || relevance.run.Environment != evidence.Environment ||
		relevance.run.CodeCommit != evidence.CodeCommit || !slices.Contains(relevance.run.Inputs, evidence.Plan) {
		return errors.New("evaluation: coverage evidence differs from unit")
	}
	return nil
}

// requireCoverageEvaluationEvidence verifies only claims consumed by this
// bounded projection. The metered plan and causality reads below prove its
// remaining comparison and causal claims; quality authorities stay under the
// full RequireEvaluationEvidence owner.
func requireCoverageEvaluationEvidence(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	relevance coverageEvaluationRelevance,
) (coverageEvaluationAuthority, error) {
	evidence, err := evaluationEvidenceCodec.RequireVerified(ctx, reader, id, relevance.verify)
	if err != nil {
		return coverageEvaluationAuthority{}, err
	}
	return coverageAuthorityFromEvidence(evidence), nil
}

func coverageAuthorityFromEvidence(evidence EvaluationEvidence) coverageEvaluationAuthority {
	var causal *runrecord.CausalContext
	if evidence.Causal != nil {
		cloned := *evidence.Causal
		cloned.Motivation = slices.Clone(evidence.Causal.Motivation)
		causal = &cloned
	}
	return coverageEvaluationAuthority{
		ID: evidence.ID, Plan: evidence.Plan, ModelDefinition: evidence.ModelDefinition,
		Recipe: evidence.Recipe, Dataset: evidence.Dataset, Split: evidence.Split,
		Environment: evidence.Environment, CodeCommit: evidence.CodeCommit, Causal: causal,
	}
}

type coverageUnitFacts struct {
	outcomeKnown         bool
	failed               bool
	recovered            bool
	runTerminal          bool
	wallMeasured         bool
	wallNS               uint64
	causalAuthoritative  bool
	costed               bool
	costUnits            uint64
	pair                 coveragePairContext
	causal               *runrecord.CausalContext
	failureObservation   artifact.ID
	failureNormalization artifact.ID
	run                  runrecord.Run
}

func projectEvidenceCoverage(
	ctx context.Context,
	store *overgodb.Store,
	query CoverageQuery,
	observations coverageObservationCache,
) (CoverageProjection, error) {
	head, sequence := store.Head()
	projection := CoverageProjection{
		Version:                    EvidenceCoverageProjectionVersion,
		FailureClassifierVersion:   executionfailure.ClassifierVersion,
		CausalityProjectionVersion: overgodb.CausalityProjectionVersion,
		Query:                      query.ID, Head: head, Sequence: sequence,
		Units: uint64(len(query.Units)), Pairs: uint64(len(query.Pairs)),
	}
	budget := newCoverageBudget(query.Bounds)
	units := make(map[artifact.ID]projectedCoverageUnit, len(query.Units))
	for _, unit := range query.Units {
		projected, sources, err := projectCoverageUnit(ctx, store, unit, head, sequence, &budget, observations)
		if err != nil {
			return CoverageProjection{}, err
		}
		units[unit.Attempt] = projected
		projection.SourceChunks = append(projection.SourceChunks, sources...)
		for axis, state := range projected.axes {
			counts := projection.axis(axis)
			incrementAxisCoverage(counts, state)
		}
		if projected.failed {
			projection.Failures++
			if projected.classified {
				projection.FailuresClassified++
			} else {
				projection.FailuresUnclassified++
			}
		}
		if projected.costed {
			projection.Costed++
		} else {
			projection.Uncosted++
		}
		if projected.recovered {
			projection.Recovered++
		}
	}
	for _, pair := range query.Pairs {
		baseline, candidate := units[pair.Baseline], units[pair.Candidate]
		if coverageUnitComplete(baseline) && coverageUnitComplete(candidate) &&
			baseline.pairValid && candidate.pairValid && baseline.pair == candidate.pair {
			projection.Paired++
		} else {
			projection.Unpaired++
		}
	}
	sort.Slice(projection.SourceChunks, func(i, j int) bool {
		return artifact.CompareID(projection.SourceChunks[i], projection.SourceChunks[j]) < 0
	})
	projection.SourceChunks = slices.Compact(projection.SourceChunks)
	if currentHead, currentSequence := store.Head(); currentHead != head || currentSequence != sequence {
		return CoverageProjection{}, errors.New("evaluation: evidence coverage journal head changed")
	}
	return projection, nil
}

func projectCoverageUnit(
	ctx context.Context,
	store *overgodb.Store,
	unit CoverageUnit,
	head artifact.CommitID,
	sequence uint64,
	budget *coverageBudget,
	observations coverageObservationCache,
) (projectedCoverageUnit, []artifact.ID, error) {
	projected := projectedCoverageUnit{axes: make(map[CoverageAxis]coverageState, len(unit.Required))}
	var stream runrecord.ObservationStream
	var foundStream bool
	var err error
	if unit.ObservationHead.Valid() {
		stream, foundStream = observations[coverageObservationKey{attempt: unit.Attempt, head: unit.ObservationHead}]
		if !foundStream {
			stream, err = runrecord.RequireObservationStream(
				ctx, store, unit.Attempt, unit.ObservationHead, budget.observation,
			)
			foundStream = err == nil
		}
	} else {
		stream, foundStream, err = runrecord.LoadObservationStream(ctx, store, unit.Attempt, budget.observation)
	}
	if err != nil {
		return projectedCoverageUnit{}, nil, err
	}
	if foundStream {
		if err := budget.takeObservation(stream); err != nil {
			return projectedCoverageUnit{}, nil, err
		}
	}
	streamCost, streamCosted := stream.Aggregate.Measure(runrecord.ResourceCostUnits)
	streamWall, streamWallMeasured := stream.Aggregate.Measure(runrecord.ResourceWallNS)
	facts, err := loadCoverageUnitFacts(
		ctx, store, unit, budget,
	)
	if err != nil {
		return projectedCoverageUnit{}, nil, err
	}
	if streamCosted && facts.costed && streamCost != facts.costUnits {
		return projectedCoverageUnit{}, nil, errors.New("evaluation: typed and observed attempt cost differ")
	}
	if streamWallMeasured && facts.wallMeasured && streamWall != facts.wallNS {
		return projectedCoverageUnit{}, nil, errors.New("evaluation: typed and observed attempt wall time differ")
	}
	projected.costed = streamCosted || facts.costed
	evaluationFacts, err := coverageEvaluationSources(ctx, store, unit, facts.run, budget)
	if err != nil {
		return projectedCoverageUnit{}, nil, err
	}
	projected.pair, projected.pairValid = facts.pair, facts.pair.valid()
	if evaluationFacts.pair.valid() {
		if projected.pairValid && projected.pair != evaluationFacts.pair {
			return projectedCoverageUnit{}, nil, errors.New("evaluation: evidence coverage pair contexts differ")
		}
		projected.pair, projected.pairValid = evaluationFacts.pair, true
	}
	causalState, recovered, err := coverageCausality(
		ctx, store, unit.Attempt, facts.causal, facts.causalAuthoritative,
		evaluationFacts.causalExecutions,
		head, sequence, budget,
	)
	if err != nil {
		return projectedCoverageUnit{}, nil, err
	}
	projected.failed = facts.failed
	projected.recovered = recovered || facts.recovered
	projected.classified, err = coverageFailure(
		ctx, store, unit, facts, budget,
	)
	if err != nil {
		return projectedCoverageUnit{}, nil, err
	}
	for _, requirement := range unit.Required {
		switch requirement.Axis {
		case CoverageMeasurement:
			projected.axes[requirement.Axis] = measurementCoverage(stream, foundStream, requirement)
		case CoverageEvaluation:
			if evaluationFacts.observed {
				projected.axes[requirement.Axis] = coverageObserved
			} else {
				projected.axes[requirement.Axis] = coverageMissing
			}
		case CoverageHardware:
			projected.axes[requirement.Axis] = hardwareCoverage(stream, foundStream, requirement)
		case CoverageCausal:
			projected.axes[requirement.Axis] = causalState
		}
	}
	return projected, slices.Clone(stream.ChunkIDs), nil
}

func (projection *CoverageProjection) axis(axis CoverageAxis) *AxisCoverage {
	switch axis {
	case CoverageMeasurement:
		return &projection.Measurement
	case CoverageEvaluation:
		return &projection.Evaluation
	case CoverageHardware:
		return &projection.Hardware
	default:
		return &projection.Causal
	}
}

func incrementAxisCoverage(counts *AxisCoverage, state coverageState) {
	switch state {
	case coverageObserved:
		counts.Observed++
	case coverageDegraded:
		counts.Degraded++
	default:
		counts.Missing++
	}
}

func coverageUnitComplete(unit projectedCoverageUnit) bool {
	for _, state := range unit.axes {
		if state != coverageObserved {
			return false
		}
	}
	return len(unit.axes) != 0
}

func measurementCoverage(
	stream runrecord.ObservationStream,
	found bool,
	requirement CoverageRequirement,
) coverageState {
	if !found {
		return coverageMissing
	}
	if len(requirement.Metrics) == 0 {
		return sampleCoverageState(stream.Coverage.MeasuredSamples, requirement.ExpectedSamples)
	}
	observedAny := false
	complete := true
	for _, metric := range requirement.Metrics {
		count := observationMetricSamples(stream.Coverage.Metrics, metric)
		observedAny = observedAny || count != 0
		complete = complete && count >= requirement.ExpectedSamples
	}
	if complete {
		return coverageObserved
	}
	if observedAny {
		return coverageDegraded
	}
	return coverageMissing
}

func hardwareCoverage(
	stream runrecord.ObservationStream,
	found bool,
	requirement CoverageRequirement,
) coverageState {
	if !found {
		return coverageMissing
	}
	observedHardware := observationKindSamples(stream.Coverage.Kinds, runrecord.ObservationSampleHardware)
	observedRelevant := stream.Coverage.HardwareMeasuredSamples
	state := coverageMissing
	if len(requirement.Metrics) == 0 {
		state = sampleCoverageState(observedRelevant, requirement.ExpectedSamples)
	} else {
		observedAny, complete := false, true
		for _, metric := range requirement.Metrics {
			count := observationMetricSamples(stream.Coverage.HardwareMetrics, metric)
			observedAny = observedAny || count != 0
			complete = complete && count >= requirement.ExpectedSamples
		}
		switch {
		case complete:
			state = coverageObserved
		case observedAny:
			state = coverageDegraded
		}
	}
	if state == coverageMissing && observedHardware != 0 {
		state = coverageDegraded
	}
	if state == coverageObserved && !stream.Scope.Hardware.Valid() {
		return coverageDegraded
	}
	return state
}

func sampleCoverageState(observed, expected uint32) coverageState {
	if observed >= expected {
		return coverageObserved
	}
	if observed != 0 {
		return coverageDegraded
	}
	return coverageMissing
}

func observationMetricSamples(counts []runrecord.ObservationMetricSamples, metric runrecord.ResourceMetric) uint32 {
	index, found := slices.BinarySearchFunc(counts, metric, func(count runrecord.ObservationMetricSamples, target runrecord.ResourceMetric) int {
		return cmp.Compare(count.Metric, target)
	})
	if !found {
		var missing uint32
		return missing
	}
	return counts[index].Samples
}

func observationKindSamples(counts []runrecord.ObservationKindSamples, kind runrecord.ObservationSampleKind) uint32 {
	index, found := slices.BinarySearchFunc(counts, kind, func(count runrecord.ObservationKindSamples, target runrecord.ObservationSampleKind) int {
		return cmp.Compare(count.Kind, target)
	})
	if !found {
		var missing uint32
		return missing
	}
	return counts[index].Samples
}

func loadCoverageUnitFacts(
	ctx context.Context,
	store *overgodb.Store,
	unit CoverageUnit,
	budget *coverageBudget,
) (coverageUnitFacts, error) {
	var facts coverageUnitFacts
	if !unit.Terminal.Valid() {
		return facts, nil
	}
	if err := budget.takeFact(); err != nil {
		return coverageUnitFacts{}, err
	}
	descriptor, found, err := store.Artifact(ctx, unit.Terminal)
	if err != nil {
		return coverageUnitFacts{}, err
	}
	if !found {
		return coverageUnitFacts{}, errors.New("evaluation: evidence coverage terminal content is absent")
	}
	switch {
	case descriptor.MediaType == runrecord.RunMediaType &&
		(descriptor.Schema == runrecord.RunSchema || descriptor.Schema == runrecord.LegacyRunSchema):
		if unit.Terminal != unit.Attempt {
			return coverageUnitFacts{}, errors.New("evaluation: run terminal differs from coverage unit")
		}
		record, requireErr := runrecord.RequireRun(ctx, store, unit.Terminal)
		if requireErr != nil {
			return coverageUnitFacts{}, requireErr
		}
		facts.outcomeKnown, facts.failed, facts.runTerminal = true, record.Outcome == runrecord.OutcomeFailed, true
		if record.Version == artifact.SecondDocumentVersion {
			facts.wallMeasured, facts.wallNS = true, record.MeasuredNS
		}
		facts.run = record
	case descriptor.MediaType == runrecord.AttemptMediaType && descriptor.Schema == runrecord.AttemptSchema:
		if unit.Terminal != unit.Attempt {
			return coverageUnitFacts{}, errors.New("evaluation: attempt terminal differs from coverage unit")
		}
		record, requireErr := runrecord.RequireAttemptRecord(ctx, store, unit.Terminal)
		if requireErr != nil {
			return coverageUnitFacts{}, requireErr
		}
		facts.outcomeKnown, facts.failed, facts.causal = true, record.Outcome == runrecord.OutcomeFailed, record.Causal
		facts.causalAuthoritative = true
		facts.costed, facts.costUnits = record.CostUnits != 0, record.CostUnits
		facts.pair = coveragePairContext{
			domain: coveragePairAttempt, task: record.TaskContract, environment: record.Environment,
		}
	case descriptor.MediaType == runrecord.TerminalAttemptReceiptMediaType &&
		descriptor.Schema == runrecord.TerminalAttemptReceiptSchema:
		receipt, requireErr := runrecord.RequireTerminalAttemptReceipt(ctx, store, unit.Terminal)
		if requireErr != nil || receipt.Operation != unit.Attempt {
			return coverageUnitFacts{}, errors.Join(requireErr, errors.New("evaluation: terminal receipt differs from coverage unit"))
		}
		facts = coverageTerminalFacts(receipt)
	default:
		return coverageUnitFacts{}, errors.New("evaluation: unsupported evidence coverage terminal contract")
	}
	return facts, nil
}

func coverageTerminalFacts(receipt runrecord.TerminalAttemptReceipt) coverageUnitFacts {
	return coverageUnitFacts{
		outcomeKnown: true, failed: receipt.Outcome == runrecord.OutcomeFailed,
		recovered:            receipt.Outcome == runrecord.OutcomeRecovered,
		failureObservation:   receipt.FailureObservation,
		failureNormalization: receipt.FailureNormalization,
	}
}

func coverageEvaluationSources(
	ctx context.Context,
	store *overgodb.Store,
	unit CoverageUnit,
	terminalRun runrecord.Run,
	budget *coverageBudget,
) (coverageEvaluationFacts, error) {
	if unit.Attempt.Kind() != artifact.KindRun {
		return coverageEvaluationFacts{}, nil
	}
	needsEvaluation := len(unit.EvaluationEvidence) != 0
	for _, requirement := range unit.Required {
		needsEvaluation = needsEvaluation || requirement.Axis == CoverageEvaluation
	}
	if !needsEvaluation {
		return coverageEvaluationFacts{}, nil
	}
	evaluationID, found, err := artifact.ResolveAlias(ctx, store, runrecord.EvaluationRunAlias(unit.Attempt))
	if err != nil {
		return coverageEvaluationFacts{}, err
	}
	facts := coverageEvaluationFacts{observed: found}
	var evaluation runrecord.Evaluation
	var execution runrecord.Run
	if found {
		if err := budget.takeFact(); err != nil {
			return coverageEvaluationFacts{}, err
		}
		record, requireErr := runrecord.RequireEvaluation(ctx, store, evaluationID)
		if requireErr != nil || record.Run != unit.Attempt {
			return coverageEvaluationFacts{}, errors.Join(requireErr, errors.New("evaluation: coverage evaluation differs from unit"))
		}
		evaluation = record
		if terminalRun.ID == unit.Attempt {
			execution = terminalRun
		} else {
			if err := budget.takeFact(); err != nil {
				return coverageEvaluationFacts{}, err
			}
			execution, requireErr = runrecord.RequireRun(ctx, store, unit.Attempt)
		}
		if requireErr != nil || execution.ID != unit.Attempt || execution.Outcome != runrecord.OutcomeSucceeded ||
			evaluation.Run != execution.ID || evaluation.Recipe != execution.Recipe {
			return coverageEvaluationFacts{}, errors.Join(requireErr, errors.New("evaluation: coverage run differs from exact evaluation"))
		}
	} else if len(unit.EvaluationEvidence) != 0 {
		return coverageEvaluationFacts{}, errors.New("evaluation: selected evidence lacks the unit's exact evaluation")
	}
	facts.causalExecutions = make([]coverageCausalExecution, 0, len(unit.EvaluationEvidence))
	for _, id := range unit.EvaluationEvidence {
		if err := budget.takeFact(); err != nil {
			return coverageEvaluationFacts{}, err
		}
		evidence, requireErr := requireCoverageEvaluationEvidence(ctx, store, id, coverageEvaluationRelevance{
			run: execution, evaluation: evaluation,
		})
		if requireErr != nil {
			return coverageEvaluationFacts{}, errors.Join(requireErr, errors.New("evaluation: coverage evidence differs from unit"))
		}
		pair, pairErr := coverageEvaluationPairContext(ctx, store, evidence, budget)
		if pairErr != nil {
			return coverageEvaluationFacts{}, pairErr
		}
		if facts.pair.valid() && facts.pair != pair {
			return coverageEvaluationFacts{}, errors.New("evaluation: selected evidence comparison contexts differ")
		}
		facts.pair = pair
		facts.causalExecutions = append(facts.causalExecutions, coverageCausalExecution{
			id: evidence.ID, typed: evidence.Causal, requiresTyped: true,
		})
	}
	return facts, nil
}

func coverageEvaluationPairContext(
	ctx context.Context,
	store *overgodb.Store,
	evidence coverageEvaluationAuthority,
	budget *coverageBudget,
) (coveragePairContext, error) {
	if cached, found := budget.plans[evidence.Plan]; found {
		if !cached.matches(evidence) {
			return coveragePairContext{}, errors.New("evaluation: cached coverage plan differs from evidence")
		}
		return cached.pair, nil
	}
	if err := budget.takeFact(); err != nil {
		return coveragePairContext{}, err
	}
	content, found, err := artifact.ReadContent(ctx, store, evidence.Plan)
	if err != nil || !found {
		return coveragePairContext{}, errors.Join(err, errors.New("evaluation: coverage plan content is absent"))
	}
	if err := evaluationPlanContract.ValidateContent(content, evidence.Plan); err != nil {
		return coveragePairContext{}, err
	}
	plan, err := ParsePlan(content.Data)
	if err != nil || plan.identity != evidence.Plan || plan.body.ModelDefinition != evidence.ModelDefinition ||
		plan.body.RuntimeRecipe != evidence.Recipe || plan.body.Dataset != evidence.Dataset ||
		plan.body.Split != evidence.Split || plan.body.Environment != evidence.Environment ||
		plan.body.CodeCommit != evidence.CodeCommit {
		return coveragePairContext{}, errors.Join(err, errors.New("evaluation: coverage plan differs from evidence"))
	}
	context := coveragePairContext{
		domain: coveragePairEvaluation, task: plan.body.CaseProfile,
		dataset: evidence.Dataset, split: evidence.Split, environment: evidence.Environment,
	}
	if !context.valid() {
		return coveragePairContext{}, errors.New("evaluation: incomplete coverage comparison context")
	}
	budget.plans[evidence.Plan] = coveragePlanContext{
		pair: context, modelDefinition: plan.body.ModelDefinition,
		recipe: plan.body.RuntimeRecipe, codeCommit: plan.body.CodeCommit,
	}
	return context, nil
}

func coverageCausality(
	ctx context.Context,
	store *overgodb.Store,
	unit artifact.ID,
	typed *runrecord.CausalContext,
	requiresTyped bool,
	attached []coverageCausalExecution,
	head artifact.CommitID,
	sequence uint64,
	budget *coverageBudget,
) (coverageState, bool, error) {
	executions := slices.Clone(attached)
	if unit.Kind() == artifact.KindEvidence {
		executions = append(executions, coverageCausalExecution{
			id: unit, typed: typed, requiresTyped: requiresTyped,
		})
	}
	sort.Slice(executions, func(i, j int) bool { return artifact.CompareID(executions[i].id, executions[j].id) < 0 })
	executions = slices.CompactFunc(executions, func(left, right coverageCausalExecution) bool {
		return left.id == right.id
	})
	observed := false
	degraded := false
	recovered := false
	var root artifact.ID
	for _, execution := range executions {
		result, err := store.QueryCausality(ctx, overgodb.CausalityQuery{
			Execution: &execution.id, MaxResults: budget.facts,
		})
		if err != nil {
			return coverageMissing, false, err
		}
		if result.Head != head || result.Sequence != sequence ||
			result.ProjectionVersion != overgodb.CausalityProjectionVersion || result.Truncated {
			return coverageMissing, false, errors.New("evaluation: causal coverage query is truncated or stale")
		}
		if err := budget.takeFacts(len(result.Links)); err != nil {
			return coverageMissing, false, err
		}
		links := slices.Clone(result.Links)
		if execution.requiresTyped && execution.typed == nil {
			if len(links) != 0 {
				return coverageMissing, false, errors.New("evaluation: indexed causality lacks typed evidence")
			}
			continue
		}
		var typedLink *artifact.CausalLink
		if execution.typed != nil {
			link, linkErr := execution.typed.Link(execution.id)
			if linkErr != nil {
				return coverageMissing, false, linkErr
			}
			typedLink = &link
			if len(links) == 0 {
				links = append(links, link)
			}
		}
		for _, link := range links {
			if typedLink != nil && !typedLink.Equal(link) {
				return coverageMissing, false, errors.New("evaluation: typed and indexed causality differ")
			}
			if root.Valid() && link.Root != root {
				return coverageMissing, false, errors.New("evaluation: causal evidence roots differ")
			}
			root = link.Root
			trigger := runrecord.CausalTrigger(link.Trigger)
			if _, known := runrecord.TriggerContractFor(trigger); known {
				observed = true
				if trigger == runrecord.TriggerRecovery {
					recovered = true
				}
			} else {
				degraded = true
			}
		}
	}
	if degraded {
		return coverageDegraded, recovered, nil
	}
	if observed {
		return coverageObserved, recovered, nil
	}
	return coverageMissing, recovered, nil
}

func coverageFailure(
	ctx context.Context,
	store *overgodb.Store,
	unit CoverageUnit,
	facts coverageUnitFacts,
	budget *coverageBudget,
) (bool, error) {
	if !facts.runTerminal {
		if !facts.failed || !facts.failureObservation.Valid() {
			return false, nil
		}
		if err := budget.takeFact(); err != nil {
			return false, err
		}
		return coverageFailureClassification(
			ctx, store, facts.failureObservation, facts.failureNormalization, budget,
		)
	}
	if len(unit.FailureObservations) == 0 {
		return false, nil
	}
	if !facts.outcomeKnown {
		return false, errors.New("evaluation: failure observation lacks typed terminal outcome")
	}
	if !facts.failed {
		return false, errors.New("evaluation: failure observation contradicts run outcome")
	}
	classified := true
	for _, id := range unit.FailureObservations {
		if err := budget.takeFact(); err != nil {
			return false, err
		}
		observation, requireErr := runrecord.RequireFailureObservation(ctx, store, id)
		if requireErr != nil || observation.Run != unit.Attempt {
			return false, errors.Join(requireErr, errors.New("evaluation: failure observation differs from unit"))
		}
		current, classifyErr := coverageFailureClassification(
			ctx, store, observation.ID, artifact.ID{}, budget,
		)
		if classifyErr != nil {
			return false, classifyErr
		}
		classified = classified && current
	}
	return classified, nil
}

func coverageFailureClassification(
	ctx context.Context,
	store *overgodb.Store,
	observationID artifact.ID,
	normalizationID artifact.ID,
	budget *coverageBudget,
) (bool, error) {
	observation, err := runrecord.RequireFailureObservation(ctx, store, observationID)
	if err != nil {
		return false, err
	}
	expected := runrecord.NormalizeFailureObservation(observation)
	var normalization runrecord.FailureNormalization
	found := false
	if normalizationID.Valid() {
		normalization, err = runrecord.RequireFailureNormalization(ctx, store, normalizationID)
		found = err == nil
	} else {
		normalization, found, err = runrecord.ResolveFailureNormalization(
			ctx, store, observation.ID, expected.ClassifierVersion,
		)
	}
	if err != nil {
		return false, err
	}
	if !found {
		return false, nil
	}
	if err := budget.takeFact(); err != nil {
		return false, err
	}
	if normalization.Observation != observation.ID {
		return false, errors.New("evaluation: failure normalization changes observation")
	}
	if normalization.ClassifierVersion != expected.ClassifierVersion {
		return false, nil
	}
	if normalization.Cause != expected.Cause || normalization.Rule != expected.Rule {
		return false, errors.New("evaluation: failure normalization differs from current classifier")
	}
	return true, nil
}
