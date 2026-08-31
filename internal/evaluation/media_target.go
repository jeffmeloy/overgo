package evaluation

import (
	"errors"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
)

const (
	mediaTargetPlanMedia    = "application/vnd.overgo.media-target-plan+json"
	mediaTargetPlanSchema   = "overgo/media-target-plan/v1"
	mediaTargetReportMedia  = "application/vnd.overgo.media-target-report+json"
	mediaTargetReportSchema = "overgo/media-target-report/v1"
)

type MediaOracle string

const (
	MediaArtifactIdentity MediaOracle = "artifact-identity"
	MediaDecode           MediaOracle = "decode"
	MediaExtent           MediaOracle = "extent"
	MediaFiniteValues     MediaOracle = "finite-values"
)

type MediaScorer string

const (
	MediaTranscriptScorer  MediaScorer = "transcript"
	MediaSignalScorer      MediaScorer = "signal"
	MediaReconstruction    MediaScorer = "reconstruction"
	MediaIndependentScorer MediaScorer = "independent-verifier"
)

type MediaScorerSpec struct {
	Name      string              `json:"name"`
	Kind      MediaScorer         `json:"kind"`
	Verifier  artifact.ID         `json:"verifier"`
	Unit      string              `json:"unit,omitzero"`
	Direction runrecord.Direction `json:"direction"`
}

type MediaTargetCase struct {
	Record      string              `json:"record"`
	Expected    artifact.ID         `json:"expected"`
	Assumptions []NumericAssumption `json:"assumptions,omitempty"`
}

type MediaTargetSuite struct {
	Oracles []MediaOracle     `json:"oracles"`
	Scorers []MediaScorerSpec `json:"scorers"`
	Cases   []MediaTargetCase `json:"cases"`
}

type MediaTargetPlan struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	View      artifact.ID                      `json:"view"`
	Signature recipecontract.ModalitySignature `json:"signature"`
	Oracles   []MediaOracle                    `json:"oracles"`
	Scorers   []MediaScorerSpec                `json:"scorers"`
	Cases     []MediaTargetCase                `json:"cases"`
}

type MediaOracleResult struct {
	Kind   MediaOracle `json:"kind"`
	Passed bool        `json:"passed"`
	Detail string      `json:"detail,omitzero"`
}

type MediaVerifierResult struct {
	Name     string      `json:"name"`
	Verifier artifact.ID `json:"verifier"`
	Value    float64     `json:"value"`
}

type MediaTargetObservation struct {
	Record    string                `json:"record"`
	Output    artifact.ID           `json:"output"`
	Oracles   []MediaOracleResult   `json:"oracles"`
	Verifiers []MediaVerifierResult `json:"verifiers"`
}

type MediaRecordResult struct {
	Record      string                `json:"record"`
	Expected    artifact.ID           `json:"expected"`
	Output      artifact.ID           `json:"output"`
	Assumptions []NumericAssumption   `json:"assumptions,omitempty"`
	Oracles     []MediaOracleResult   `json:"oracles"`
	Verifiers   []MediaVerifierResult `json:"verifiers"`
}

type MediaTargetReport struct {
	ID      artifact.ID         `json:"-"`
	Version uint16              `json:"version"`
	Plan    artifact.ID         `json:"plan"`
	Records []MediaRecordResult `json:"records"`
	Metrics []runrecord.Metric  `json:"metrics"`
}

var (
	mediaTargetPlanCodec = artifact.JSONDocumentCodec(
		"media target plan", artifact.KindProfile, mediaTargetPlanMedia, mediaTargetPlanSchema,
		canonicalizeMediaTargetPlan, func(value MediaTargetPlan) artifact.ID { return value.ID },
		func(value *MediaTargetPlan, id artifact.ID) { value.ID = id }, cloneMediaTargetPlan,
	)
	mediaTargetReportCodec = artifact.JSONDocumentCodec(
		"media target report", artifact.KindEvaluation, mediaTargetReportMedia, mediaTargetReportSchema,
		canonicalizeMediaTargetReport, func(value MediaTargetReport) artifact.ID { return value.ID },
		func(value *MediaTargetReport, id artifact.ID) { value.ID = id }, cloneMediaTargetReport,
	)
)

// CompileMediaTargetPlan binds a media suite to a held-out dataset view.
func CompileMediaTargetPlan(view SFTEvaluationView, suite MediaTargetSuite) (MediaTargetPlan, error) {
	if _, err := view.Content(); err != nil {
		return MediaTargetPlan{}, err
	}
	if len(suite.Cases) != len(view.Records) {
		return MediaTargetPlan{}, errors.New("evaluation: media target cases differ from held-out view")
	}
	records := make(map[string]struct{}, len(view.Records))
	for _, record := range view.Records {
		records[record.ID] = struct{}{}
	}
	for _, target := range suite.Cases {
		if _, found := records[target.Record]; !found {
			return MediaTargetPlan{}, errors.New("evaluation: media target record is outside held-out view")
		}
	}
	return mediaTargetPlanCodec.NewInitial(MediaTargetPlan{
		View: view.ID, Signature: view.Signature.Clone(),
		Oracles: slices.Clone(suite.Oracles), Scorers: slices.Clone(suite.Scorers), Cases: cloneMediaCases(suite.Cases),
	})
}

func (plan MediaTargetPlan) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{plan.View}
	for _, scorer := range plan.Scorers {
		parents = append(parents, scorer.Verifier)
	}
	for _, target := range plan.Cases {
		parents = append(parents, target.Expected)
	}
	return mediaTargetPlanCodec.Batch(key, plan, artifact.DependencyLineage(plan.ID, parents...), nil)
}

// ScoreMediaTargets verifies media observations against a compiled plan.
func ScoreMediaTargets(plan MediaTargetPlan, observations []MediaTargetObservation) (MediaTargetReport, error) {
	if err := mediaTargetPlanCodec.ValidateIdentity(plan); err != nil || len(observations) != len(plan.Cases) {
		return MediaTargetReport{}, errors.New("evaluation: media target observations differ from plan")
	}
	observations = cloneMediaObservations(observations)
	sort.Slice(observations, func(i, j int) bool { return observations[i].Record < observations[j].Record })
	report := MediaTargetReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.ID,
		Records: make([]MediaRecordResult, len(plan.Cases)), Metrics: make([]runrecord.Metric, len(plan.Scorers)),
	}
	for index, target := range plan.Cases {
		observation := &observations[index]
		if observation.Record != target.Record || observation.Output.Kind() != artifact.KindOutput ||
			len(observation.Oracles) != len(plan.Oracles) || len(observation.Verifiers) != len(plan.Scorers) {
			return MediaTargetReport{}, errors.New("evaluation: media result differs from plan")
		}
		sort.Slice(observation.Oracles, func(i, j int) bool { return observation.Oracles[i].Kind < observation.Oracles[j].Kind })
		sort.Slice(observation.Verifiers, func(i, j int) bool { return observation.Verifiers[i].Name < observation.Verifiers[j].Name })
		for oracleIndex, oracle := range plan.Oracles {
			if observation.Oracles[oracleIndex].Kind != oracle || !observation.Oracles[oracleIndex].Passed {
				return MediaTargetReport{}, errors.New("evaluation: deterministic media oracle failed")
			}
		}
		for scorerIndex, scorer := range plan.Scorers {
			result := observation.Verifiers[scorerIndex]
			if result.Name != scorer.Name || result.Verifier != scorer.Verifier || !finite(result.Value) {
				return MediaTargetReport{}, errors.New("evaluation: media verifier differs from plan")
			}
			report.Metrics[scorerIndex].Value += result.Value
		}
		report.Records[index] = MediaRecordResult{
			Record: target.Record, Expected: target.Expected, Output: observation.Output,
			Assumptions: slices.Clone(target.Assumptions), Oracles: slices.Clone(observation.Oracles),
			Verifiers: slices.Clone(observation.Verifiers),
		}
	}
	for index, scorer := range plan.Scorers {
		report.Metrics[index] = runrecord.Metric{
			Name: scorer.Name, Value: report.Metrics[index].Value / float64(len(report.Records)),
			Unit: scorer.Unit, Direction: scorer.Direction,
		}
	}
	return mediaTargetReportCodec.New(report)
}

func (report MediaTargetReport) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{report.Plan}
	for _, record := range report.Records {
		parents = append(parents, record.Output)
	}
	return mediaTargetReportCodec.Batch(key, report, artifact.DependencyLineage(report.ID, parents...), nil)
}

func canonicalizeMediaTargetPlan(plan *MediaTargetPlan) error {
	if plan == nil || plan.Version != artifact.InitialDocumentVersion || plan.View.Kind() != artifact.KindProfile ||
		plan.Signature.Validate() != nil || len(plan.Signature.Outputs) != 1 || len(plan.Oracles) == 0 ||
		len(plan.Scorers) == 0 || len(plan.Cases) == 0 {
		return errors.New("evaluation: invalid media target plan")
	}
	output := plan.Signature.Outputs[0]
	if output != recipecontract.ModalityAudio && output != recipecontract.ModalityImage && output != recipecontract.ModalityVideo {
		return errors.New("evaluation: unsupported media target modality")
	}
	slices.Sort(plan.Oracles)
	for index, oracle := range plan.Oracles {
		if !validMediaOracle(oracle) || index > 0 && plan.Oracles[index-1] == oracle {
			return errors.New("evaluation: invalid or duplicate media oracle")
		}
	}
	sort.Slice(plan.Scorers, func(i, j int) bool { return plan.Scorers[i].Name < plan.Scorers[j].Name })
	for index, scorer := range plan.Scorers {
		if scorer.Name == "" || scorer.Verifier.Kind() != artifact.KindProfile || !validMediaScorer(scorer.Kind) ||
			!validMetricDirection(scorer.Direction) || index > 0 && plan.Scorers[index-1].Name == scorer.Name {
			return errors.New("evaluation: invalid or duplicate media scorer")
		}
	}
	sort.Slice(plan.Cases, func(i, j int) bool { return plan.Cases[i].Record < plan.Cases[j].Record })
	for index := range plan.Cases {
		target := &plan.Cases[index]
		sort.Slice(target.Assumptions, func(i, j int) bool { return target.Assumptions[i].Name < target.Assumptions[j].Name })
		if target.Record == "" || target.Expected.Kind() != artifact.KindOutput ||
			index > 0 && plan.Cases[index-1].Record == target.Record || !validAssumptions(target.Assumptions) {
			return errors.New("evaluation: invalid media target case")
		}
	}
	return nil
}

func canonicalizeMediaTargetReport(report *MediaTargetReport) error {
	if report == nil || report.Version != artifact.InitialDocumentVersion || report.Plan.Kind() != artifact.KindProfile ||
		len(report.Records) == 0 || len(report.Metrics) == 0 {
		return errors.New("evaluation: invalid media target report")
	}
	for index, record := range report.Records {
		if record.Record == "" || record.Expected.Kind() != artifact.KindOutput || record.Output.Kind() != artifact.KindOutput ||
			index > 0 && report.Records[index-1].Record >= record.Record {
			return errors.New("evaluation: invalid media target result")
		}
	}
	for index, metric := range report.Metrics {
		if metric.Name == "" || !finite(metric.Value) || !validMetricDirection(metric.Direction) ||
			index > 0 && report.Metrics[index-1].Name >= metric.Name {
			return errors.New("evaluation: invalid media target metric")
		}
	}
	return nil
}

func validMediaOracle(value MediaOracle) bool {
	return value == MediaArtifactIdentity || value == MediaDecode || value == MediaExtent || value == MediaFiniteValues
}

func validMediaScorer(value MediaScorer) bool {
	return value == MediaTranscriptScorer || value == MediaSignalScorer || value == MediaReconstruction || value == MediaIndependentScorer
}

func validMetricDirection(value runrecord.Direction) bool {
	return value == runrecord.DirectionMinimize || value == runrecord.DirectionMaximize
}

func validAssumptions(values []NumericAssumption) bool {
	for index, assumption := range values {
		if assumption.Name == "" || assumption.Value == "" || index > 0 && values[index-1].Name == assumption.Name {
			return false
		}
	}
	return true
}

func cloneMediaCases(values []MediaTargetCase) []MediaTargetCase {
	result := slices.Clone(values)
	for index := range result {
		result[index].Assumptions = slices.Clone(result[index].Assumptions)
	}
	return result
}

func cloneMediaTargetPlan(plan MediaTargetPlan) MediaTargetPlan {
	plan.Signature = plan.Signature.Clone()
	plan.Oracles = slices.Clone(plan.Oracles)
	plan.Scorers = slices.Clone(plan.Scorers)
	plan.Cases = cloneMediaCases(plan.Cases)
	return plan
}

func cloneMediaObservations(values []MediaTargetObservation) []MediaTargetObservation {
	result := slices.Clone(values)
	for index := range result {
		result[index].Oracles = slices.Clone(result[index].Oracles)
		result[index].Verifiers = slices.Clone(result[index].Verifiers)
	}
	return result
}

func cloneMediaTargetReport(report MediaTargetReport) MediaTargetReport {
	report.Records = slices.Clone(report.Records)
	for index := range report.Records {
		report.Records[index].Assumptions = slices.Clone(report.Records[index].Assumptions)
		report.Records[index].Oracles = slices.Clone(report.Records[index].Oracles)
		report.Records[index].Verifiers = slices.Clone(report.Records[index].Verifiers)
	}
	report.Metrics = slices.Clone(report.Metrics)
	return report
}
