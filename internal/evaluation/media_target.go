package evaluation

import (
	"cmp"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/recipecontract"
	"overgo/internal/runrecord"
)

const (
	mediaTargetPlanMedia    = "application/vnd.overgo.media-target-plan+json"
	mediaTargetPlanSchema   = "overgo/media-target-plan/v1"
	mediaTargetReportMedia  = "application/vnd.overgo.media-target-report+json"
	mediaTargetReportSchema = "overgo/media-target-report/v1"
)

// mediaOracle names a deterministic pass/fail check every media output must pass.
type mediaOracle string

const (
	// mediaArtifactIdentity checks the output artifact identity.
	mediaArtifactIdentity mediaOracle = "artifact-identity"
	// mediaDecode checks the output decodes.
	mediaDecode mediaOracle = "decode"
	// mediaExtent checks the output extent.
	mediaExtent mediaOracle = "extent"
	// mediaFiniteValues checks the output holds only finite values.
	mediaFiniteValues mediaOracle = "finite-values"
)

// mediaScorer names the kind of verifier-backed media metric.
type mediaScorer string

const (
	// mediaTranscriptScorer scores a transcript of the media output.
	mediaTranscriptScorer mediaScorer = "transcript"
	// mediaSignalScorer scores the output signal.
	mediaSignalScorer mediaScorer = "signal"
	// mediaReconstruction scores reconstruction of the expected output.
	mediaReconstruction mediaScorer = "reconstruction"
	// mediaIndependentScorer scores through an independent verifier.
	mediaIndependentScorer mediaScorer = "independent-verifier"
)

// mediaScorerSpec declares one named media metric, its verifier profile, unit, and direction.
type mediaScorerSpec struct {
	Name      string              `json:"name"`
	Kind      mediaScorer         `json:"kind"`
	Verifier  artifact.ID         `json:"verifier"`
	Unit      string              `json:"unit,omitzero"`
	Direction runrecord.Direction `json:"direction"`
}

// mediaTargetCase binds a held-out record to its expected output artifact and stated assumptions.
type mediaTargetCase struct {
	Record      string              `json:"record"`
	Expected    artifact.ID         `json:"expected"`
	Assumptions []numericAssumption `json:"assumptions,omitempty"`
}

// mediaTargetSuite declares oracles, scorers, and cases before binding to a held-out view.
type mediaTargetSuite struct {
	Oracles []mediaOracle     `json:"oracles"`
	Scorers []mediaScorerSpec `json:"scorers"`
	Cases   []mediaTargetCase `json:"cases"`
}

// mediaTargetPlan binds a media suite to one held-out view and its audio, image, or video output signature.
type mediaTargetPlan struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	View      artifact.ID                      `json:"view"`
	Signature recipecontract.ModalitySignature `json:"signature"`
	Oracles   []mediaOracle                    `json:"oracles"`
	Scorers   []mediaScorerSpec                `json:"scorers"`
	Cases     []mediaTargetCase                `json:"cases"`
}

// mediaOracleResult records one oracle outcome for an observed media output.
type mediaOracleResult struct {
	Kind   mediaOracle `json:"kind"`
	Passed bool        `json:"passed"`
	Detail string      `json:"detail,omitzero"`
}

// mediaVerifierResult records one named scorer value from its verifier.
type mediaVerifierResult struct {
	Name     string      `json:"name"`
	Verifier artifact.ID `json:"verifier"`
	Value    float64     `json:"value"`
}

// mediaTargetObservation carries one record's output artifact with its oracle and verifier results.
type mediaTargetObservation struct {
	Record    string                `json:"record"`
	Output    artifact.ID           `json:"output"`
	Oracles   []mediaOracleResult   `json:"oracles"`
	Verifiers []mediaVerifierResult `json:"verifiers"`
}

// mediaRecordResult pairs a case's expected output and assumptions with its observed output and results.
type mediaRecordResult struct {
	Record      string                `json:"record"`
	Expected    artifact.ID           `json:"expected"`
	Output      artifact.ID           `json:"output"`
	Assumptions []numericAssumption   `json:"assumptions,omitempty"`
	Oracles     []mediaOracleResult   `json:"oracles"`
	Verifiers   []mediaVerifierResult `json:"verifiers"`
}

// mediaTargetReport holds per-record media results and per-scorer mean metrics for one plan.
type mediaTargetReport struct {
	ID      artifact.ID         `json:"-"`
	Version uint16              `json:"version"`
	Plan    artifact.ID         `json:"plan"`
	Records []mediaRecordResult `json:"records"`
	Metrics []runrecord.Metric  `json:"metrics"`
}

var (
	mediaTargetPlanCodec = artifact.JSONDocumentCodec(
		"media target plan", artifact.KindProfile, mediaTargetPlanMedia, mediaTargetPlanSchema,
		canonicalizeMediaTargetPlan, func(value mediaTargetPlan) artifact.ID { return value.ID },
		func(value *mediaTargetPlan, id artifact.ID) { value.ID = id }, cloneMediaTargetPlan,
	)
	mediaTargetReportCodec = artifact.JSONDocumentCodec(
		"media target report", artifact.KindEvaluation, mediaTargetReportMedia, mediaTargetReportSchema,
		canonicalizeMediaTargetReport, func(value mediaTargetReport) artifact.ID { return value.ID },
		func(value *mediaTargetReport, id artifact.ID) { value.ID = id }, cloneMediaTargetReport,
	)
)

// compileMediaTargetPlan binds a media suite to a held-out dataset view.
func compileMediaTargetPlan(view SFTEvaluationView, suite mediaTargetSuite) (mediaTargetPlan, error) {
	if _, err := view.Content(); err != nil {
		return mediaTargetPlan{}, err
	}
	if len(suite.Cases) != len(view.Records) {
		return mediaTargetPlan{}, errors.New("evaluation: media target cases differ from held-out view")
	}
	records := make(map[string]struct{}, len(view.Records))
	for _, record := range view.Records {
		records[record.ID] = struct{}{}
	}
	for _, target := range suite.Cases {
		if _, found := records[target.Record]; !found {
			return mediaTargetPlan{}, errors.New("evaluation: media target record is outside held-out view")
		}
	}
	return mediaTargetPlanCodec.NewInitial(mediaTargetPlan{
		View: view.ID, Signature: view.Signature.Clone(),
		Oracles: slices.Clone(suite.Oracles), Scorers: slices.Clone(suite.Scorers), Cases: cloneMediaCases(suite.Cases),
	})
}

// Batch stages the plan with lineage to its view, verifiers, and expected outputs.
func (plan mediaTargetPlan) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{plan.View}
	for _, scorer := range plan.Scorers {
		parents = append(parents, scorer.Verifier)
	}
	for _, target := range plan.Cases {
		parents = append(parents, target.Expected)
	}
	return mediaTargetPlanCodec.Batch(key, plan, artifact.DependencyLineage(plan.ID, parents...), nil)
}

// scoreMediaTargets verifies media observations against a compiled plan.
func scoreMediaTargets(plan mediaTargetPlan, observations []mediaTargetObservation) (mediaTargetReport, error) {
	if err := mediaTargetPlanCodec.ValidateIdentity(plan); err != nil || len(observations) != len(plan.Cases) {
		return mediaTargetReport{}, errors.New("evaluation: media target observations differ from plan")
	}
	observations = cloneMediaObservations(observations)
	slices.SortFunc(observations, func(a, b mediaTargetObservation) int { return cmp.Compare(a.Record, b.Record) })
	report := mediaTargetReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.ID,
		Records: make([]mediaRecordResult, len(plan.Cases)), Metrics: make([]runrecord.Metric, len(plan.Scorers)),
	}
	for index, target := range plan.Cases {
		observation := &observations[index]
		if observation.Record != target.Record || observation.Output.Kind() != artifact.KindOutput ||
			len(observation.Oracles) != len(plan.Oracles) || len(observation.Verifiers) != len(plan.Scorers) {
			return mediaTargetReport{}, errors.New("evaluation: media result differs from plan")
		}
		slices.SortFunc(observation.Oracles, func(a, b mediaOracleResult) int { return cmp.Compare(a.Kind, b.Kind) })
		slices.SortFunc(observation.Verifiers, func(a, b mediaVerifierResult) int { return cmp.Compare(a.Name, b.Name) })
		for oracleIndex, oracle := range plan.Oracles {
			if observation.Oracles[oracleIndex].Kind != oracle || !observation.Oracles[oracleIndex].Passed {
				return mediaTargetReport{}, errors.New("evaluation: deterministic media oracle failed")
			}
		}
		for scorerIndex, scorer := range plan.Scorers {
			result := observation.Verifiers[scorerIndex]
			if result.Name != scorer.Name || result.Verifier != scorer.Verifier || !checked.Finite64(result.Value) {
				return mediaTargetReport{}, errors.New("evaluation: media verifier differs from plan")
			}
			report.Metrics[scorerIndex].Value += result.Value
		}
		report.Records[index] = mediaRecordResult{
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

// Batch stages the report with lineage to its plan and observed outputs.
func (report mediaTargetReport) Batch(key string) (artifact.Batch, error) {
	parents := []artifact.ID{report.Plan}
	for _, record := range report.Records {
		parents = append(parents, record.Output)
	}
	return mediaTargetReportCodec.Batch(key, report, artifact.DependencyLineage(report.ID, parents...), nil)
}

func canonicalizeMediaTargetPlan(plan *mediaTargetPlan) error {
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
	slices.SortFunc(plan.Scorers, func(a, b mediaScorerSpec) int { return cmp.Compare(a.Name, b.Name) })
	for index, scorer := range plan.Scorers {
		if scorer.Name == "" || scorer.Verifier.Kind() != artifact.KindProfile || !validMediaScorer(scorer.Kind) ||
			!validMetricDirection(scorer.Direction) || index > 0 && plan.Scorers[index-1].Name == scorer.Name {
			return errors.New("evaluation: invalid or duplicate media scorer")
		}
	}
	slices.SortFunc(plan.Cases, func(a, b mediaTargetCase) int { return cmp.Compare(a.Record, b.Record) })
	for index := range plan.Cases {
		target := &plan.Cases[index]
		slices.SortFunc(target.Assumptions, func(a, b numericAssumption) int { return cmp.Compare(a.Name, b.Name) })
		if target.Record == "" || target.Expected.Kind() != artifact.KindOutput ||
			index > 0 && plan.Cases[index-1].Record == target.Record || !validAssumptions(target.Assumptions) {
			return errors.New("evaluation: invalid media target case")
		}
	}
	return nil
}

func canonicalizeMediaTargetReport(report *mediaTargetReport) error {
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
		if metric.Name == "" || !checked.Finite64(metric.Value) || !validMetricDirection(metric.Direction) ||
			index > 0 && report.Metrics[index-1].Name >= metric.Name {
			return errors.New("evaluation: invalid media target metric")
		}
	}
	return nil
}

func validMediaOracle(value mediaOracle) bool {
	return value == mediaArtifactIdentity || value == mediaDecode || value == mediaExtent || value == mediaFiniteValues
}

func validMediaScorer(value mediaScorer) bool {
	return value == mediaTranscriptScorer || value == mediaSignalScorer || value == mediaReconstruction || value == mediaIndependentScorer
}

func validAssumptions(values []numericAssumption) bool {
	for index, assumption := range values {
		if assumption.Name == "" || assumption.Value == "" || index > 0 && values[index-1].Name == assumption.Name {
			return false
		}
	}
	return true
}

func cloneMediaCases(values []mediaTargetCase) []mediaTargetCase {
	result := slices.Clone(values)
	for index := range result {
		result[index].Assumptions = slices.Clone(result[index].Assumptions)
	}
	return result
}

func cloneMediaTargetPlan(plan mediaTargetPlan) mediaTargetPlan {
	plan.Signature = plan.Signature.Clone()
	plan.Oracles = slices.Clone(plan.Oracles)
	plan.Scorers = slices.Clone(plan.Scorers)
	plan.Cases = cloneMediaCases(plan.Cases)
	return plan
}

func cloneMediaObservations(values []mediaTargetObservation) []mediaTargetObservation {
	result := slices.Clone(values)
	for index := range result {
		result[index].Oracles = slices.Clone(result[index].Oracles)
		result[index].Verifiers = slices.Clone(result[index].Verifiers)
	}
	return result
}

func cloneMediaTargetReport(report mediaTargetReport) mediaTargetReport {
	report.Records = slices.Clone(report.Records)
	for index := range report.Records {
		report.Records[index].Assumptions = slices.Clone(report.Records[index].Assumptions)
		report.Records[index].Oracles = slices.Clone(report.Records[index].Oracles)
		report.Records[index].Verifiers = slices.Clone(report.Records[index].Verifiers)
	}
	report.Metrics = slices.Clone(report.Metrics)
	return report
}
