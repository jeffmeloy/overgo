package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

const (
	heldoutVerdictMediaType = "application/vnd.overgo.heldout-verdict+json"
	// heldoutVerdictSchema's second version binds the trained output and the
	// exact evaluation inputs; a first-version verdict names neither, so it
	// identifies no trained result and is not read.
	heldoutVerdictSchema = "overgo/heldout-verdict/v2"
)

var heldoutVerdictCodec = artifact.JSONDocumentCodec(
	"held-out verdict", artifact.KindEvaluation, heldoutVerdictMediaType, heldoutVerdictSchema,
	canonicalizeHeldoutVerdict, func(value heldoutVerdict) artifact.ID { return value.ID },
	func(value *heldoutVerdict, id artifact.ID) { value.ID = id }, func(value heldoutVerdict) heldoutVerdict { return value },
)

// heldoutNumericScorers maps an objective's declared metric to the numeric
// scorer that measures it on held-out records.
var heldoutNumericScorers = map[trainingprogram.EvaluationMetric]numericScorer{
	trainingprogram.MetricForecastMAE:   numericAbsoluteError,
	trainingprogram.MetricTableAccuracy: numericLabelAccuracy,
	trainingprogram.MetricTableMSE:      numericSquaredError,
	trainingprogram.MetricTokenAccuracy: numericLabelAccuracy,
}

// heldoutTextScorers maps an objective's declared metric to the text scorer
// that measures it on generated text. A generated answer is trimmed of
// surrounding space before it is matched.
var heldoutTextScorers = map[trainingprogram.EvaluationMetric]textScorerSpec{
	trainingprogram.MetricGeneratedAnswer: {Kind: textGeneratedAnswer, Normalization: []textNormalization{textTrimSpaceNormalization}},
}

// heldoutSignalMetrics are the declared metrics a signal verifier measures
// on a produced media signal against the expected one.
var heldoutSignalMetrics = []trainingprogram.EvaluationMetric{trainingprogram.MetricAudioSNR}

// HeldoutCase is one held-out record's expected answer as the objective's
// metric reads it: target values or a label for a numeric metric, the
// expected signal for a media one, accepted answers for generated text. A
// preference pair's expected answer is its chosen response, which its
// observation's scores already carry.
type HeldoutCase struct {
	Record  string
	Values  []float64
	Label   string
	Answers []string
}

// HeldoutObservation is one model's answer for a held-out record: predicted
// values or a label, a produced signal, generated text, a preference pair's
// sequence scores under the model as policy and under its reference, or an
// unpaired completion's, whose mismatched scores follow its prompt with
// the completion of the record Partner names.
type HeldoutObservation struct {
	Record     string
	Values     []float64
	Label      string
	Text       string
	Preference trainingprogram.PreferenceScores
	Partner    string
	Unpaired   trainingprogram.UnpairedScores
}

// HeldoutPlan measures an objective's declared metric on its held-out view
// through the target plan of the metric's kind: numeric targets, generated
// text, a media signal or preference pairs. The judge reads every kind the
// same way, as the view a plan binds and the reports it scores.
type HeldoutPlan struct {
	id, view artifact.ID
	batch    func(string) (artifact.Batch, error)
	score    func([]HeldoutObservation) (HeldoutReport, error)
}

// HeldoutReport is one model's scores under a held-out plan.
type HeldoutReport struct {
	id, plan artifact.ID
	metrics  []runrecord.Metric
	batch    func(string) (artifact.Batch, error)
}

// CompileHeldoutPlan binds the target plan that measures the objective's
// declared metric to its held-out view, one case per view record. A
// preference objective compiles through CompilePreferenceHeldoutPlan, or
// CompileUnpairedHeldoutPlan for unpaired completions, since its records
// are scored against a reference rather than an expected answer.
func CompileHeldoutPlan(objective trainingprogram.ObjectiveDocument, view SFTEvaluationView, cases []HeldoutCase) (HeldoutPlan, error) {
	if view.Objective != objective.ID {
		return HeldoutPlan{}, errors.New("evaluation: held-out view binds another objective")
	}
	if scorer, found := heldoutNumericScorers[objective.Metric]; found {
		return compileNumericHeldout(view, scorer, cases)
	}
	if scorer, found := heldoutTextScorers[objective.Metric]; found {
		return compileTextHeldout(view, scorer, cases)
	}
	if slices.Contains(heldoutSignalMetrics, objective.Metric) {
		return compileSignalHeldout(view, objective.Metric, cases)
	}
	return HeldoutPlan{}, fmt.Errorf("evaluation: no held-out scorer measures %q", objective.Metric)
}

// ID names the compiled target plan.
func (plan HeldoutPlan) ID() artifact.ID { return plan.id }

// Batch stages the target plan with its lineage.
func (plan HeldoutPlan) Batch(key string) (artifact.Batch, error) {
	if plan.batch == nil {
		return artifact.Batch{}, errors.New("evaluation: held-out plan is not compiled")
	}
	return plan.batch(key)
}

// Score scores one model's observations, one per plan case, into its report.
func (plan HeldoutPlan) Score(observations []HeldoutObservation) (HeldoutReport, error) {
	if plan.score == nil {
		return HeldoutReport{}, errors.New("evaluation: held-out plan is not compiled")
	}
	return plan.score(observations)
}

// ID names the scored report.
func (report HeldoutReport) ID() artifact.ID { return report.id }

// Batch stages the report with its lineage.
func (report HeldoutReport) Batch(key string) (artifact.Batch, error) {
	if report.batch == nil {
		return artifact.Batch{}, errors.New("evaluation: held-out report is not scored")
	}
	return report.batch(key)
}

// heldoutMetricName names the report metric that measures an objective's
// declared metric, if a held-out plan measures it.
func heldoutMetricName(metric trainingprogram.EvaluationMetric) (string, bool) {
	if scorer, found := heldoutNumericScorers[metric]; found {
		return string(scorer), true
	}
	if scorer, found := heldoutTextScorers[metric]; found {
		return string(scorer.Kind), true
	}
	if slices.Contains(heldoutSignalMetrics, metric) {
		return string(metric), true
	}
	if metric == trainingprogram.MetricPreferenceLoss {
		return preferenceLossMetric, true
	}
	if metric == trainingprogram.MetricUnpairedPreferenceLoss {
		return unpairedLossMetric, true
	}
	return "", false
}

// HeldoutTraining is what a held-out verdict binds of the training it
// judges: the observed training session, the model that session trained,
// the checkpoint the trained model published, and the held-out membership
// as the judged models read it, the exact evaluation inputs.
type HeldoutTraining struct {
	Observation, Model artifact.ID
	Trained            trainingprogram.Checkpoint
	Inputs             artifact.ID
}

// heldoutVerdict judges a trained model against its base model on one
// objective's held-out view: both are scored by the same target plan, and
// the verdict passes when the trained model strictly improves the metric
// the objective declares, in the metric's own direction. No threshold is
// involved; Passed is recomputed from the recorded values. Trained names the
// checkpoint the trained model published and Inputs the held-out membership
// as the judged models read it.
type heldoutVerdict struct {
	ID             artifact.ID                      `json:"-"`
	Version        uint16                           `json:"version"`
	Objective      artifact.ID                      `json:"objective"`
	View           artifact.ID                      `json:"view"`
	Plan           artifact.ID                      `json:"plan"`
	Observation    artifact.ID                      `json:"observation"`
	Trained        artifact.ID                      `json:"trained"`
	Inputs         artifact.ID                      `json:"inputs"`
	Baseline       artifact.ID                      `json:"baseline"`
	Candidate      artifact.ID                      `json:"candidate"`
	Metric         trainingprogram.EvaluationMetric `json:"metric"`
	Direction      runrecord.Direction              `json:"direction"`
	BaselineValue  float64                          `json:"baseline_value"`
	CandidateValue float64                          `json:"candidate_value"`
	Passed         bool                             `json:"passed"`
}

// JudgeHeldout compares the base model's and the trained model's reports on
// one plan bound to the objective's held-out view. The trained checkpoint
// must have trained the observed model on exactly the view's training
// membership of the objective's dataset.
func JudgeHeldout(
	objective trainingprogram.ObjectiveDocument,
	view SFTEvaluationView,
	plan HeldoutPlan,
	baseline, candidate HeldoutReport,
	training HeldoutTraining,
) (heldoutVerdict, error) {
	if _, err := objective.Content(); err != nil {
		return heldoutVerdict{}, err
	}
	if err := sftEvaluationViewCodec.ValidateIdentity(view); err != nil {
		return heldoutVerdict{}, err
	}
	if plan.batch == nil || baseline.batch == nil || candidate.batch == nil {
		return heldoutVerdict{}, errors.New("evaluation: held-out plan or reports are not scored")
	}
	if view.Objective != objective.ID || plan.view != view.ID || baseline.plan != plan.id || candidate.plan != plan.id {
		return heldoutVerdict{}, errors.New("evaluation: held-out reports are not bound to this objective's view")
	}
	trained := training.Trained
	if err := trained.ValidateIdentity(); err != nil {
		return heldoutVerdict{}, err
	}
	if trained.Model != training.Model || trained.Dataset != objective.Dataset || trained.Split != view.TrainingMembership {
		return heldoutVerdict{}, fmt.Errorf("evaluation: checkpoint %s did not train the observed model %s on the objective's training membership %s",
			trained.ID(), training.Model, view.TrainingMembership)
	}
	name, found := heldoutMetricName(objective.Metric)
	if !found {
		return heldoutVerdict{}, fmt.Errorf("evaluation: no held-out scorer measures %q", objective.Metric)
	}
	baseMetric, baseFound := reportMetric(baseline.metrics, name)
	candidateMetric, candidateFound := reportMetric(candidate.metrics, name)
	if !baseFound || !candidateFound || baseMetric.Direction != candidateMetric.Direction {
		return heldoutVerdict{}, fmt.Errorf("evaluation: held-out reports do not both carry %q", name)
	}
	return heldoutVerdictCodec.New(heldoutVerdict{
		Version: artifact.SecondDocumentVersion, Objective: objective.ID, View: view.ID, Plan: plan.id,
		Observation: training.Observation, Trained: trained.ID(), Inputs: training.Inputs,
		Baseline: baseline.id, Candidate: candidate.id, Metric: objective.Metric,
		Direction: baseMetric.Direction, BaselineValue: baseMetric.Value, CandidateValue: candidateMetric.Value,
	})
}

// Content encodes the verdict as its canonical artifact document.
func (verdict heldoutVerdict) Content() (artifact.Content, error) {
	return heldoutVerdictCodec.Content(verdict)
}

// Batch stages the verdict with lineage to the objective, view, plan,
// training observation, trained checkpoint, evaluation inputs and both
// reports it judged; the inputs' identity is declared beside it.
func (verdict heldoutVerdict) Batch(key string) (artifact.Batch, error) {
	return declaring(func(key string) (artifact.Batch, error) {
		return heldoutVerdictCodec.Batch(key, verdict, artifact.DependencyLineage(
			verdict.ID, verdict.Objective, verdict.View, verdict.Plan, verdict.Observation, verdict.Trained, verdict.Inputs,
			verdict.Baseline, verdict.Candidate,
		), nil)
	}, key, []artifact.Descriptor{{ID: verdict.Inputs}})
}

// HeldoutBatch stages the view, plan, both reports and the verdict judged
// from them as one publication. An artifact several parts declare, such as
// an output two models both produced, is declared once.
func HeldoutBatch(view SFTEvaluationView, plan HeldoutPlan, baseline, candidate HeldoutReport, verdict heldoutVerdict) (artifact.Batch, error) {
	batch := artifact.Batch{Key: "training/heldout/" + verdict.ID.String()}
	for _, stage := range []func(string) (artifact.Batch, error){view.Batch, plan.Batch, baseline.Batch, candidate.Batch, verdict.Batch} {
		part, err := stage(batch.Key)
		if err != nil {
			return artifact.Batch{}, err
		}
		for _, declared := range part.Artifacts {
			if !slices.Contains(batch.Artifacts, declared) {
				batch.Artifacts = append(batch.Artifacts, declared)
			}
		}
		batch.Contents = append(batch.Contents, part.Contents...)
		batch.Lineage = append(batch.Lineage, part.Lineage...)
	}
	return batch, nil
}

// CommittedHeldoutVerdictPassed reads one committed held-out verdict and
// reports whether it passed for current, re-deriving everything it claims:
// the objective it judged declares current's contract, its view recompiles
// from the committed training membership and the held-out one its published
// split pairs it with, each report rescores to itself from the cases it
// retains, its trained checkpoint is committed and trained the model its
// observed session trained, its reports recompute to the same verdict, and
// its training observation is evidence current already carries.
func CommittedHeldoutVerdictPassed(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
	current trainingprogram.ObjectiveDocument,
) (bool, error) {
	verdict, err := heldoutVerdictCodec.Require(ctx, reader, id)
	if err != nil {
		return false, err
	}
	objective, err := trainingprogram.LoadObjective(ctx, reader, verdict.Objective)
	if err != nil {
		return false, err
	}
	if !trainingprogram.SameContract(objective, current) {
		return false, fmt.Errorf("evaluation: verdict %s judged a different objective contract", id)
	}
	if err := reconstructHeldoutVerdict(ctx, reader, verdict, objective); err != nil {
		return false, err
	}
	if !slices.Contains(current.Evidence, verdict.Observation) {
		return false, fmt.Errorf("evaluation: verdict %s training observation is not evidence of this objective", id)
	}
	return verdict.Passed, nil
}

// reconstructHeldoutVerdict is the shared evidence check for approval and
// reporting. It verifies retained inputs and reports, including valid losses;
// the caller owns whether a current objective may adopt that evidence.
func reconstructHeldoutVerdict(ctx context.Context, reader artifact.Reader, verdict heldoutVerdict, objective trainingprogram.ObjectiveDocument) error {
	view, err := sftEvaluationViewCodec.Require(ctx, reader, verdict.View)
	if err != nil {
		return err
	}
	training, heldout, err := viewMemberships(ctx, reader, view)
	if err != nil {
		return err
	}
	if partner, err := dataset.SplitPartner(ctx, reader, training.ID); err != nil || partner.ID != heldout.ID {
		return errors.Join(fmt.Errorf("evaluation: verdict %s judges on a membership its split does not pair with the training one", verdict.ID), err)
	}
	recompiled, err := CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		return err
	}
	if recompiled.ID != view.ID {
		return fmt.Errorf("evaluation: verdict %s view does not recompile from its memberships", verdict.ID)
	}
	plan, reports, err := requireHeldout(ctx, reader, verdict.Plan, verdict.Baseline, verdict.Candidate)
	if err != nil {
		return err
	}
	trained, err := requireTraining(ctx, reader, verdict)
	if err != nil {
		return err
	}
	judged, err := JudgeHeldout(objective, view, plan, reports[0], reports[1], trained)
	if err != nil {
		return err
	}
	if judged.ID != verdict.ID {
		return fmt.Errorf("evaluation: verdict %s differs from its recomputation", verdict.ID)
	}
	return nil
}

// VisitHeldoutVerdicts visits every committed held-out verdict, newest
// first, with the training observation it judged and its declared metric's
// value for the base and for the trained model, after reconstructing the
// checkpoint, memberships and reports. Valid losses remain visible. Each
// value is summarized as
// an evaluation of the objective's held-out view under the verdict's target
// plan, in the verdict's direction. A verdict that cannot be reconstructed
// stops the visit with its error, or, when unreadable is given, is reported to
// it with its own identity and the observation it judged and the visit goes on,
// so one verdict that cannot be read costs the row it judges its quality and no
// other; a cancelled read stops the visit either way, it is no unreadable verdict.
func VisitHeldoutVerdicts(ctx context.Context, store *overgodb.Store, visit func(observation artifact.ID, base, trained EvalSummary) error, unreadable func(verdict, observation artifact.ID, reason error)) error {
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{heldoutVerdictCodec.Contract}, Order: overgodb.DocumentNewestFirst,
	}, heldoutVerdictCodec.Parse, func(_ overgodb.DocumentView, verdict heldoutVerdict) error {
		objective, err := trainingprogram.LoadObjective(ctx, store, verdict.Objective)
		if err == nil {
			err = reconstructHeldoutVerdict(ctx, store, verdict, objective)
		}
		if err != nil {
			// A cancelled read is no unreadable verdict: it stops the visit.
			if unreadable == nil || ctx.Err() != nil {
				return err
			}
			unreadable(verdict.ID, verdict.Observation, err)
			return nil
		}
		// A reconstructed target report compares exact retained targets and
		// inputs, independently of unknown execution metadata. It does not
		// qualify as known execution or benchmark reuse evidence.
		comparison, err := artifact.JSONID(artifact.KindProfile, struct {
			Plan   artifact.ID `json:"plan"`
			Inputs artifact.ID `json:"inputs"`
		}{verdict.Plan, verdict.Inputs})
		if err != nil {
			return err
		}
		metric := string(verdict.Metric)
		summary := func(value float64) EvalSummary {
			return EvalSummary{
				Suite: verdict.View.String(), Prompting: verdict.Plan.String(), Record: verdict.ID,
				Comparison: comparison,
				Metrics:    map[string]float64{metric: value}, Directions: map[string]runrecord.Direction{metric: verdict.Direction},
			}
		}
		return visit(verdict.Observation, summary(verdict.BaselineValue), summary(verdict.CandidateValue))
	})
	return err
}

// IsHeldoutVerdict reports whether a descriptor names a held-out verdict.
func IsHeldoutVerdict(descriptor artifact.Descriptor) bool {
	return descriptor.MediaType == heldoutVerdictMediaType && descriptor.Schema == heldoutVerdictSchema
}

// requireHeldout reads a committed target plan and the two reports it
// scored, as the kind of plan its descriptor names.
func requireHeldout(ctx context.Context, reader artifact.Reader, plan artifact.ID, reports ...artifact.ID) (HeldoutPlan, []HeldoutReport, error) {
	descriptor, found, err := reader.Artifact(ctx, plan)
	if err != nil {
		return HeldoutPlan{}, nil, err
	}
	if !found {
		return HeldoutPlan{}, nil, fmt.Errorf("evaluation: held-out plan %s is absent", plan)
	}
	switch descriptor.MediaType {
	case numericTargetPlanMedia:
		return requireTargets(ctx, reader, numericTargetPlanCodec, numericTargetReportCodec, plan, reports,
			func(plan numericTargetPlan) artifact.ID { return plan.View }, numericTargetPlan.rescore)
	case textTargetPlanMediaType:
		return requireTargets(ctx, reader, textTargetPlanCodec, textTargetReportCodec, plan, reports,
			func(plan textTargetPlan) artifact.ID { return plan.View }, textTargetPlan.rescore)
	case mediaTargetPlanMedia:
		return requireTargets(ctx, reader, mediaTargetPlanCodec, mediaTargetReportCodec, plan, reports,
			func(plan mediaTargetPlan) artifact.ID { return plan.View }, mediaTargetPlan.rescore)
	case preferenceTargetPlanMedia:
		return requireTargets(ctx, reader, preferenceTargetPlanCodec, preferenceTargetReportCodec, plan, reports,
			func(plan preferenceTargetPlan) artifact.ID { return plan.View }, preferenceTargetPlan.rescore)
	case unpairedTargetPlanMedia:
		return requireTargets(ctx, reader, unpairedTargetPlanCodec, unpairedTargetReportCodec, plan, reports,
			func(plan unpairedTargetPlan) artifact.ID { return plan.View }, unpairedTargetPlan.rescore)
	default:
		return HeldoutPlan{}, nil, fmt.Errorf("evaluation: %s is not a held-out target plan", plan)
	}
}

// requireTargets reads one kind's committed plan and reports through their
// codecs and rescores each report through the plan from the per-case
// answers it retains: a report whose cases, scores or metrics are not what
// those answers score to is refused, so the judge reads metrics only a
// rescoring derived.
func requireTargets[Plan, Report any](
	ctx context.Context,
	reader artifact.Reader,
	planCodec artifact.DocumentCodec[Plan],
	reportCodec artifact.DocumentCodec[Report],
	id artifact.ID,
	reportIDs []artifact.ID,
	view func(Plan) artifact.ID,
	rescore func(Plan, Report) (HeldoutReport, error),
) (HeldoutPlan, []HeldoutReport, error) {
	plan, err := planCodec.Require(ctx, reader, id)
	if err != nil {
		return HeldoutPlan{}, nil, err
	}
	reports := make([]HeldoutReport, len(reportIDs))
	for index, reportID := range reportIDs {
		report, err := reportCodec.Require(ctx, reader, reportID)
		if err != nil {
			return HeldoutPlan{}, nil, err
		}
		rescored, err := rescore(plan, report)
		if err == nil && rescored.id != reportID {
			err = errors.New("its retained cases score to another report")
		}
		if err != nil {
			return HeldoutPlan{}, nil, fmt.Errorf("evaluation: held-out report %s does not rescore under plan %s: %w", reportID, id, err)
		}
		reports[index] = rescored
	}
	return HeldoutPlan{id: id, view: view(plan), batch: func(key string) (artifact.Batch, error) {
		return planCodec.Batch(key, plan, nil, nil)
	}}, reports, nil
}

// requireTraining reads what a committed verdict binds of the training it
// judged: its observation, which must be a succeeded training session, and
// its trained checkpoint, which must be committed.
func requireTraining(ctx context.Context, reader artifact.Reader, verdict heldoutVerdict) (HeldoutTraining, error) {
	observation, err := runrecord.RequireServingObservation(ctx, reader, verdict.Observation)
	if err != nil {
		return HeldoutTraining{}, err
	}
	if observation.Task != recipe.TaskTraining || observation.Outcome != runrecord.OutcomeSucceeded {
		return HeldoutTraining{}, fmt.Errorf("evaluation: verdict %s observation %s is not a succeeded training session", verdict.ID, verdict.Observation)
	}
	content, found, err := artifact.ReadContent(ctx, reader, verdict.Trained)
	if err != nil {
		return HeldoutTraining{}, err
	}
	if !found {
		return HeldoutTraining{}, fmt.Errorf("evaluation: verdict %s trained checkpoint %s is not committed", verdict.ID, verdict.Trained)
	}
	trained, err := trainingprogram.ParseCheckpoint(content.Data)
	if err != nil || trained.ID() != verdict.Trained {
		return HeldoutTraining{}, errors.Join(fmt.Errorf("evaluation: verdict %s trained output %s is not a checkpoint", verdict.ID, verdict.Trained), err)
	}
	return HeldoutTraining{Observation: verdict.Observation, Model: observation.Model, Trained: trained, Inputs: verdict.Inputs}, nil
}

func viewMemberships(ctx context.Context, reader artifact.Reader, view SFTEvaluationView) (dataset.Membership, dataset.Membership, error) {
	training, trainingFound, err := dataset.LoadMembership(ctx, reader, view.TrainingMembership)
	if err != nil {
		return dataset.Membership{}, dataset.Membership{}, err
	}
	heldout, heldoutFound, err := dataset.LoadMembership(ctx, reader, view.HeldoutMembership)
	if err != nil {
		return dataset.Membership{}, dataset.Membership{}, err
	}
	if !trainingFound || !heldoutFound {
		return dataset.Membership{}, dataset.Membership{}, errors.New("evaluation: held-out view memberships are not committed")
	}
	return training, heldout, nil
}

func reportMetric(metrics []runrecord.Metric, name string) (runrecord.Metric, bool) {
	index := slices.IndexFunc(metrics, func(metric runrecord.Metric) bool { return metric.Name == name })
	if index < 0 {
		return runrecord.Metric{}, false
	}
	return metrics[index], true
}

func canonicalizeHeldoutVerdict(verdict *heldoutVerdict) error {
	if verdict == nil || verdict.Version != artifact.SecondDocumentVersion ||
		verdict.Objective.Kind() != artifact.KindProfile || verdict.View.Kind() != artifact.KindProfile ||
		verdict.Plan.Kind() != artifact.KindProfile || verdict.Observation.Kind() != artifact.KindEvidence ||
		verdict.Trained.Kind() != artifact.KindCheckpoint || verdict.Inputs.Kind() != artifact.KindProfile ||
		verdict.Baseline.Kind() != artifact.KindEvaluation || verdict.Candidate.Kind() != artifact.KindEvaluation ||
		verdict.Baseline == verdict.Candidate || verdict.Metric == "" || !validMetricDirection(verdict.Direction) {
		return errors.New("evaluation: invalid held-out verdict")
	}
	switch verdict.Direction {
	case runrecord.DirectionMinimize:
		verdict.Passed = verdict.CandidateValue < verdict.BaselineValue
	default:
		verdict.Passed = verdict.CandidateValue > verdict.BaselineValue
	}
	return nil
}
