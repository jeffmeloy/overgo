// advisories: the statistical layer's driver (floor component 6). Builds the
// observation series for a metric from the store's run+evaluation history,
// derives the regression threshold from disjoint historical blocks (the
// target alarm budget is a recorded decision), and commits any advisory the
// directional detector raises.
//
// Median/MAD and empirical quantiles avoid a Gaussian model, but historical
// calibration does not establish a sequential or regime-stable false-alarm
// guarantee. First history calibrates, later observations advise:
// with insufficient history the driver reports its calibration state and
// enforces nothing — no history, no enforcement.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/finding"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.MainNamed("advisories", run)
}

func run() error {
	flags := flag.NewFlagSet("advisories", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	metric := flags.String("metric", "gate_wall_ns", "evaluation metric to watch")
	budget := flags.Float64("alarm-budget", 0, "target alarm budget in (0,1); required decision input, not a guaranteed false-alarm rate")
	reason := flags.String("reason", "", "why this alarm budget; recorded with the decision")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if *budget <= 0 || *budget >= 1 {
		return errors.New("-alarm-budget in (0,1) is required: it is a recorded target, not a guaranteed false-alarm rate")
	}
	if strings.TrimSpace(*reason) == "" {
		return errors.New("-reason is required: the alarm budget is a recorded decision, not a default")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	repository := strings.TrimSpace(*repoFlag)
	if repository == "" {
		repository = roots.Store
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()

	observationLimit := calibrationObservationLimit(*budget)
	observations, err := loadObservations(ctx, store, *metric, observationLimit)
	if err != nil {
		return err
	}
	if err := recordBudgetDecision(ctx, store, *metric, *budget, *reason); err != nil {
		return err
	}
	calibration := planCalibration(len(observations), *budget)
	if !calibration.Sufficient() {
		fmt.Printf("calibrating: metric=%s observations=%d disjoint_scores=%d required_scores=%d alarm_budget=%g; no advisory threshold yet\n",
			*metric, len(observations), calibration.AvailableScores, calibration.RequiredScores, *budget)
		return nil
	}
	surprises := disjointSurprises(observations[:calibration.LatestStart], *metric, calibration.Window)
	if len(surprises) < calibration.RequiredScores {
		return errors.New("advisories: calibration contract produced insufficient disjoint scores")
	}
	threshold := empiricalQuantile(surprises, 1-*budget)
	if math.IsNaN(threshold) || math.IsInf(threshold, 0) {
		return errors.New("advisories: calibration produced a non-finite threshold")
	}
	if threshold <= 0 {
		// A flat history has zero surprise everywhere; any nonzero deviation
		// is then novel by construction. The smallest positive representable
		// threshold keeps the detector armed without asserting a scale.
		threshold = math.SmallestNonzeroFloat64
	}
	latest := observations[calibration.LatestStart:]
	advisory, raised, err := runrecord.DetectRegression(latest, *metric, calibration.Window, threshold)
	if err != nil {
		return err
	}
	fmt.Printf("empirical advisory threshold: metric=%s observations=%d window=%d disjoint_scores=%d alarm_budget=%g threshold=%g\n",
		*metric, len(observations), calibration.Window, len(surprises), *budget, threshold)
	fmt.Println("audit: directional evidence only; repeated sequential looks and regime changes are not covered by the recorded per-look budget")
	if !raised {
		fmt.Println("verdict: no directional regression at the empirical threshold")
		return nil
	}
	batch, err := advisory.Batch("advisory/" + *metric + "/" + advisory.LatestRun.String())
	if err != nil {
		return err
	}
	alias := runrecord.AdvisoryAlias(advisory.Recipe, advisory.Environment, advisory.Metric)
	previous, found, err := artifact.ResolveAlias(ctx, store, alias)
	if err != nil {
		return err
	}
	binding := artifact.AliasBinding{Name: alias, Target: advisory.ID}
	if found {
		binding.Previous = &previous
	}
	batch.Aliases = append(batch.Aliases, binding)
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	fmt.Printf("ADVISORY raised: latest=%g baseline_median=%g mad=%g surprise=%g (committed %s)\n",
		advisory.LatestValue, advisory.BaselineMedian, advisory.MAD, advisory.Surprise(), advisory.ID)
	return escalate(ctx, store, advisory, binding.Previous)
}

// escalate opens a finding only after the same series raises on two
// non-overlapping windows. One open finding per series; repeats add no new
// document.
func escalate(ctx context.Context, store *overgodb.Store, latest runrecord.Advisory, previous *artifact.ID) error {
	seriesKey := latest.Recipe.String() + "/" + latest.Environment.String() + "/" + latest.Metric
	if previous == nil {
		fmt.Println("audit: directional advisory only; a later non-overlapping window is required for a finding")
		return nil
	}
	content, found, err := artifact.ReadContent(ctx, store, *previous)
	if err != nil {
		return err
	}
	if !found {
		return errors.New("advisories: prior series advisory is absent")
	}
	prior, err := runrecord.ParseAdvisory(content.Data)
	if err != nil {
		return err
	}
	if prior.Metric != latest.Metric || prior.Recipe != latest.Recipe || prior.Environment != latest.Environment {
		return errors.New("advisories: active series alias is inconsistent")
	}
	if !nonOverlappingConfirmation(prior, latest) {
		fmt.Println("audit: directional advisory only; a later non-overlapping window is required for a finding")
		return nil
	}
	findingAlias := "finding/active/regression/" + strings.TrimPrefix(runrecord.AdvisoryAlias(
		latest.Recipe, latest.Environment, latest.Metric), runrecord.AdvisoryAliasRoot)
	_, openFindingExists, err := artifact.ResolveAlias(ctx, store, findingAlias)
	if err != nil {
		return err
	}
	if openFindingExists {
		fmt.Println("audit: confirmed regression already has an open finding; no duplicate emitted")
		return nil
	}
	document, err := finding.New(
		"repeated directional regression in series "+seriesKey,
		finding.SeverityMedium, finding.StatusOpen,
		[]artifact.ID{latest.Recipe, latest.Environment}, []artifact.ID{latest.ID, prior.ID},
		"Diagnose the regression source via the advisories' phase deltas; close with the fix landed and this series back under its empirical threshold.",
		"advisories for this series stop raising across two consecutive non-overlapping windows after the fix commit.",
	)
	if err != nil {
		return err
	}
	batch, err := document.Batch("finding/" + document.ID.String())
	if err != nil {
		return err
	}
	binding := artifact.AliasBinding{Name: findingAlias, Target: document.ID}
	batch.Aliases = append(batch.Aliases, binding)
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	fmt.Printf("FINDING opened (non-overlapping two-window repetition): %s\n", document.ID)
	return nil
}

func nonOverlappingConfirmation(prior, latest runrecord.Advisory) bool {
	return prior.LatestSequence < latest.WindowStart
}

// loadObservations joins evaluations to runs in store-owned introduction order.
func loadObservations(ctx context.Context, store *overgodb.Store, metric string, maxResults int) ([]runrecord.Observation, error) {
	query := overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{
			Kind: artifact.KindEvaluation, MediaType: runrecord.EvaluationMediaType, Schema: runrecord.EvaluationSchema,
		}}, Order: overgodb.DocumentNewestFirst, MaxResults: maxResults,
	}
	observations := make([]runrecord.Observation, 0, maxResults)
	for len(observations) < maxResults {
		page, err := overgodb.VisitDecodedDocuments(ctx, store, query, runrecord.ParseEvaluation,
			func(view overgodb.DocumentView, evaluation runrecord.Evaluation) error {
				if len(observations) == maxResults || !slices.ContainsFunc(evaluation.Metrics, func(value runrecord.Metric) bool {
					return value.Name == metric
				}) {
					return nil
				}
				run, err := runrecord.RequireRun(ctx, store, evaluation.Run)
				if err != nil {
					return err
				}
				observations = append(observations, runrecord.Observation{
					Sequence: view.Sequence, Run: run, Evaluation: evaluation,
				})
				return nil
			})
		if err != nil {
			return nil, err
		}
		if page.Next == nil {
			break
		}
		query.Cursor = page.Next
	}
	slices.Reverse(observations)
	return observations, nil
}

func calibrationObservationLimit(budget float64) int {
	var observed int
	return planCalibration(observed, budget).RequiredObservations
}

type calibrationPlan struct {
	Window               int
	RequiredScores       int
	AvailableScores      int
	LatestStart          int
	RequiredObservations int
}

func planCalibration(observations int, budget float64) calibrationPlan {
	// With n exchangeable calibration scores, the smallest resolvable upper
	// tail is 1/(n+1). The budget therefore owns the history requirement; the
	// detector's minimum valid window minimizes evidence consumption.
	required := int(math.Ceil(1/budget)) - 1
	if required < 2 {
		required = 2
	}
	window := runrecord.MinAdvisoryWindow
	block := window + 1
	latestStart := observations - block
	available := 0
	if latestStart >= 0 {
		available = latestStart / block
	}
	return calibrationPlan{
		Window: window, RequiredScores: required,
		AvailableScores: available, LatestStart: latestStart, RequiredObservations: (required + 1) * block,
	}
}

func (plan calibrationPlan) Sufficient() bool {
	return plan.LatestStart >= 0 && plan.AvailableScores >= plan.RequiredScores
}

// disjointSurprises partitions history from its newest edge. Every baseline
// and evaluated point belongs to exactly one score, preventing overlapping
// evidence from masquerading as independent calibration history.
func disjointSurprises(observations []runrecord.Observation, metric string, window int) []float64 {
	var out []float64
	block := window + 1
	start := len(observations) % block
	for start+block <= len(observations) {
		slice := observations[start : start+block]
		advisory, raised, err := runrecord.DetectRegression(slice, metric, window, math.SmallestNonzeroFloat64)
		if err != nil {
			return nil
		}
		if !raised {
			out = append(out, 0)
		} else {
			out = append(out, advisory.Surprise())
		}
		start += block
	}
	return out
}

func empiricalQuantile(values []float64, q float64) float64 {
	if len(values) == 0 {
		return 0
	}
	ordered := append([]float64(nil), values...)
	slices.Sort(ordered)
	index := int(math.Ceil(q*float64(len(ordered)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(ordered) {
		index = len(ordered) - 1
	}
	return ordered[index]
}

// recordBudgetDecision commits the alarm budget as a content-addressed
// decision document; identical budget+reason recommits idempotently.
func recordBudgetDecision(ctx context.Context, store *overgodb.Store, metric string, budget float64, reason string) error {
	decider := "unknown"
	if out, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
		decider = strings.TrimSpace(string(out))
	}
	payload := []byte(fmt.Sprintf(
		"alarm-budget decision\nmetric: %s\nbudget: %g\nreason: %s\ndecider_commit: %s\nreopen: re-derive when measured triage cost changes\n",
		metric, budget, reason, decider))
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
	if err != nil {
		return err
	}
	_, err = store.Commit(ctx, artifact.Batch{
		Key: "advisories/alarm-budget/" + metric + "/" + id.String(),
		Contents: []artifact.Content{{
			Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(payload))},
			Data:       payload,
		}},
	})
	if err != nil && strings.Contains(err.Error(), "batch key already names") {
		return nil // same decision, already recorded
	}
	return err
}
