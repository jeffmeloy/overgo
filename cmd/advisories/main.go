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
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/finding"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

func main() {
	clioptions.MainNamed("advisories", run)
}

func run() error {
	flags := flag.NewFlagSet("advisories", flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "RepoDB store; empty resolves via the data-root contract")
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
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()

	observations, err := loadObservations(ctx, store, *metric)
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
	fmt.Println("honesty: directional evidence only; repeated sequential looks and regime changes are not covered by the recorded per-look budget")
	if !raised {
		fmt.Println("verdict: no directional regression at the empirical threshold")
		return nil
	}
	batch, err := advisory.Batch("advisory/" + *metric + "/" + advisory.LatestRun.String())
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	fmt.Printf("ADVISORY raised: latest=%g baseline_median=%g mad=%g surprise=%g (committed %s)\n",
		advisory.LatestValue, advisory.BaselineMedian, advisory.MAD, advisory.Surprise(), advisory.ID)
	return escalate(ctx, store, advisory)
}

// escalate opens a finding only after the same series raises on two
// non-overlapping windows. One open finding per series; repeats add no new
// document.
func escalate(ctx context.Context, store *repodb.Store, latest runrecord.Advisory) error {
	result, err := store.Query(ctx, repodb.Query{Kind: artifact.KindEvidence, MaxResults: 100_000})
	if err != nil {
		return err
	}
	var confirming []runrecord.Advisory
	openFindingExists := false
	seriesKey := latest.Recipe.String() + "/" + latest.Environment.String() + "/" + latest.Metric
	for _, descriptor := range result.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil || !ok {
			continue
		}
		switch descriptor.MediaType {
		case runrecord.AdvisoryMediaType:
			prior, err := runrecord.ParseAdvisory(content.Data)
			if err != nil || prior.Metric != latest.Metric ||
				prior.Recipe != latest.Recipe || prior.Environment != latest.Environment ||
				!nonOverlappingConfirmation(prior, latest) {
				continue
			}
			confirming = append(confirming, prior)
		case finding.MediaType:
			document, err := finding.Parse(content.Data)
			if err != nil || document.Status != finding.StatusOpen {
				continue
			}
			if strings.Contains(document.Title, seriesKey) {
				openFindingExists = true
			}
		}
	}
	if len(confirming) == 0 {
		fmt.Println("honesty: directional advisory only; a later non-overlapping window is required for a finding")
		return nil
	}
	if openFindingExists {
		fmt.Println("honesty: confirmed regression already has an open finding; no duplicate emitted")
		return nil
	}
	evidence := []artifact.ID{latest.ID}
	for _, prior := range confirming {
		evidence = append(evidence, prior.ID)
	}
	document, err := finding.New(
		"repeated directional regression in series "+seriesKey,
		finding.SeverityMedium, finding.StatusOpen,
		[]artifact.ID{latest.Recipe, latest.Environment}, evidence,
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
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	fmt.Printf("FINDING opened (non-overlapping two-window repetition, %d prior advisor(ies)): %s\n", len(confirming), document.ID)
	return nil
}

func nonOverlappingConfirmation(prior, latest runrecord.Advisory) bool {
	return prior.LatestSequence < latest.WindowStart
}

// loadObservations pairs run and evaluation documents by evaluation.Run and
// orders them by TRUE chronology: the store's commit sequence, joined through
// the gate's batch-key convention ("gate/<codeCommit>") against the Run's own
// CodeCommit. Query returns artifacts in content-hash order -- uncorrelated
// with time -- so any arrival-order sequencing would feed the detector a
// shuffled series (caught before enforcement activated; this join is the
// zero-store-change fix, and a store-owned introduction sequence is the
// eventual owner if non-gate series need it).
func loadObservations(ctx context.Context, store *repodb.Store, metric string) ([]runrecord.Observation, error) {
	commitsResult, err := store.Query(ctx, repodb.Query{FromSequence: 1, MaxResults: 100_000})
	if err != nil {
		return nil, err
	}
	sequenceByCommitKey := map[string]uint64{}
	for _, view := range commitsResult.Commits {
		sequenceByCommitKey[view.Key] = view.Sequence
	}
	runs := map[artifact.ID]runrecord.Run{}
	runsResult, err := store.Query(ctx, repodb.Query{Kind: artifact.KindRun, MaxResults: 100_000})
	if err != nil {
		return nil, err
	}
	for _, descriptor := range runsResult.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil || !ok {
			continue
		}
		parsed, err := runrecord.ParseRun(content.Data)
		if err != nil {
			continue // other run schemas are not this series
		}
		runs[parsed.ID] = parsed
	}
	evaluationsResult, err := store.Query(ctx, repodb.Query{Kind: artifact.KindEvaluation, MaxResults: 100_000})
	if err != nil {
		return nil, err
	}
	var observations []runrecord.Observation
	unmapped := 0
	for _, descriptor := range evaluationsResult.Artifacts {
		content, ok, err := store.Content(ctx, descriptor.ID)
		if err != nil || !ok {
			continue
		}
		evaluation, err := runrecord.ParseEvaluation(content.Data)
		if err != nil {
			continue
		}
		run, ok := runs[evaluation.Run]
		if !ok {
			continue
		}
		hasMetric := false
		for _, m := range evaluation.Metrics {
			hasMetric = hasMetric || m.Name == metric
		}
		if !hasMetric {
			continue
		}
		sequence, ok := sequenceByCommitKey["gate/"+run.CodeCommit]
		if !ok {
			unmapped++
			continue
		}
		observations = append(observations, runrecord.Observation{
			Sequence: sequence, Run: run, Evaluation: evaluation,
		})
	}
	if unmapped > 0 {
		fmt.Printf("honesty: %d observation(s) skipped -- no commit-key chronology for their series\n", unmapped)
	}
	sort.Slice(observations, func(i, j int) bool { return observations[i].Sequence < observations[j].Sequence })
	return observations, nil
}

type calibrationPlan struct {
	Window          int
	RequiredScores  int
	AvailableScores int
	LatestStart     int
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
		AvailableScores: available, LatestStart: latestStart,
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
	sort.Float64s(ordered)
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
func recordBudgetDecision(ctx context.Context, store *repodb.Store, metric string, budget float64, reason string) error {
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
