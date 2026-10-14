package evaluation

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
)

// Sequence scoring is the held-out-corpus suite shape: each case is
// one sequence, the model's per-token log-likelihood over it is the
// observation, and the suite aggregates to mean negative log
// likelihood per token and perplexity. This is how a domain without
// labeled answers -- a DNA corpus slice -- scores in-domain: the
// model either predicts its domain's sequences or it does not.
const (
	// sequenceScoringKind names the held-out-corpus suite envelope.
	sequenceScoringKind        = "sequence-scoring"
	sequenceScoringMediaType   = "application/vnd.overgo.sequence-scoring-dataset+json"
	sequenceScoringSchema      = "overgo/sequence-scoring-dataset/v1"
	sequenceScoringSplitMedia  = "application/vnd.overgo.sequence-scoring-split+json"
	sequenceScoringSplitSchema = "overgo/sequence-scoring-split/v1"
	sequenceScoringReportMedia = "application/vnd.overgo.sequence-scoring-report+json"
	sequenceScoringReportSch   = "overgo/sequence-scoring-report/v1"
)

var (
	sequenceScoringDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: sequenceScoringMediaType, Schema: sequenceScoringSchema,
	}
	sequenceScoringSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: sequenceScoringSplitMedia, Schema: sequenceScoringSplitSchema,
	}
	sequenceScoringReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: sequenceScoringReportMedia, Schema: sequenceScoringReportSch,
	}
)

// sequenceScoringCase is one held-out sequence.
type sequenceScoringCase struct {
	Name  string `json:"name"`
	Group string `json:"group"`
	Text  string `json:"text"`
}

// sequenceScoringSuite scores every case's full text.
type sequenceScoringSuite struct {
	Kind   string                `json:"kind"`
	Schema string                `json:"schema"`
	Source string                `json:"source"`
	Cases  []sequenceScoringCase `json:"cases"`
}

// sequenceScoringPlan is the compiled suite with its authorities.
type sequenceScoringPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    sequenceScoringSuite
}

// sequenceObservation is one scored sequence.
type sequenceObservation struct {
	Name           string  `json:"name"`
	Group          string  `json:"group"`
	LogProbability float64 `json:"log_probability"`
	Tokens         uint64  `json:"tokens"`
}

// sequenceScoringReport aggregates held-out likelihoods: mean NLL per
// token across every scored token, and its exponential as perplexity.
type sequenceScoringReport struct {
	ID              artifact.ID           `json:"-"`
	Version         uint16                `json:"version"`
	Plan            artifact.ID           `json:"plan"`
	Dataset         artifact.ID           `json:"dataset"`
	Observations    []sequenceObservation `json:"observations"`
	TokensScored    uint64                `json:"tokens_scored"`
	MeanNLLPerToken float64               `json:"mean_nll_per_token"`
	Perplexity      float64               `json:"perplexity"`
}

// compileSequenceScoring validates and identifies the suite.
func compileSequenceScoring(suite sequenceScoringSuite) (sequenceScoringPlan, error) {
	if suite.Kind != sequenceScoringKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return sequenceScoringPlan{}, errors.New("evaluation: invalid sequence-scoring suite")
	}
	suite.Cases = slices.Clone(suite.Cases)
	names := make(map[string]struct{}, len(suite.Cases))
	for _, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || testCase.Text == "" || strings.TrimSpace(testCase.Group) == "" {
			return sequenceScoringPlan{}, errors.New("evaluation: invalid sequence-scoring case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return sequenceScoringPlan{}, errors.New("evaluation: duplicate sequence-scoring case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, dataset, split, err := suiteIdentities(suite, suite.Cases)
	if err != nil {
		return sequenceScoringPlan{}, err
	}
	return sequenceScoringPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

// bindSequenceScoring binds the compiled suite to its authorities.
func bindSequenceScoring(compiled sequenceScoringPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, sequenceScoringKind, authorities)
}

// evaluateSequenceScoring scores every sequence and publishes the
// report through the campaign ledger.
func evaluateSequenceScoring(
	ctx context.Context,
	repository artifact.Repository,
	scorer continuationScorer,
	compiled sequenceScoringPlan,
	plan Plan,
) (sequenceScoringReport, error) {
	if ctx == nil || repository == nil || scorer == nil || compiled.identity != plan.body.CaseProfile {
		return sequenceScoringReport{}, errors.New("evaluation: sequence-scoring authority differs")
	}
	contents, err := compiled.contents()
	if err != nil {
		return sequenceScoringReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return sequenceScoringReport{}, err
	}
	report := sequenceScoringReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
	}
	// The model's declared scoring prefix shapes pure-sequence input --
	// a hybrid DNA tokenizer scores inside its declared region, exactly
	// the trained format; models without a declaration score from the
	// empty context.
	prefix := runtimeScoringPrefix(scorer)
	var totalNLL float64
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for _, testCase := range compiled.suite.Cases {
		scores, err := scorer.ScoreContinuations(ctx, prefix, []string{testCase.Text})
		if err != nil {
			return sequenceScoringReport{}, err
		}
		progress.advance()
		if len(scores) != 1 || scores[0].Tokens == 0 || scores[0].LogProbability > 0 ||
			math.IsNaN(scores[0].LogProbability) || math.IsInf(scores[0].LogProbability, 0) {
			return sequenceScoringReport{}, errors.New("evaluation: sequence score is not a finite likelihood")
		}
		report.Observations = append(report.Observations, sequenceObservation{
			Name: testCase.Name, Group: testCase.Group,
			LogProbability: scores[0].LogProbability, Tokens: scores[0].Tokens,
		})
		totalNLL -= scores[0].LogProbability
		report.TokensScored += scores[0].Tokens
	}
	report.MeanNLLPerToken = totalNLL / float64(report.TokensScored)
	report.Perplexity = math.Exp(report.MeanNLLPerToken)
	if math.IsNaN(report.Perplexity) || math.IsInf(report.Perplexity, 0) {
		return sequenceScoringReport{}, errors.New("evaluation: perplexity is not finite")
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return sequenceScoringReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, sequenceScoringReportContract, report,
	); err != nil {
		return sequenceScoringReport{}, err
	}
	return report, nil
}

func (p sequenceScoringPlan) contents() ([]artifact.Content, error) {
	return datasetContents(
		p.dataset, p.split, p.suite.Cases, sequenceScoringDatasetContract, sequenceScoringSplitContract,
	)
}

// sequenceScoringMetrics report likelihood quality: lower is better
// for both, unlike the accuracy suites.
func sequenceScoringMetrics(meanNLL, perplexity float64) []runrecord.Metric {
	return []runrecord.Metric{
		{Name: "mean_nll_per_token", Value: meanNLL, Direction: runrecord.DirectionMinimize},
		{Name: "perplexity", Value: perplexity, Direction: runrecord.DirectionMinimize},
	}
}
