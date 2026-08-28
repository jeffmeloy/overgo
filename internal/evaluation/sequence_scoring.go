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
	// SequenceScoringKind names the held-out-corpus suite envelope.
	SequenceScoringKind        = "sequence-scoring"
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

// SequenceScoringCase is one held-out sequence.
type SequenceScoringCase struct {
	Name  string `json:"name"`
	Group string `json:"group"`
	Text  string `json:"text"`
}

// SequenceScoringSuite scores every case's full text.
type SequenceScoringSuite struct {
	Kind   string                `json:"kind"`
	Schema string                `json:"schema"`
	Source string                `json:"source"`
	Cases  []SequenceScoringCase `json:"cases"`
}

// SequenceScoringPlan is the compiled suite with its authorities.
type SequenceScoringPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    SequenceScoringSuite
}

// SequenceObservation is one scored sequence.
type SequenceObservation struct {
	Name           string  `json:"name"`
	Group          string  `json:"group"`
	LogProbability float64 `json:"log_probability"`
	Tokens         uint64  `json:"tokens"`
}

// SequenceScoringReport aggregates held-out likelihoods: mean NLL per
// token across every scored token, and its exponential as perplexity.
type SequenceScoringReport struct {
	ID              artifact.ID           `json:"-"`
	Version         uint16                `json:"version"`
	Plan            artifact.ID           `json:"plan"`
	Dataset         artifact.ID           `json:"dataset"`
	Observations    []SequenceObservation `json:"observations"`
	TokensScored    uint64                `json:"tokens_scored"`
	MeanNLLPerToken float64               `json:"mean_nll_per_token"`
	Perplexity      float64               `json:"perplexity"`
}

// CompileSequenceScoring validates and identifies the suite.
func CompileSequenceScoring(suite SequenceScoringSuite) (SequenceScoringPlan, error) {
	if suite.Kind != SequenceScoringKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return SequenceScoringPlan{}, errors.New("evaluation: invalid sequence-scoring suite")
	}
	suite.Cases = slices.Clone(suite.Cases)
	names := make(map[string]struct{}, len(suite.Cases))
	for _, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || testCase.Text == "" || strings.TrimSpace(testCase.Group) == "" {
			return SequenceScoringPlan{}, errors.New("evaluation: invalid sequence-scoring case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return SequenceScoringPlan{}, errors.New("evaluation: duplicate sequence-scoring case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return SequenceScoringPlan{}, err
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, suite.Cases)
	if err != nil {
		return SequenceScoringPlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: dataset})
	if err != nil {
		return SequenceScoringPlan{}, err
	}
	return SequenceScoringPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

// BindSequenceScoring binds the compiled suite to its authorities.
func BindSequenceScoring(compiled SequenceScoringPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, SequenceScoringKind, authorities)
}

// EvaluateSequenceScoring scores every sequence and publishes the
// report through the campaign ledger.
func EvaluateSequenceScoring(
	ctx context.Context,
	repository artifact.Repository,
	scorer ContinuationScorer,
	compiled SequenceScoringPlan,
	plan Plan,
) (SequenceScoringReport, error) {
	if ctx == nil || repository == nil || scorer == nil || compiled.identity != plan.body.CaseProfile {
		return SequenceScoringReport{}, errors.New("evaluation: sequence-scoring authority differs")
	}
	contents, err := compiled.contents()
	if err != nil {
		return SequenceScoringReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return SequenceScoringReport{}, err
	}
	report := SequenceScoringReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
	}
	// The model's declared scoring prefix shapes pure-sequence input --
	// a hybrid DNA tokenizer scores inside its declared region, exactly
	// the trained format; models without a declaration score from the
	// empty context.
	prefix := runtimeScoringPrefix(scorer)
	var totalNLL float64
	for _, testCase := range compiled.suite.Cases {
		scores, err := scorer.ScoreContinuations(ctx, prefix, []string{testCase.Text})
		if err != nil {
			return SequenceScoringReport{}, err
		}
		if len(scores) != 1 || scores[0].Tokens == 0 || scores[0].LogProbability > 0 ||
			math.IsNaN(scores[0].LogProbability) || math.IsInf(scores[0].LogProbability, 0) {
			return SequenceScoringReport{}, errors.New("evaluation: sequence score is not a finite likelihood")
		}
		report.Observations = append(report.Observations, SequenceObservation{
			Name: testCase.Name, Group: testCase.Group,
			LogProbability: scores[0].LogProbability, Tokens: scores[0].Tokens,
		})
		totalNLL -= scores[0].LogProbability
		report.TokensScored += scores[0].Tokens
	}
	report.MeanNLLPerToken = totalNLL / float64(report.TokensScored)
	report.Perplexity = math.Exp(report.MeanNLLPerToken)
	if math.IsNaN(report.Perplexity) || math.IsInf(report.Perplexity, 0) {
		return SequenceScoringReport{}, errors.New("evaluation: perplexity is not finite")
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return SequenceScoringReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, sequenceScoringReportContract, report,
	); err != nil {
		return SequenceScoringReport{}, err
	}
	return report, nil
}

func (p SequenceScoringPlan) contents() ([]artifact.Content, error) {
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
