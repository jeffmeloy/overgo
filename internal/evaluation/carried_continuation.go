package evaluation

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
)

const (
	// carriedContinuationKind names the suite whose cases ask a model with a
	// memory it writes to itself what that memory is worth: it reads the start
	// of a document chunk by chunk, and is then asked, shown nothing, for the
	// opening of the next chunk, once with what it remembered and once without.
	carriedContinuationKind         = "carried-continuation"
	carriedContinuationMediaType    = "application/vnd.overgo.carried-continuation-dataset+json"
	carriedContinuationSchema       = "overgo/carried-continuation-dataset/v1"
	carriedContinuationSplitMedia   = "application/vnd.overgo.carried-continuation-split+json"
	carriedContinuationSplitSchema  = "overgo/carried-continuation-split/v1"
	carriedContinuationReportMedia  = "application/vnd.overgo.carried-continuation-report+json"
	carriedContinuationReportSchema = "overgo/carried-continuation-report/v1"

	wikitextDocumentsSource = "wikitext-2-raw-v1/test/carried"
	wikitextDocumentsSchema = "overgo/wikitext-2-raw-v1-test-documents/v1"
	// carriedChunks chunks are written into the memory before the deferred
	// turn: with the model's four seed slots they fill its eight, so nothing
	// written is yet pushed out. carriedOpening tokens of the next chunk are
	// asked for, the length the model was trained to defer.
	carriedChunks, carriedOpening = 4, 16
	// documentCharacters characters of an article are kept: enough for the
	// chunks and the opening at the fewest tokens to a character the split's
	// articles have, with room to spare.
	documentCharacters = 11000
	// headingMark opens and closes the line that titles an article of the
	// split, and sectionMark a line that titles a section within one.
	headingMark, sectionMark = " = ", " = = "
)

var (
	carriedContinuationDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: carriedContinuationMediaType, Schema: carriedContinuationSchema,
	}
	carriedContinuationSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: carriedContinuationSplitMedia, Schema: carriedContinuationSplitSchema,
	}
	carriedContinuationReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: carriedContinuationReportMedia, Schema: carriedContinuationReportSchema,
	}
)

// carriedScorer is what a model with a memory it writes to itself offers a
// suite: the likelihood of the next chunk's opening tokens on a deferred turn
// after a document's first chunks, with the memory they wrote and with a
// fresh one.
type carriedScorer interface {
	ScoreCarried(ctx context.Context, text string, chunks, opening int) (carried, reset sequencescore.Score, err error)
}

// CarriedContinuationCase is the start of one document.
type CarriedContinuationCase struct {
	Name string `json:"name"`
	Text string `json:"text"`
}

// CarriedContinuationSuite is a set of documents, with how many chunks of each
// a model writes into its memory and how many tokens of the chunk after them
// it is asked for.
type CarriedContinuationSuite struct {
	Kind    string                    `json:"kind"`
	Schema  string                    `json:"schema"`
	Source  string                    `json:"source"`
	Chunks  int                       `json:"chunks"`
	Opening int                       `json:"opening"`
	Cases   []CarriedContinuationCase `json:"cases"`
}

type carriedContinuationPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    CarriedContinuationSuite
}

type carriedObservation struct {
	Name        string  `json:"name"`
	CarriedLoss float64 `json:"carried_loss"`
	ResetLoss   float64 `json:"reset_loss"`
}

// carriedContinuationReport gives, a token, what the memory carried: the mean
// over the documents of the deferred turn's loss with a fresh memory less its
// loss with the memory the document wrote, with the two losses and the
// standard error of the gain over the documents.
type carriedContinuationReport struct {
	ID           artifact.ID          `json:"-"`
	Version      uint16               `json:"version"`
	Plan         artifact.ID          `json:"plan"`
	Dataset      artifact.ID          `json:"dataset"`
	Observations []carriedObservation `json:"observations"`
	CarriedGain  float64              `json:"carried_gain"`
	GainError    float64              `json:"carried_gain_error"`
	CarriedLoss  float64              `json:"carried_loss"`
	ResetLoss    float64              `json:"reset_loss"`
}

// compileCarriedContinuation validates and identifies the suite.
func compileCarriedContinuation(suite CarriedContinuationSuite) (carriedContinuationPlan, error) {
	if suite.Kind != carriedContinuationKind || strings.TrimSpace(suite.Schema) == "" || strings.TrimSpace(suite.Source) == "" ||
		suite.Chunks <= 0 || suite.Opening <= 0 || len(suite.Cases) == 0 {
		return carriedContinuationPlan{}, errors.New("evaluation: invalid carried-continuation suite")
	}
	names := make(map[string]struct{}, len(suite.Cases))
	for _, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || strings.TrimSpace(testCase.Text) == "" {
			return carriedContinuationPlan{}, errors.New("evaluation: invalid carried-continuation case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return carriedContinuationPlan{}, errors.New("evaluation: duplicate carried-continuation case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, dataset, split, err := suiteIdentities(suite, suite.Cases)
	if err != nil {
		return carriedContinuationPlan{}, err
	}
	return carriedContinuationPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func bindCarriedContinuation(compiled carriedContinuationPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, carriedContinuationKind, authorities)
}

// evaluateCarriedContinuation asks the model for every document's deferred
// turn, with its memory and without, and publishes what the memory carried
// through the campaign ledger.
func evaluateCarriedContinuation(
	ctx context.Context,
	repository artifact.Repository,
	runtime Runtime,
	compiled carriedContinuationPlan,
	plan Plan,
) (carriedContinuationReport, error) {
	scorer, remembers := runtime.(carriedScorer)
	if ctx == nil || repository == nil || !remembers || compiled.identity != plan.body.CaseProfile {
		return carriedContinuationReport{}, errors.New("evaluation: carried-continuation authority differs, or the model carries no memory to score")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases, carriedContinuationDatasetContract, carriedContinuationSplitContract,
	)
	if err != nil {
		return carriedContinuationReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return carriedContinuationReport{}, err
	}
	report := carriedContinuationReport{Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset}
	gains := make([]RecordScore, 0, len(compiled.suite.Cases))
	documents := float64(len(compiled.suite.Cases))
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for _, testCase := range compiled.suite.Cases {
		carried, reset, err := scorer.ScoreCarried(ctx, testCase.Text, compiled.suite.Chunks, compiled.suite.Opening)
		if err != nil {
			return carriedContinuationReport{}, fmt.Errorf("evaluation: carried-continuation case %q: %w", testCase.Name, err)
		}
		if !carried.Valid() || !reset.Valid() || carried.LogProbability > 0 || reset.LogProbability > 0 {
			return carriedContinuationReport{}, fmt.Errorf("evaluation: carried-continuation case %q: the scores are not finite likelihoods", testCase.Name)
		}
		observation := carriedObservation{Name: testCase.Name, CarriedLoss: -carried.LogProbability / float64(carried.Tokens), ResetLoss: -reset.LogProbability / float64(reset.Tokens)}
		report.Observations = append(report.Observations, observation)
		gains = append(gains, RecordScore{Record: testCase.Name, Value: observation.ResetLoss - observation.CarriedLoss})
		report.CarriedLoss += observation.CarriedLoss / documents
		report.ResetLoss += observation.ResetLoss / documents
		progress.advance()
	}
	report.CarriedGain, report.GainError = report.ResetLoss-report.CarriedLoss, meanError(gains)
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return carriedContinuationReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, carriedContinuationReportContract, report,
	); err != nil {
		return carriedContinuationReport{}, err
	}
	return report, nil
}

// carriedContinuationMetrics report what a memory carried, in nats a token:
// the gain is better higher, each loss lower. The gain is named so that it is
// the metric a reader meets first.
func carriedContinuationMetrics(gain, carried, reset float64) []runrecord.Metric {
	return []runrecord.Metric{
		{Name: "carried_gain", Value: gain, Direction: runrecord.DirectionMaximize},
		{Name: "carried_loss", Value: carried, Direction: runrecord.DirectionMinimize},
		{Name: "reset_loss", Value: reset, Direction: runrecord.DirectionMinimize},
	}
}

// BuildWikiTextDocuments builds the carried-continuation suite of the
// WikiText-2 test split from the lines of its text column, in row order: the
// start of every article long enough to hold the chunks a model writes into
// its memory and the opening of the chunk after them, cut to a fixed number of
// characters and ended on a word. An article opens at the line that titles it.
// It returns the suite and the number of its cases.
func BuildWikiTextDocuments(lines []string) (any, int, error) {
	suite := CarriedContinuationSuite{
		Kind: carriedContinuationKind, Schema: wikitextDocumentsSchema, Source: wikitextDocumentsSource, Chunks: carriedChunks, Opening: carriedOpening,
	}
	var article []rune
	keep := func() {
		if len(article) < documentCharacters {
			return
		}
		document := string(article[:documentCharacters])
		if word := strings.LastIndexByte(document, ' '); word > 0 {
			document = document[:word]
		}
		suite.Cases = append(suite.Cases, CarriedContinuationCase{Name: fmt.Sprintf("%s/%02d", wikitextDocumentsSource, len(suite.Cases)), Text: document})
	}
	for _, line := range lines {
		titled := strings.TrimRight(line, "\n")
		if strings.HasPrefix(titled, headingMark) && strings.HasSuffix(titled, headingMark) && !strings.HasPrefix(titled, sectionMark) {
			keep()
			article = article[:0]
		}
		article = append(article, []rune(line)...)
	}
	keep()
	if _, err := compileCarriedContinuation(suite); err != nil {
		return nil, 0, err
	}
	return suite, len(suite.Cases), nil
}
