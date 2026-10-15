package evaluation

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/inference"
)

const (
	// descriptionMatchingKind names the suite whose cases ask a model to
	// describe what it sees and score which of the candidate descriptions its
	// own matches best: the protocol for a model that answers in descriptions,
	// not in letters.
	descriptionMatchingKind         = "description-matching"
	descriptionMatchingMediaType    = "application/vnd.overgo.description-matching-dataset+json"
	descriptionMatchingSchema       = "overgo/description-matching-dataset/v1"
	descriptionMatchingSplitMedia   = "application/vnd.overgo.description-matching-split+json"
	descriptionMatchingSplitSchema  = "overgo/description-matching-split/v1"
	descriptionMatchingReportMedia  = "application/vnd.overgo.description-matching-report+json"
	descriptionMatchingReportSchema = "overgo/description-matching-report/v1"
	// frameMatchingSource and frameMatchingSchema name the suite of the video
	// clips' middle frames, each asked of as one image.
	frameMatchingSource = "vidgen-1m/frame-description-matching"
	frameMatchingSchema = "overgo/vidgen-frame-description-matching/v1"
	imageExtension      = ".png"
	// frameDescriptionTokens bounds the description a model writes of a frame.
	frameDescriptionTokens = 48
	frameQuestion          = "Describe this image in detail."
	// minimumCandidates is the fewest descriptions a case chooses among.
	minimumCandidates = 2
	// harmonicMeanFactor is the two in the harmonic mean of two shares.
	harmonicMeanFactor = 2
)

var (
	descriptionMatchingDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: descriptionMatchingMediaType, Schema: descriptionMatchingSchema,
	}
	descriptionMatchingSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: descriptionMatchingSplitMedia, Schema: descriptionMatchingSplitSchema,
	}
	descriptionMatchingReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: descriptionMatchingReportMedia, Schema: descriptionMatchingReportSchema,
	}
)

// DescriptionMatchingCase asks for a description of what the prompt shows; the
// case is right when the candidate its description matches best, by the share of
// words they have in common, is the answer's.
type DescriptionMatchingCase struct {
	Name       string   `json:"name"`
	Prompt     string   `json:"prompt"`
	MaxTokens  int      `json:"max_tokens"`
	Candidates []string `json:"candidates"`
	Answer     int      `json:"answer"`
}

// DescriptionMatchingSuite is a set of description-matching cases.
type DescriptionMatchingSuite struct {
	Kind   string                    `json:"kind"`
	Schema string                    `json:"schema"`
	Source string                    `json:"source"`
	Cases  []DescriptionMatchingCase `json:"cases"`
}

type descriptionMatchingPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    DescriptionMatchingSuite
}

type descriptionMatchingObservation struct {
	Name     string      `json:"name"`
	Raw      string      `json:"raw"`
	Chosen   int         `json:"chosen"`
	Accepted bool        `json:"accepted"`
	Producer artifact.ID `json:"producer,omitzero"`
}

type descriptionMatchingReport struct {
	ID           artifact.ID                      `json:"-"`
	Version      uint16                           `json:"version"`
	Plan         artifact.ID                      `json:"plan"`
	Dataset      artifact.ID                      `json:"dataset"`
	Observations []descriptionMatchingObservation `json:"observations"`
	Accuracy     float64                          `json:"accuracy"`
}

// compileDescriptionMatching validates and identifies the suite.
func compileDescriptionMatching(suite DescriptionMatchingSuite) (descriptionMatchingPlan, error) {
	if suite.Kind != descriptionMatchingKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return descriptionMatchingPlan{}, errors.New("evaluation: invalid description-matching suite")
	}
	names := make(map[string]struct{}, len(suite.Cases))
	for _, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || testCase.Prompt == "" || testCase.MaxTokens <= 0 ||
			len(testCase.Candidates) < minimumCandidates || testCase.Answer < 0 || testCase.Answer >= len(testCase.Candidates) ||
			slices.Contains(testCase.Candidates, "") {
			return descriptionMatchingPlan{}, errors.New("evaluation: invalid description-matching case")
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return descriptionMatchingPlan{}, errors.New("evaluation: duplicate description-matching case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, dataset, split, err := suiteIdentities(suite, suite.Cases)
	if err != nil {
		return descriptionMatchingPlan{}, err
	}
	return descriptionMatchingPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func bindDescriptionMatching(compiled descriptionMatchingPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, descriptionMatchingKind, authorities)
}

// evaluateDescriptionMatching asks every case for its description, finds the
// candidate each matches best, and publishes the share it gets right through
// the campaign ledger.
func evaluateDescriptionMatching(
	ctx context.Context,
	repository artifact.Repository,
	generator Generator,
	compiled descriptionMatchingPlan,
	plan Plan,
) (descriptionMatchingReport, error) {
	if ctx == nil || repository == nil || generator == nil || compiled.identity != plan.body.CaseProfile {
		return descriptionMatchingReport{}, errors.New("evaluation: description-matching authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases, descriptionMatchingDatasetContract, descriptionMatchingSplitContract,
	)
	if err != nil {
		return descriptionMatchingReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return descriptionMatchingReport{}, err
	}
	report := descriptionMatchingReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
		Observations: make([]descriptionMatchingObservation, len(compiled.suite.Cases)),
	}
	var producers []artifact.ID
	correct := 0
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for index, testCase := range compiled.suite.Cases {
		_, text, err := generator.Generate(ctx, testCase.Prompt, inference.GenerateOptions{MaxNewTokens: testCase.MaxTokens})
		if err != nil {
			return descriptionMatchingReport{}, fmt.Errorf("evaluation: description-matching case %q: %w", testCase.Name, err)
		}
		chosen := bestMatch(text, testCase.Candidates)
		accepted := chosen == testCase.Answer
		if accepted {
			correct++
		}
		report.Observations[index] = descriptionMatchingObservation{Name: testCase.Name, Raw: strings.ToValidUTF8(text, string(unicode.ReplacementChar)), Chosen: chosen, Accepted: accepted, Producer: producingRun(generator)}
		if producer := report.Observations[index].Producer; producer.Valid() {
			producers = append(producers, producer)
		}
		progress.hit(accepted)
	}
	report.Accuracy = float64(correct) / float64(len(report.Observations))
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return descriptionMatchingReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, descriptionMatchingReportContract, report, producers...,
	); err != nil {
		return descriptionMatchingReport{}, err
	}
	return report, nil
}

// bestMatch is the index of the candidate whose words the description shares
// the largest share of, scored as the harmonic mean of the share of the
// description's words the candidate holds and of the candidate's the
// description holds, or -1 when the best is shared by two candidates or none
// holds a word of it.
func bestMatch(description string, candidates []string) int {
	words := wordCounts(description)
	best, tied := -1, false
	var bestScore float64
	for index, candidate := range candidates {
		score := wordOverlap(words, wordCounts(candidate))
		switch {
		case score > bestScore:
			best, bestScore, tied = index, score, false
		case score == bestScore && score > 0:
			tied = true
		}
	}
	if tied {
		return -1
	}
	return best
}

func wordCounts(text string) map[string]int {
	counts := map[string]int{}
	for _, word := range strings.FieldsFunc(strings.ToLower(text), func(character rune) bool {
		return !unicode.IsLetter(character) && !unicode.IsDigit(character)
	}) {
		counts[word]++
	}
	return counts
}

// wordOverlap is the harmonic mean of the share of each text's words that the
// other holds, counting a word as often as both repeat it.
func wordOverlap(first, second map[string]int) float64 {
	var common, firstTotal, secondTotal int
	for word, count := range first {
		common += min(count, second[word])
		firstTotal += count
	}
	for _, count := range second {
		secondTotal += count
	}
	if common == 0 {
		return 0
	}
	precision, recall := float64(common)/float64(firstTotal), float64(common)/float64(secondTotal)
	return harmonicMeanFactor * precision * recall / (precision + recall)
}

// BuildFrameDescriptionMatching builds the held-out suite of the clips' middle
// frames for a model that reads one image: the clips, captions, seed and
// distractors of the video suite, each case asking the model to describe the
// frame named by the clip's identity, which the caller cuts, and scoring which
// of the four captions the description matches best.
func BuildFrameDescriptionMatching(spec CaptionMatchingSpec) (DescriptionMatchingSuite, error) {
	rounds, captions, err := captionRounds(spec)
	if err != nil {
		return DescriptionMatchingSuite{}, err
	}
	suite := DescriptionMatchingSuite{Kind: descriptionMatchingKind, Schema: frameMatchingSchema, Source: frameMatchingSource}
	for _, round := range rounds {
		candidates := make([]string, len(round.options))
		for index, option := range round.options {
			candidates[index] = captions[option]
		}
		suite.Cases = append(suite.Cases, DescriptionMatchingCase{
			Name:       suite.Source + "/" + round.vid,
			Prompt:     mediaPrompt(MediaImage, round.vid+imageExtension, frameQuestion),
			MaxTokens:  frameDescriptionTokens,
			Candidates: candidates,
			Answer:     slices.Index(round.options, round.vid),
		})
	}
	return suite, nil
}
