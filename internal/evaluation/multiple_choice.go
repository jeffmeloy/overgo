package evaluation

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/sequencescore"
)

const (
	MultipleChoiceKind  = "multiple-choice"
	AggregationAccuracy = "accuracy"
	// TieBreakFirst accepts the first maximum when candidate scores tie.
	TieBreakFirst              = "first"
	multipleChoiceMediaType    = "application/vnd.overgo.multiple-choice-dataset+json"
	multipleChoiceSchema       = "overgo/multiple-choice-dataset/v1"
	multipleChoiceSplitMedia   = "application/vnd.overgo.multiple-choice-split+json"
	multipleChoiceSplitSchema  = "overgo/multiple-choice-split/v1"
	multipleChoiceReportMedia  = "application/vnd.overgo.multiple-choice-report+json"
	multipleChoiceReportSchema = "overgo/multiple-choice-report/v1"
)

var (
	multipleChoiceDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: multipleChoiceMediaType, Schema: multipleChoiceSchema,
	}
	multipleChoiceSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: multipleChoiceSplitMedia, Schema: multipleChoiceSplitSchema,
	}
	multipleChoiceReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: multipleChoiceReportMedia, Schema: multipleChoiceReportSchema,
	}
)

type MultipleChoiceCase struct {
	Name                string   `json:"name"`
	Prompt              string   `json:"prompt"`
	Candidates          []string `json:"candidates"`
	CandidateCharacters []uint64 `json:"candidate_characters,omitzero"`
	Answer              int      `json:"answer"`
}

type MultipleChoiceSuite struct {
	Kind          string                      `json:"kind"`
	Schema        string                      `json:"schema"`
	Source        string                      `json:"source"`
	Normalization sequencescore.Normalization `json:"normalization"`
	TieBreak      string                      `json:"tie_break,omitzero"`
	Aggregation   string                      `json:"aggregation"`
	Cases         []MultipleChoiceCase        `json:"cases"`
}

type multipleChoicePlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    MultipleChoiceSuite
	// delimiter is what a likelihood scoring puts between a case's prompt and
	// each of its candidates, where the suite's candidates are bare labels
	// and do not carry it: lm-eval's target delimiter, for a task whose
	// choices it writes bare. It is no part of the suite, so the suite's
	// identity, and with it every plan that asks its questions another way,
	// is what it was; a likelihood plan that scores with it names a method of
	// its own.
	delimiter string
}

// delimitedChoiceMethod names the likelihood scoring of a suite of bare
// labels with the target delimiter before each, so that its records are
// another plan than those scored with each label glued to the answer cue,
// which every earlier likelihood record of such a suite was.
const delimitedChoiceMethod = "likelihood-target-delimiter"

// delimited returns the suite as a likelihood scoring is to score it: each
// candidate with the plan's delimiter before it.
func (p multipleChoicePlan) delimited() MultipleChoiceSuite {
	if p.delimiter == "" {
		return p.suite
	}
	scored := p.suite
	scored.Cases = make([]MultipleChoiceCase, len(p.suite.Cases))
	for index, testCase := range p.suite.Cases {
		testCase.Candidates = slices.Clone(testCase.Candidates)
		for candidate := range testCase.Candidates {
			testCase.Candidates[candidate] = p.delimiter + testCase.Candidates[candidate]
		}
		scored.Cases[index] = testCase
	}
	return scored
}

type choiceObservation struct {
	Name     string                `json:"name"`
	Values   []float64             `json:"values"`
	Scores   []sequencescore.Score `json:"scores,omitzero"`
	Selected int                   `json:"selected"`
	Answer   int                   `json:"answer"`
	Tied     bool                  `json:"tied"`
	// Raw is the generated answer text when the case was answered by
	// generation through the chat template; likelihood scoring leaves
	// it empty and Selected below zero means no candidate letter was
	// found in it.
	Raw string `json:"raw,omitzero"`
}

type MultipleChoiceReport struct {
	ID      artifact.ID `json:"-"`
	Version uint16      `json:"version"`
	Plan    artifact.ID `json:"plan"`
	Dataset artifact.ID `json:"dataset"`
	// Answers names what the plan's execution was asked (Plan.observed): the
	// store holds, under it, each case's answer as the model gave it, which
	// the observations here were scored from. A report written before answers
	// were kept apart from their scores names none.
	Answers      artifact.ID         `json:"answers,omitzero"`
	Observations []choiceObservation `json:"observations"`
	Accuracy     float64             `json:"accuracy"`
}

type accuracyGroup struct {
	Name     string  `json:"name"`
	Correct  uint64  `json:"correct"`
	Total    uint64  `json:"total"`
	Accuracy float64 `json:"accuracy"`
}

type continuationScorer interface {
	ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error)
}

func CompileMultipleChoice(suite MultipleChoiceSuite) (multipleChoicePlan, error) {
	if suite.Kind != MultipleChoiceKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || suite.Aggregation != AggregationAccuracy ||
		len(suite.Cases) == 0 || suite.Normalization != sequencescore.NormalizationSum &&
		suite.Normalization != sequencescore.NormalizationMean && suite.Normalization != sequencescore.NormalizationCharacters ||
		suite.TieBreak != "" && suite.TieBreak != TieBreakFirst {
		return multipleChoicePlan{}, errors.New("evaluation: invalid multiple-choice suite")
	}
	suite.Cases = slices.Clone(suite.Cases)
	names := make(map[string]struct{}, len(suite.Cases))
	for index := range suite.Cases {
		testCase := &suite.Cases[index]
		testCase.Candidates = slices.Clone(testCase.Candidates)
		testCase.CandidateCharacters = slices.Clone(testCase.CandidateCharacters)
		if suite.Normalization == sequencescore.NormalizationCharacters && len(testCase.CandidateCharacters) != len(testCase.Candidates) {
			return multipleChoicePlan{}, errors.New("evaluation: candidate character extent differs")
		}
		if err := validateMultipleChoiceCase(*testCase); err != nil {
			return multipleChoicePlan{}, err
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return multipleChoicePlan{}, errors.New("evaluation: duplicate multiple-choice case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, err := artifact.JSONID(artifact.KindProfile, suite)
	if err != nil {
		return multipleChoicePlan{}, err
	}
	dataset, err := artifact.JSONID(artifact.KindDataset, suite.Cases)
	if err != nil {
		return multipleChoicePlan{}, err
	}
	split, err := artifact.JSONID(artifact.KindDatasetShard, struct {
		Dataset artifact.ID `json:"dataset"`
	}{Dataset: dataset})
	if err != nil {
		return multipleChoicePlan{}, err
	}
	return multipleChoicePlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func validateMultipleChoiceCase(testCase MultipleChoiceCase) error {
	if strings.TrimSpace(testCase.Name) == "" || testCase.Prompt == "" || len(testCase.Candidates) < 2 ||
		testCase.Answer < 0 || testCase.Answer >= len(testCase.Candidates) {
		return errors.New("evaluation: invalid multiple-choice case")
	}
	if len(testCase.CandidateCharacters) != 0 && len(testCase.CandidateCharacters) != len(testCase.Candidates) {
		return errors.New("evaluation: candidate character extent differs")
	}
	if slices.Contains(testCase.CandidateCharacters, 0) {
		return errors.New("evaluation: candidate character count is zero")
	}
	candidates := make(map[string]struct{}, len(testCase.Candidates))
	for _, candidate := range testCase.Candidates {
		if candidate == "" {
			return errors.New("evaluation: empty multiple-choice candidate")
		}
		if _, duplicate := candidates[candidate]; duplicate {
			return errors.New("evaluation: duplicate multiple-choice candidate")
		}
		candidates[candidate] = struct{}{}
	}
	return nil
}

func BindMultipleChoice(compiled multipleChoicePlan, authorities ExactAuthorities) (Plan, error) {
	scorer := struct {
		Version       uint16                      `json:"version"`
		Kind          string                      `json:"kind"`
		Normalization sequencescore.Normalization `json:"normalization"`
		TieBreak      string                      `json:"tie_break,omitzero"`
		Aggregation   string                      `json:"aggregation"`
		// Method is empty for likelihood scoring (every earlier plan)
		// and names the generated-letter method under the chat-template
		// protocol, so the two are distinct scorer authorities.
		Method string `json:"method,omitzero"`
	}{
		Version: artifact.InitialDocumentVersion, Kind: MultipleChoiceKind,
		Normalization: compiled.suite.Normalization, TieBreak: compiled.suite.TieBreak, Aggregation: compiled.suite.Aggregation,
	}
	if prompting := authorities.Execution.Prompting; prompting == PromptingChatTemplate || prompting == PromptingHostedChat || prompting == PromptingReasoning {
		if err := validateChatChoiceSuite(compiled.suite.Cases); err != nil {
			return Plan{}, err
		}
		scorer.Method = chatChoiceMethod
		switch prompting {
		case PromptingHostedChat:
			scorer.Method = hostedChoiceMethod
		case PromptingReasoning:
			// parser = the scorer's method, so another parser reads answers already given without asking again.
			scorer.Method = authorities.Execution.Reasoning.Parser
			if _, known := reasonedParsers[scorer.Method]; !known {
				return Plan{}, fmt.Errorf("evaluation: the %s protocol reads answers by parser %q, which is none of the %d this code holds", prompting.Label(), scorer.Method, len(reasonedParsers))
			}
		}
	} else if compiled.delimiter != "" {
		scorer.Method = delimitedChoiceMethod
	}
	return bindAsked(compiled.dataset, compiled.split, compiled.identity, compiled.suite, scorer, authorities)
}

func evaluateMultipleChoice(
	ctx context.Context,
	repository artifact.Repository,
	scorer continuationScorer,
	compiled multipleChoicePlan,
	plan Plan,
) (MultipleChoiceReport, error) {
	if ctx == nil || repository == nil || scorer == nil || compiled.identity != plan.body.CaseProfile {
		return MultipleChoiceReport{}, errors.New("evaluation: multiple-choice authority differs")
	}
	contents, err := compiled.contents()
	if err != nil {
		return MultipleChoiceReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return MultipleChoiceReport{}, err
	}
	report := MultipleChoiceReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
	}
	report.Observations, report.Answers, report.Accuracy, err = scoreMultipleChoice(ctx, repository, plan, scorer, compiled.suite)
	if err != nil {
		return MultipleChoiceReport{}, err
	}
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return MultipleChoiceReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, multipleChoiceReportContract, report, report.Answers,
	); err != nil {
		return MultipleChoiceReport{}, err
	}
	return report, nil
}

// choiceAnswer is what a model gave one case of a choice suite, before it is
// scored: under a protocol that generates, what it wrote, and under one that
// scores continuations, the score it gave each candidate.
type choiceAnswer struct {
	Written *ExactResult          `json:"written,omitempty"`
	Scores  []sequencescore.Score `json:"scores,omitempty"`
}

// askedChoice is what one case of a choice suite gives the model: its name,
// its prompt and its candidates, and not the answer it is scored against.
type askedChoice struct {
	Name       string   `json:"name"`
	Prompt     string   `json:"prompt"`
	Candidates []string `json:"candidates"`
}

// scoreMultipleChoice answers each case of a suite and scores the answers. A
// case's answer is the one the store holds for what the plan's execution is
// asked, or the model's now, which is written down as it is given
// (loadAnswers): a suite scored again, under another normalization or
// tie-break or against corrected answers, puts no case to the model, and one
// that stopped part way resumes after the cases it answered.
//   - plan under the reasoning protocol => its cases are published as development evidence before any is asked
func scoreMultipleChoice(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	scorer continuationScorer,
	suite MultipleChoiceSuite,
) ([]choiceObservation, artifact.ID, float64, error) {
	asked := make([]askedChoice, len(suite.Cases))
	for index, testCase := range suite.Cases {
		asked[index] = askedChoice{Name: testCase.Name, Prompt: testCase.Prompt, Candidates: testCase.Candidates}
	}
	held, err := loadAnswers[choiceAnswer](ctx, repository, plan, asked)
	if err != nil {
		return nil, artifact.ID{}, 0, err
	}
	observations := make([]choiceObservation, len(suite.Cases))
	correct := 0
	progress := trackProgress(ctx, suite.Source, len(suite.Cases))
	held.said(progress, len(suite.Cases))
	chat, generative := scorer.(chatChoiceRuntime)
	reasoner, _ := scorer.(reasoningRuntime)
	reasons := plan.reasoning != reasoningPolicy{}
	if reasons && reasoner == nil {
		return nil, artifact.ID{}, 0, fmt.Errorf("evaluation: the %s protocol asks a runtime that shapes a turn with thinking switched and reads what follows a thought, and %T does neither", PromptingReasoning.Label(), scorer)
	}
	if reasons {
		if err := exposeDevelopment(ctx, repository, plan, suite.Cases); err != nil {
			return nil, artifact.ID{}, 0, err
		}
	}
	var probe chatProbe
	for index, testCase := range suite.Cases {
		answer, err := held.answer(ctx, index, func() (choiceAnswer, error) {
			if reasons {
				written, err := answerChoiceByReasoning(ctx, reasoner, plan.reasoning, testCase)
				return choiceAnswer{Written: &written}, err
			}
			if generative {
				written, err := answerChoiceByGeneration(ctx, chat, testCase)
				return choiceAnswer{Written: &written}, err
			}
			scores, err := scorer.ScoreContinuations(ctx, testCase.Prompt, testCase.Candidates)
			if err == nil && len(scores) != len(testCase.Candidates) {
				err = errors.New("evaluation: candidate score extent differs")
			}
			return choiceAnswer{Scores: scores}, err
		})
		if err != nil {
			return nil, artifact.ID{}, 0, err
		}
		if reasons {
			observations[index], err = readReasonedChoice(testCase, answer, plan.reasoning, reasoner)
		} else {
			observations[index], err = scoreChoice(testCase, answer, suite.Normalization)
		}
		if err != nil {
			return nil, artifact.ID{}, 0, errors.Join(err, held.write(ctx))
		}
		if answer.Written != nil {
			letters := chatChoiceLetters(testCase)
			if index == 0 {
				progress.sample(fmt.Sprintf("first generated answer %s: %q against candidates %q",
					observations[index].Name, observations[index].Raw, letters))
			}
			if err := probe.judge(observations[index], letters, index == len(suite.Cases)-1); err != nil {
				return nil, artifact.ID{}, 0, errors.Join(err, held.write(ctx))
			}
		}
		hit := choiceCorrect(observations[index], suite.TieBreak)
		if hit {
			correct++
		}
		progress.hit(hit)
	}
	if err := held.write(ctx); err != nil {
		return nil, artifact.ID{}, 0, err
	}
	accuracy := float64(correct) / float64(len(observations))
	if math.IsNaN(accuracy) || math.IsInf(accuracy, 0) {
		return nil, artifact.ID{}, 0, errors.New("evaluation: multiple-choice accuracy is not finite")
	}
	return observations, held.observed, accuracy, nil
}

// scoreChoice reads one case's score from the answer the model gave it. What
// a model wrote selects the first candidate letter standing alone in it, or
// nothing; the scores it gave the candidates select the one the suite's
// normalization finds likeliest.
func scoreChoice(testCase MultipleChoiceCase, answer choiceAnswer, normalization sequencescore.Normalization) (choiceObservation, error) {
	observation := choiceObservation{Name: testCase.Name, Answer: testCase.Answer}
	if answer.Written != nil {
		observation.Raw, observation.Selected = answer.Written.Text, unmatchedChoice
		if index, matched := matchChoiceLetter(answer.Written.Text, chatChoiceLetters(testCase)); matched {
			observation.Selected = index
		}
		return observation, nil
	}
	if len(answer.Scores) != len(testCase.Candidates) {
		return choiceObservation{}, fmt.Errorf("evaluation: case %s was answered with %d scores for its %d candidates", testCase.Name, len(answer.Scores), len(testCase.Candidates))
	}
	scores := slices.Clone(answer.Scores)
	if normalization == sequencescore.NormalizationCharacters {
		for index := range scores {
			scores[index].Characters = testCase.CandidateCharacters[index]
		}
	}
	selection, err := sequencescore.Select(scores, normalization)
	if err != nil {
		return choiceObservation{}, err
	}
	observation.Values, observation.Scores, observation.Selected, observation.Tied = selection.Values, scores, selection.Index, selection.Tied
	return observation, nil
}

func choiceCorrect(observation choiceObservation, tieBreak string) bool {
	return observation.Selected == observation.Answer && (!observation.Tied || tieBreak == TieBreakFirst)
}

func aggregateChoiceAccuracy(groups []string, observations []choiceObservation, tieBreak string) ([]accuracyGroup, error) {
	if len(groups) != len(observations) {
		return nil, errors.New("evaluation: choice groups differ from observations")
	}
	correct := make([]bool, len(observations))
	for index, observation := range observations {
		correct[index] = choiceCorrect(observation, tieBreak)
	}
	return aggregateAccuracy(groups, correct)
}

func aggregateAccuracy(groups []string, correct []bool) ([]accuracyGroup, error) {
	if len(groups) != len(correct) {
		return nil, errors.New("evaluation: groups differ from outcomes")
	}
	byName := make(map[string]*accuracyGroup)
	for index, group := range groups {
		metric := byName[group]
		if metric == nil {
			metric = &accuracyGroup{Name: group}
			byName[group] = metric
		}
		metric.Total++
		if correct[index] {
			metric.Correct++
		}
	}
	result := make([]accuracyGroup, 0, len(byName))
	for _, metric := range byName {
		metric.Accuracy = float64(metric.Correct) / float64(metric.Total)
		result = append(result, *metric)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Name < result[j].Name })
	return result, nil
}

func scoreChoiceGroups(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	scorer continuationScorer,
	suite MultipleChoiceSuite,
	groups []string,
) ([]choiceObservation, []accuracyGroup, artifact.ID, float64, error) {
	observations, answers, accuracy, err := scoreMultipleChoice(ctx, repository, plan, scorer, suite)
	if err != nil {
		return nil, nil, artifact.ID{}, 0, err
	}
	metrics, err := aggregateChoiceAccuracy(groups, observations, suite.TieBreak)
	return observations, metrics, answers, accuracy, err
}

func (p multipleChoicePlan) contents() ([]artifact.Content, error) {
	return datasetContents(
		p.dataset, p.split, p.suite.Cases, multipleChoiceDatasetContract, multipleChoiceSplitContract,
	)
}
