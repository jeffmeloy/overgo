package evaluation

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"overgo/internal/artifact"
	"overgo/internal/inference"
)

const (
	// functionCallKind names the suite whose cases ask a model to call one
	// function and score the call it writes against the calls a case accepts.
	functionCallKind         = "function-call"
	functionCallMediaType    = "application/vnd.overgo.function-call-dataset+json"
	functionCallSchema       = "overgo/function-call-dataset/v1"
	functionCallSplitMedia   = "application/vnd.overgo.function-call-split+json"
	functionCallSplitSchema  = "overgo/function-call-split/v1"
	functionCallReportMedia  = "application/vnd.overgo.function-call-report+json"
	functionCallReportSchema = "overgo/function-call-report/v1"
	// toolCallOpening is the marker a function-calling model may write before
	// the JSON list of its calls.
	toolCallOpening = "<tool_call>"
	// ignoredStringRunes are what a string value's comparison drops: the
	// Berkeley Function-Calling Leaderboard's own, so a call is read as it is.
	ignoredStringRunes = " ,./-_*^"
)

var (
	functionCallDatasetContract = artifact.DocumentContract{
		Kind: artifact.KindDataset, MediaType: functionCallMediaType, Schema: functionCallSchema,
	}
	functionCallSplitContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: functionCallSplitMedia, Schema: functionCallSplitSchema,
	}
	functionCallReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: functionCallReportMedia, Schema: functionCallReportSchema,
	}
)

// FunctionCallCase asks for one call of one function. Arguments holds, for
// each parameter, the values the case accepts; an empty string among them
// lets the call leave the parameter out.
type FunctionCallCase struct {
	Name      string                       `json:"name"`
	Prompt    string                       `json:"prompt"`
	MaxTokens int                          `json:"max_tokens"`
	Function  string                       `json:"function"`
	Arguments map[string][]json.RawMessage `json:"arguments"`
}

// FunctionCallSuite is a set of function-call cases from one source.
type FunctionCallSuite struct {
	Kind   string             `json:"kind"`
	Schema string             `json:"schema"`
	Source string             `json:"source"`
	Cases  []FunctionCallCase `json:"cases"`
}

type functionCallPlan struct {
	identity artifact.ID
	dataset  artifact.ID
	split    artifact.ID
	suite    FunctionCallSuite
}

type functionCallObservation struct {
	Name     string `json:"name"`
	Raw      string `json:"raw"`
	Accepted bool   `json:"accepted"`
	Why      string `json:"why,omitempty"`
}

type functionCallReport struct {
	ID           artifact.ID               `json:"-"`
	Version      uint16                    `json:"version"`
	Plan         artifact.ID               `json:"plan"`
	Dataset      artifact.ID               `json:"dataset"`
	Observations []functionCallObservation `json:"observations"`
	Accuracy     float64                   `json:"accuracy"`
}

// compileFunctionCall validates and identifies the suite.
func compileFunctionCall(suite FunctionCallSuite) (functionCallPlan, error) {
	if suite.Kind != functionCallKind || strings.TrimSpace(suite.Schema) == "" ||
		strings.TrimSpace(suite.Source) == "" || len(suite.Cases) == 0 {
		return functionCallPlan{}, errors.New("evaluation: invalid function-call suite")
	}
	names := make(map[string]struct{}, len(suite.Cases))
	for _, testCase := range suite.Cases {
		if strings.TrimSpace(testCase.Name) == "" || testCase.Prompt == "" || testCase.MaxTokens <= 0 ||
			strings.TrimSpace(testCase.Function) == "" {
			return functionCallPlan{}, errors.New("evaluation: invalid function-call case")
		}
		for parameter, accepted := range testCase.Arguments {
			if parameter == "" || len(accepted) == 0 {
				return functionCallPlan{}, errors.New("evaluation: a function-call parameter accepts no value")
			}
		}
		if _, duplicate := names[testCase.Name]; duplicate {
			return functionCallPlan{}, errors.New("evaluation: duplicate function-call case")
		}
		names[testCase.Name] = struct{}{}
	}
	identity, dataset, split, err := suiteIdentities(suite, suite.Cases)
	if err != nil {
		return functionCallPlan{}, err
	}
	return functionCallPlan{identity: identity, dataset: dataset, split: split, suite: suite}, nil
}

func bindFunctionCall(compiled functionCallPlan, authorities ExactAuthorities) (Plan, error) {
	return bindKindPlan(compiled.dataset, compiled.split, compiled.identity, compiled.suite, functionCallKind, authorities)
}

// evaluateFunctionCall asks every case of the model, scores the call it
// writes, and publishes the report through the campaign ledger.
func evaluateFunctionCall(
	ctx context.Context,
	repository artifact.Repository,
	generator Generator,
	compiled functionCallPlan,
	plan Plan,
) (functionCallReport, error) {
	if ctx == nil || repository == nil || generator == nil || compiled.identity != plan.body.CaseProfile {
		return functionCallReport{}, errors.New("evaluation: function-call authority differs")
	}
	contents, err := datasetContents(
		compiled.dataset, compiled.split, compiled.suite.Cases, functionCallDatasetContract, functionCallSplitContract,
	)
	if err != nil {
		return functionCallReport{}, err
	}
	if err := publishPlanAuthorities(ctx, repository, plan, contents); err != nil {
		return functionCallReport{}, err
	}
	report := functionCallReport{
		Version: artifact.InitialDocumentVersion, Plan: plan.identity, Dataset: compiled.dataset,
		Observations: make([]functionCallObservation, len(compiled.suite.Cases)),
	}
	correct := 0
	progress := trackProgress(ctx, compiled.suite.Source, len(compiled.suite.Cases))
	for index, testCase := range compiled.suite.Cases {
		_, text, err := generator.Generate(ctx, testCase.Prompt, inference.GenerateOptions{MaxNewTokens: testCase.MaxTokens})
		if err != nil {
			return functionCallReport{}, fmt.Errorf("evaluation: function-call case %q: %w", testCase.Name, err)
		}
		why := scoreFunctionCall(text, testCase)
		if why == "" {
			correct++
		}
		report.Observations[index] = functionCallObservation{Name: testCase.Name, Raw: strings.ToValidUTF8(text, string(utf8.RuneError)), Accepted: why == "", Why: why}
		progress.hit(why == "")
	}
	report.Accuracy = float64(correct) / float64(len(report.Observations))
	id, err := artifact.JSONID(artifact.KindEvaluation, report)
	if err != nil {
		return functionCallReport{}, err
	}
	report.ID = id
	if err := publishCampaignDocument(
		ctx, repository, report.Plan, report.Dataset, report.ID, functionCallReportContract, report,
	); err != nil {
		return functionCallReport{}, err
	}
	return report, nil
}

// scoreFunctionCall reads the model's output as a JSON list of calls and
// returns why its first is not the call the case accepts, or "" for an accepted one:
// the function's name, each argument one of its accepted values, and every
// parameter without an omittable value present.
func scoreFunctionCall(output string, testCase FunctionCallCase) string {
	payload := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(output), toolCallOpening))
	var calls []struct {
		Name      string                     `json:"name"`
		Arguments map[string]json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(payload), &calls); err != nil {
		return "the output is not a JSON list of calls"
	}
	// The leaderboard's simple category asks for one call, and a small model
	// repeats its call or follows it with another: the first is its answer.
	if len(calls) == 0 {
		return "the output holds no call"
	}
	call := calls[0]
	if call.Name != testCase.Function && !strings.HasSuffix(call.Name, "."+testCase.Function) {
		return fmt.Sprintf("calls %q, want %q", call.Name, testCase.Function)
	}
	for parameter, value := range call.Arguments {
		accepted, known := testCase.Arguments[parameter]
		if !known {
			return fmt.Sprintf("passes %q, which the function does not take", parameter)
		}
		if !slices.ContainsFunc(accepted, func(candidate json.RawMessage) bool { return sameValue(candidate, value) }) {
			return fmt.Sprintf("passes %s as %s, which is not an accepted value", parameter, bytes.TrimSpace(value))
		}
	}
	for parameter, accepted := range testCase.Arguments {
		_, present := call.Arguments[parameter]
		if !present && !slices.ContainsFunc(accepted, func(candidate json.RawMessage) bool { return string(bytes.TrimSpace(candidate)) == `""` }) {
			return fmt.Sprintf("leaves out %q, which the call needs", parameter)
		}
	}
	return ""
}

// sameValue reports whether two JSON values are the same: numbers by value,
// strings as the leaderboard reads them, lists and objects member by member.
func sameValue(left, right json.RawMessage) bool {
	var first, second any
	if json.Unmarshal(left, &first) != nil || json.Unmarshal(right, &second) != nil {
		return false
	}
	return equalValues(first, second)
}

func equalValues(left, right any) bool {
	switch first := left.(type) {
	case string:
		second, ok := right.(string)
		return ok && standardString(first) == standardString(second)
	case []any:
		second, ok := right.([]any)
		return ok && slices.EqualFunc(first, second, equalValues)
	case map[string]any:
		second, ok := right.(map[string]any)
		if !ok || len(first) != len(second) {
			return false
		}
		for key, value := range first {
			other, present := second[key]
			if !present || !equalValues(value, other) {
				return false
			}
		}
		return true
	default:
		return left == right
	}
}

// standardString is a string as the leaderboard compares it: lower case,
// without the separators it ignores, and with one kind of quote.
func standardString(value string) string {
	dropped := strings.Map(func(character rune) rune {
		if strings.ContainsRune(ignoredStringRunes, character) {
			return -1
		}
		return character
	}, value)
	return strings.ReplaceAll(strings.ToLower(dropped), "'", `"`)
}
