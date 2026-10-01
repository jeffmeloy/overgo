package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/jsonfile"
	"overgo/internal/tokenizer"
)

const smokeOraclePath = "docs/verification/smoke-oracles.json"

// Declarations bind a narrow smoke claim to an exact active recipe.
type smokeOracle struct {
	Model         artifact.ID                     `json:"model"`
	Recipe        artifact.ID                     `json:"recipe"`
	Domain        string                          `json:"domain"`
	Chat          bool                            `json:"chat"`
	Generation    *evaluation.GeneratedAnswerCase `json:"generation,omitempty"`
	ReferenceFile string                          `json:"reference_file,omitzero"`
	Reference     artifact.ID                     `json:"reference,omitzero"`
	Claim         string                          `json:"claim"`
}

type smokeReferenceCase struct {
	Text             string              `json:"text"`
	IDs              []tokenizer.TokenID `json:"ids"`
	LogProbabilities []float64           `json:"continuation_log_probabilities"`
}

type smokeObservation struct {
	Oracle     artifact.ID                  `json:"oracle"`
	Generation *evaluation.ExactResult      `json:"generation,omitempty"`
	Scores     []inference.PerplexityResult `json:"scores,omitempty"`
	Reference  artifact.ID                  `json:"reference,omitzero"`
	Claim      string                       `json:"claim"`
}

func readSmokeOracles(path string, entries []discovery.Entry, selected string) (map[artifact.ID]smokeOracle, error) {
	var declarations []smokeOracle
	if err := jsonfile.DecodeStrict(path, &declarations); err != nil {
		return nil, err
	}
	if selected != "" {
		declarations = slices.DeleteFunc(declarations, func(value smokeOracle) bool { return value.Model.String() != selected })
	}
	return bindSmokeOracles(declarations, entries)
}

func bindSmokeOracles(declarations []smokeOracle, entries []discovery.Entry) (map[artifact.ID]smokeOracle, error) {
	if len(declarations) != len(entries) || len(entries) == 0 {
		return nil, errors.New("smoke: declaration denominator differs from live inference catalog")
	}
	bound := make(map[artifact.ID]smokeOracle, len(declarations))
	for _, declaration := range declarations {
		if _, duplicate := bound[declaration.Model]; duplicate {
			return nil, errors.New("smoke: duplicate model declaration")
		}
		if declaration.Model.Kind() != artifact.KindModel || declaration.Recipe.Kind() != artifact.KindRecipe || declaration.Claim == "" || declaration.Domain == "" {
			return nil, errors.New("smoke: missing model, recipe, domain or claim")
		}
		if declaration.Generation == nil && declaration.ReferenceFile == "" {
			return nil, errors.New("smoke: no behavioral oracle")
		}
		if declaration.ReferenceFile != "" {
			if declaration.Reference.Kind() != artifact.KindEvidence || !filepath.IsLocal(declaration.ReferenceFile) {
				return nil, errors.New("smoke: unbound native reference")
			}
		} else if declaration.Reference.Valid() {
			return nil, errors.New("smoke: reference file absent")
		}
		if declaration.Generation != nil {
			if _, err := evaluation.CompileGeneratedAnswer(evaluation.GeneratedAnswerSuite{
				Kind: evaluation.GeneratedAnswerKind, Schema: "overgo/smoke-answer/v1", Source: declaration.Claim,
				Transforms: []string{evaluation.TransformTrimSpace}, Cases: []evaluation.GeneratedAnswerCase{*declaration.Generation},
			}); err != nil {
				return nil, err
			}
		}
		bound[declaration.Model] = declaration
	}
	seen := map[artifact.ID]bool{}
	for _, entry := range entries {
		declaration, exists := bound[entry.Model]
		if !exists || seen[entry.Model] || declaration.Recipe != entry.Recipe || !entry.Present || entry.Stale != "" {
			return nil, fmt.Errorf("smoke: absent, duplicate, stale or mismatched activation %s", entry.Model)
		}
		seen[entry.Model] = true
	}
	return bound, nil
}

func readSmokeReference(oracle smokeOracle) ([]smokeReferenceCase, error) {
	data, err := os.ReadFile(oracle.ReferenceFile)
	if err != nil {
		return nil, err
	}
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, data)
	if err != nil || id != oracle.Reference {
		return nil, errors.New("smoke: native reference identity differs")
	}
	var reference struct {
		Model  artifact.ID          `json:"model"`
		Source map[string]string    `json:"source_sha256"`
		Cases  []smokeReferenceCase `json:"cases"`
	}
	// Capture provenance remains in the identified document; scoring reads its cases.
	if err := json.Unmarshal(data, &reference); err != nil {
		return nil, err
	}
	if reference.Model != oracle.Model || len(reference.Source) == 0 || len(reference.Cases) == 0 {
		return nil, errors.New("smoke: native reference has no cases")
	}
	seen := map[string]bool{}
	for _, testCase := range reference.Cases {
		if testCase.Text == "" || seen[testCase.Text] || len(testCase.IDs) < 2 || len(testCase.LogProbabilities) != len(testCase.IDs)-1 {
			return nil, errors.New("smoke: incomplete or duplicate native case")
		}
		seen[testCase.Text] = true
		distinct := false
		for _, value := range testCase.LogProbabilities {
			if math.IsNaN(value) || math.IsInf(value, 0) || value > 0 {
				return nil, errors.New("smoke: invalid native likelihood")
			}
			distinct = distinct || value != testCase.LogProbabilities[0]
		}
		if !distinct {
			return nil, errors.New("smoke: native ordering oracle is vacuous")
		}
	}
	return reference.Cases, nil
}

// Chat prompts are already formatted through the runner's artifact declaration.
type smokeGenerator struct {
	*inference.Runner
	parseSpecial bool
}

// Generate preserves special tokens in artifact-formatted chat prompts.
func (generator smokeGenerator) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	options.ParseSpecial = generator.parseSpecial
	return generator.Runner.Generate(ctx, prompt, options)
}

func acceptSmokeAnswer(testCase evaluation.GeneratedAnswerCase, result evaluation.ExactResult) error {
	if result.Name != testCase.Name || result.PromptTokens <= 0 || result.GeneratedTokens <= 0 || result.GeneratedTokens > testCase.MaxTokens ||
		!slices.ContainsFunc(testCase.Answers, func(answer string) bool { return strings.TrimSpace(result.Text) == strings.TrimSpace(answer) }) {
		return fmt.Errorf("smoke: %s produced an unaccepted continuation %q", testCase.Name, result.Text)
	}
	return nil
}

// This guard checks exact tokenization and native likelihood ordering. It does
// not assert a numerical-error tolerance or broad scientific/model quality.
func acceptSmokeScores(reference smokeReferenceCase, result inference.PerplexityResult) error {
	if result.TokenCount != len(reference.IDs) || result.EvaluatedTokens != len(reference.LogProbabilities) || len(result.Scores) != result.EvaluatedTokens {
		return errors.New("smoke: scoring denominator differs")
	}
	for index, score := range result.Scores {
		if score.Position != index+1 || score.TokenID != reference.IDs[index+1] || math.IsNaN(score.NegativeLogLik) || math.IsInf(score.NegativeLogLik, 0) || score.NegativeLogLik < 0 {
			return errors.New("smoke: invalid token scoring observation")
		}
		for previous := range index {
			expected := reference.LogProbabilities[previous] - reference.LogProbabilities[index]
			observed := score.NegativeLogLik - result.Scores[previous].NegativeLogLik
			if expected != 0 && (expected > 0) != (observed > 0) || expected != 0 && observed == 0 {
				return errors.New("smoke: native likelihood ordering regressed")
			}
		}
	}
	return nil
}

func executeSmoke(ctx context.Context, runner *inference.Runner, oracle smokeOracle, reference []smokeReferenceCase) (smokeObservation, error) {
	id, err := artifact.JSONID(artifact.KindProfile, oracle)
	result := smokeObservation{Oracle: id, Reference: oracle.Reference, Claim: oracle.Claim}
	if err != nil {
		return result, err
	}
	if oracle.Generation != nil {
		testCase := *oracle.Generation
		prompt := testCase.Prompt
		if oracle.Chat {
			prompt, err = runner.FormatChatWithOptions([]inference.ChatMessage{{Role: inference.ChatRoleUser, Content: prompt}}, inference.ChatFormatOptions{AddGenerationPrompt: true, EnableThinking: false})
			if err != nil {
				return result, err
			}
		}
		generated, err := evaluation.Record(ctx, smokeGenerator{runner, oracle.Chat}, testCase.Name, prompt, testCase.MaxTokens)
		result.Generation = &generated
		if err != nil {
			return result, err
		}
		if err := acceptSmokeAnswer(testCase, generated); err != nil {
			return result, err
		}
	}
	for _, testCase := range reference {
		scores, err := runner.Perplexity(ctx, testCase.Text)
		result.Scores = append(result.Scores, scores)
		if err != nil {
			return result, err
		}
		if err := acceptSmokeScores(testCase, scores); err != nil {
			return result, err
		}
	}
	return result, nil
}
