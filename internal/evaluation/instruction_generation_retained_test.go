package evaluation

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

// instructionGeneration is one case's response as an instruction-rules
// evaluation kept it before every evaluation kept its answers in one record
// (caseAnswers): a document for each case, under an alias named by the plan
// and the case. Nothing writes one now. The store holds those of the retained
// evaluations, and the tests of those evaluations read them here.
type instructionGeneration struct {
	ID     artifact.ID `json:"-"`
	Plan   artifact.ID `json:"plan"`
	Case   artifact.ID `json:"case"`
	Result ExactResult `json:"result"`
}

var instructionGenerationCodec = artifact.JSONDocumentCodec(
	"instruction generation", artifact.KindOutput,
	"application/vnd.overgo.instruction-generation+json", "overgo/instruction-generation/v1",
	func(value *instructionGeneration) error {
		if value.Plan.Kind() != artifact.KindProfile || value.Case.Kind() != artifact.KindDatasetShard ||
			value.Result.Name == "" || value.Result.PromptTokens <= 0 || value.Result.GeneratedTokens < 0 {
			return errors.New("evaluation: invalid instruction generation")
		}
		return nil
	},
	func(value instructionGeneration) artifact.ID { return value.ID },
	func(value *instructionGeneration, id artifact.ID) { value.ID = id },
	func(value instructionGeneration) instructionGeneration { return value },
)

func loadInstructionGeneration(ctx context.Context, repository artifact.Repository, alias string, plan Plan, caseID artifact.ID, testCase instructionRulesCase) (instructionGeneration, bool, error) {
	saved, found, err := instructionGenerationCodec.Resolve(ctx, repository, alias)
	if err == nil && found {
		err = saved.validate(plan, caseID, testCase)
	}
	return saved, found, err
}

func (saved instructionGeneration) validate(plan Plan, caseID artifact.ID, testCase instructionRulesCase) error {
	if saved.Plan != plan.identity || saved.Case != caseID || saved.Result.Name != testCase.Name || saved.Result.GeneratedTokens > testCase.MaxTokens {
		return errors.New("evaluation: retained instruction generation differs from its plan or case")
	}
	return nil
}
