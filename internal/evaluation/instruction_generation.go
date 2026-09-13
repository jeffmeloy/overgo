package evaluation

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

// A complete response survives interruption and scorer failure.
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

func recordInstructionCase(ctx context.Context, repository artifact.Repository, generator Generator, plan Plan, testCase InstructionRulesCase) (ExactResult, error) {
	if err := ctx.Err(); err != nil {
		return ExactResult{}, err
	}
	caseID, err := artifact.JSONID(artifact.KindDatasetShard, testCase)
	if err != nil {
		return ExactResult{}, err
	}
	alias := "evaluation/instruction-generations/" + plan.identity.String() + "/" + caseID.String()
	saved, found, err := loadInstructionGeneration(ctx, repository, alias, plan, caseID, testCase)
	if err != nil || found {
		return saved.Result, err
	}
	result, err := recordInstruction(ctx, generator, testCase.Name, testCase.Prompt, testCase.MaxTokens)
	if err != nil {
		return ExactResult{}, err
	}
	saved, err = instructionGenerationCodec.New(instructionGeneration{Plan: plan.identity, Case: caseID, Result: result})
	if err != nil {
		return ExactResult{}, err
	}
	if err := saved.validate(plan, caseID, testCase); err != nil {
		return ExactResult{}, err
	}
	batch, err := instructionGenerationCodec.Batch(alias, saved, artifact.DependencyLineage(saved.ID, plan.identity), []artifact.AliasBinding{{Name: alias, Target: saved.ID}})
	if err != nil {
		return ExactResult{}, err
	}
	// Finish publication after generation; cancellation still prevents the next case.
	publication := context.WithoutCancel(ctx)
	if _, err := artifact.CommitBatch(publication, repository, batch); err != nil {
		// A concurrent publisher may have completed this exact case. Never replace it.
		winner, found, readErr := loadInstructionGeneration(publication, repository, alias, plan, caseID, testCase)
		if readErr == nil && found {
			comparable := winner.Result
			comparable.WallNS = result.WallNS
			if comparable == result {
				return winner.Result, nil
			}
			readErr = errors.New("evaluation: concurrent instruction results disagree")
		}
		return ExactResult{}, fmt.Errorf("evaluation: persist instruction %s: %w", testCase.Name, errors.Join(err, readErr))
	}
	return result, nil
}

func loadInstructionGeneration(ctx context.Context, repository artifact.Repository, alias string, plan Plan, caseID artifact.ID, testCase InstructionRulesCase) (instructionGeneration, bool, error) {
	saved, found, err := instructionGenerationCodec.Resolve(ctx, repository, alias)
	if err == nil && found {
		err = saved.validate(plan, caseID, testCase)
	}
	return saved, found, err
}

func (saved instructionGeneration) validate(plan Plan, caseID artifact.ID, testCase InstructionRulesCase) error {
	if saved.Plan != plan.identity || saved.Case != caseID || saved.Result.Name != testCase.Name || saved.Result.GeneratedTokens > testCase.MaxTokens {
		return errors.New("evaluation: retained instruction generation differs from its plan or case")
	}
	return nil
}
