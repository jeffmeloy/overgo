package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

func verifyInference(
	repository, path, input string,
	override sessionOverride,
	residency recipe.ResidencyPolicy,
) error {
	revision, err := cleanGoRevision()
	if err != nil {
		return err
	}
	var suite evaluation.ExactSuite
	if err := json.Unmarshal([]byte(input), &suite); err != nil {
		return fmt.Errorf("recipe: decode inference suite: %w", err)
	}
	exactPlan, err := evaluation.CompileExact(suite)
	if err != nil {
		return fmt.Errorf("recipe: compile inference suite: %w", err)
	}
	candidate, err := prepareInferenceCandidate(path, override, residency)
	if err != nil {
		return err
	}
	environment, err := capabilityEnvironment("cuda:0", "cuda")
	if err != nil {
		return err
	}
	evaluationPlan, err := evaluation.BindExact(exactPlan, evaluation.ExactAuthorities{
		ModelDefinition: candidate.resolved.Document.ID,
		RuntimeRecipe:   candidate.definition.ID,
		CodeCommit:      revision,
		Environment:     environment.ID,
		Execution:       evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return fmt.Errorf("recipe: bind inference evaluation: %w", err)
	}
	ctx := context.Background()
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	if _, err := modelrecipe.PublishResolvedModelDefinition(
		ctx, store, candidate.inventory, candidate.resolved,
	); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	if _, published, err := modelrecipe.Status(ctx, store, candidate.definition.ID); err != nil {
		return err
	} else if !published {
		if _, _, err := modelrecipe.PublishCandidate(
			ctx, store, "recipe/candidate/"+candidate.definition.ID.String(), candidate.definition,
		); err != nil {
			return err
		}
	}
	loaded, err := modelrecipe.ResolveCandidateGGUF(path, candidate.definition, candidate.resolved)
	if err != nil {
		return err
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		return err
	}
	started := time.Now()
	results, evaluateErr := evaluation.EvaluateExact(ctx, runner, exactPlan)
	closeErr := runner.Close()
	if evaluateErr != nil || closeErr != nil {
		failure := errors.Join(evaluateErr, closeErr)
		evidence := "plan=" + evaluationPlan.Identity().String() + "; " + strings.ReplaceAll(failure.Error(), "\n", " ")
		if len(evidence) > 1900 {
			evidence = evidence[:1900]
		}
		verification, publishErr := publishCapabilityFailure(
			ctx, store, candidate.definition, revision, time.Since(started),
			"cuda:0", "cuda", "contract=exact; "+evidence, "exact-mismatch",
		)
		if publishErr != nil {
			return errors.Join(failure, publishErr)
		}
		if encodeErr := json.NewEncoder(os.Stdout).Encode(map[string]any{
			"error": failure.Error(), "gate_id": verification.Gate.String(),
			"outcome": "failed", "recipe_id": candidate.definition.ID.String(),
			"run_id": verification.Run.String(),
		}); encodeErr != nil {
			return errors.Join(failure, encodeErr)
		}
		return fmt.Errorf("recipe: exact inference evaluation: %w", failure)
	}
	inputID, err := artifact.IdentifyBytes(artifact.KindDataset, []byte(input))
	if err != nil {
		return err
	}
	exactResults := append([]evaluation.ExactResult(nil), results...)
	for index := range exactResults {
		exactResults[index].WallNS = 0
	}
	encoded, err := json.Marshal(exactResults)
	if err != nil {
		return err
	}
	outputID, err := artifact.IdentifyBytes(artifact.KindOutput, encoded)
	if err != nil {
		return err
	}
	evidence := fmt.Sprintf("contract=exact;plan=%s;cases=%d;input=%s;output=%s", evaluationPlan.Identity(), len(results), inputID, outputID)
	verification, err := publishCapabilityVerification(
		ctx, store, candidate.definition, revision, time.Since(started), "cuda:0", "cuda", evidence,
	)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"gate_id": verification.Gate.String(), "recipe_id": candidate.definition.ID.String(),
		"run_id": verification.Run.String(), "results": results,
	})
}
