package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

func verifyInference(
	repository, path, input string,
	override modelintake.SessionOverride,
	residency recipe.ResidencyPolicy,
) error {
	revision, err := modelintake.CleanRevision(context.Background())
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
	ctx := context.Background()
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, store, path, override, residency)
	if err != nil {
		return err
	}
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		return err
	}
	evaluationPlan, err := evaluation.BindExact(exactPlan, evaluation.ExactAuthorities{
		ModelDefinition: candidate.Resolved.Document.ID,
		RuntimeRecipe:   candidate.Definition.ID,
		CodeCommit:      revision,
		Environment:     environment.ID,
		Execution:       evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return fmt.Errorf("recipe: bind inference evaluation: %w", err)
	}
	if _, err := modelrecipe.PublishResolvedModelDefinition(
		ctx, store, candidate.Inventory, candidate.Resolved,
	); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	if _, published, err := modelrecipe.Status(ctx, store, candidate.Definition.ID); err != nil {
		return err
	} else if !published {
		if _, _, err := modelrecipe.PublishCandidate(
			ctx, store, "recipe/candidate/"+candidate.Definition.ID.String(), candidate.Definition,
		); err != nil {
			return err
		}
	}
	loaded, err := modelrecipe.ResolveCandidateGGUF(path, candidate.Definition, candidate.Resolved)
	if err != nil {
		return err
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		return err
	}
	started := time.Now()
	reportID, evaluateErr := evaluation.EvaluateExactSharded(ctx, store, runner, exactPlan, evaluationPlan, nil)
	closeErr := runner.Close()
	if evaluateErr != nil || closeErr != nil {
		failure := errors.Join(evaluateErr, closeErr)
		evidence := "plan=" + evaluationPlan.Identity().String() + "; " + strings.ReplaceAll(failure.Error(), "\n", " ")
		if len(evidence) > 1900 {
			evidence = evidence[:1900]
		}
		verification, publishErr := modelintake.PublishFailure(
			ctx, store, candidate.Definition, revision, time.Since(started),
			"cuda:0", "cuda", "contract=exact; "+evidence, "exact-mismatch",
		)
		if publishErr != nil {
			return errors.Join(failure, publishErr)
		}
		if encodeErr := json.NewEncoder(os.Stdout).Encode(map[string]any{
			"error": failure.Error(), "gate_id": verification.Gate.String(),
			"outcome": "failed", "recipe_id": candidate.Definition.ID.String(),
			"run_id": verification.Run.String(),
		}); encodeErr != nil {
			return errors.Join(failure, encodeErr)
		}
		return fmt.Errorf("recipe: exact inference evaluation: %w", failure)
	}
	evidence := fmt.Sprintf("contract=exact;plan=%s;report=%s;cases=%d", evaluationPlan.Identity(), reportID, len(suite.Cases))
	verification, err := modelintake.PublishVerification(
		ctx, store, candidate.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence,
	)
	if err != nil {
		return err
	}
	return json.NewEncoder(os.Stdout).Encode(map[string]any{
		"gate_id": verification.Gate.String(), "recipe_id": candidate.Definition.ID.String(),
		"run_id": verification.Run.String(), "report_id": reportID.String(),
	})
}
