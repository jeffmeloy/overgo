// Package libraryintake is the model intake the library page drives: a
// downloaded model registers with its facts, its candidate recipe and any
// projector beside it, and validates by recording and replaying an exact
// suite whose evidence activates the recipe. The HTTP package receives
// these entry points from the assembly root and names no intake package.
package libraryintake

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// refusal marks an intake error the request itself caused (an artifact the
// store cannot admit), which a route answers as a refusal rather than a
// failure; the HTTP package recognises it by the LibraryRefused method
// alone, and the embedded cause speaks for itself.
type refusal struct{ error }

// LibraryRefused reports the refusal; one without a cause is no refusal.
func (r refusal) LibraryRefused() bool { return r.error != nil }

// ModelFiles resolves the model file a register or validate names and the
// projector beside it: a GGUF file, or a downloaded directory holding
// exactly one model GGUF and at most one projector GGUF, each sorted by
// what its own metadata declares. A projector named explicitly takes
// precedence over one found beside the model.
func ModelFiles(path, explicitProjector string) (model, projectorPath string, err error) {
	path = strings.TrimSpace(path)
	info, err := os.Stat(path)
	if err != nil {
		return "", "", err
	}
	candidates := []string{path}
	if info.IsDir() {
		if candidates, err = filepath.Glob(filepath.Join(path, "*.gguf")); err != nil {
			return "", "", err
		}
	}
	var projectors []string
	var models []string
	for _, candidate := range candidates {
		describes, err := projector.DescribesProjection(candidate)
		if err != nil {
			return "", "", fmt.Errorf("library: %s is not a GGUF the store can read: %w", candidate, err)
		}
		if describes {
			projectors = append(projectors, candidate)
		} else {
			models = append(models, candidate)
		}
	}
	if len(models) != 1 {
		return "", "", fmt.Errorf("library: %s holds %d model GGUF files; name the file to register", path, len(models))
	}
	if len(projectors) > 1 {
		return "", "", fmt.Errorf("library: %s holds %d projector GGUF files; name the projector to register", path, len(projectors))
	}
	if explicitProjector = strings.TrimSpace(explicitProjector); explicitProjector != "" {
		if describes, err := projector.DescribesProjection(explicitProjector); err != nil || !describes {
			return "", "", fmt.Errorf("library: %s does not declare a projector", explicitProjector)
		}
		return models[0], explicitProjector, nil
	}
	if len(projectors) == 1 {
		return models[0], projectors[0], nil
	}
	return models[0], "", nil
}

// Register publishes the model's facts and candidate recipe (known to the
// store, not yet servable) and a projector with it as the model's
// projection candidate; the receipt names what the store now holds.
func Register(ctx context.Context, repository *overgodb.Store, path, projectorPath string) (map[string]any, error) {
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, repository, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		return nil, refusal{err}
	}
	if err := modelintake.RegisterCandidate(context.WithoutCancel(ctx), repository, candidate); err != nil {
		return nil, err
	}
	receipt := map[string]any{
		"path": path, "model": candidate.Inventory.Manifest.ID,
		"definition": candidate.Resolved.Document.ID, "recipe": candidate.Definition.ID,
	}
	// A projector beside the model registers as the model's projection
	// candidate: the model's image, audio or video input, servable once
	// validated with the model.
	if projectorPath != "" {
		projection, err := modelintake.PrepareProjectionCandidate(ctx, path, projectorPath)
		if err != nil {
			return nil, refusal{err}
		}
		if err := modelintake.RegisterProjectionCandidate(context.WithoutCancel(ctx), repository, projection); err != nil {
			return nil, err
		}
		receipt["projector"], receipt["projection"], receipt["media"] = projectorPath, projection.Definition.ID, projection.Media
	}
	return receipt, nil
}

// Validate prepares text validation of a model under
// the clean source revision and returns the candidate recipe it runs under
// with the operation body that registers, records, replays, publishes and
// activates; the prompts and the token bound are the caller's policy.
func Validate(ctx context.Context, repository *overgodb.Store, path, projectorPath string, prompts []string, maxTokens int) (artifact.ID, operation.Executor, error) {
	if strings.TrimSpace(projectorPath) != "" {
		return artifact.ID{}, nil, refusal{errors.New("library: projector validation requires an executed media suite; register the pair, then use recipe -verify -task projection with an exact suite covering every declared media mode")}
	}
	revision, err := modelintake.CleanRevision(ctx)
	if err != nil {
		return artifact.ID{}, nil, refusal{err}
	}
	candidate, err := modelintake.PrepareInferenceCandidate(ctx, repository, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		return artifact.ID{}, nil, refusal{err}
	}
	execute := func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
		return validate(ctx, repository, reporter, path, candidate, revision, prompts, maxTokens)
	}
	return candidate.Definition.ID, execute, nil
}

// validate is the validation operation: register, record, replay, publish,
// activate, each a progress step the strip shows.
func validate(ctx context.Context, repository *overgodb.Store, reporter operation.Reporter, path string, candidate modelintake.Candidate, revision string, prompts []string, maxTokens int) (operation.Completion, error) {
	stages := []string{"register", "record", "replay", "publish", "activate"}
	total := uint64(len(stages))
	progress := func(stage string) { reporter.Progress(uint64(slices.Index(stages, stage)), &total) }
	progress("register")
	if err := modelintake.RegisterCandidate(ctx, repository, candidate); err != nil {
		return operation.Completion{}, err
	}
	loaded, err := modelrecipe.ResolveCandidateGGUF(path, candidate.Definition, candidate.Resolved)
	if err != nil {
		return operation.Completion{}, err
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		return operation.Completion{}, err
	}
	defer runner.Close()
	progress("record")
	started := time.Now()
	suite, err := modelintake.RecordExactSuite(ctx, runner, "library/"+filepath.Base(path), prompts, maxTokens)
	if err != nil {
		return operation.Completion{}, err
	}
	exactPlan, err := evaluation.CompileExact(suite)
	if err != nil {
		return operation.Completion{}, err
	}
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		return operation.Completion{}, err
	}
	evaluationPlan, err := evaluation.BindExact(exactPlan, evaluation.ExactAuthorities{
		ModelDefinition: candidate.Resolved.Document.ID, RuntimeRecipe: candidate.Definition.ID,
		CodeCommit: revision, Environment: environment.ID,
		Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
	})
	if err != nil {
		return operation.Completion{}, err
	}
	progress("replay")
	reportID, replayErr := evaluation.EvaluateExactSharded(ctx, repository, runner, exactPlan, evaluationPlan, nil)
	progress("publish")
	if replayErr != nil {
		evidence := "contract=exact; plan=" + evaluationPlan.Identity().String() + "; " + strings.ReplaceAll(replayErr.Error(), "\n", " ")
		verification, publishErr := modelintake.PublishFailure(ctx, repository, candidate.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence, "exact-mismatch")
		if publishErr != nil {
			return operation.Completion{}, errors.Join(replayErr, publishErr)
		}
		return operation.Completion{Run: verification.Run, Outputs: []artifact.ID{verification.Gate}}, replayErr
	}
	evidence := fmt.Sprintf("contract=exact;plan=%s;report=%s;cases=%d", evaluationPlan.Identity(), reportID, len(suite.Cases))
	verification, err := modelintake.PublishVerification(ctx, repository, candidate.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence)
	if err != nil {
		return operation.Completion{}, err
	}
	progress("activate")
	if err := modelrecipe.ActivateCapability(ctx, repository, candidate.Definition, verification, recipe.EvidenceVerified,
		"validated from the library: recorded and replayed an exact suite"); err != nil {
		return operation.Completion{}, err
	}
	outputs := []artifact.ID{verification.Gate, reportID}
	reporter.Publishing()
	reporter.Progress(total, &total)
	return operation.Completion{Run: verification.Run, Outputs: outputs}, nil
}
