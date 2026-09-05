package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/operation"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// The library lifecycle (professional GUI campaign, gui-library): a model
// or dataset the hub client downloaded is registered and validated from the
// same page. Registering a model publishes its facts and candidate recipe
// (known to the store, not yet servable); validating it runs as an
// operation that records an exact suite from the model's own greedy
// output, replays it, publishes the verification evidence and activates
// the recipe with that evidence, so the catalog then serves it as
// verified. Registering a dataset commits its directory under a name;
// validating one is the preview the datasets route already answers. The
// prompts and the token bound the validation records come from the root
// policy document, never from a list typed into this file.
//
// A projector beside the model, or one named with it, registers as the
// model's projection candidate and validates with the model
// (gui-register-projectors): the projector opens on the host and declares
// its media, that verification activates the projection recipe, and the
// served model then accepts image, audio or video input through the
// projector the server resolves from the store.

// libraryValidationPolicyPath names the root document with the prompts a
// validation records and replays.
const libraryValidationPolicyPath = "library_validation.json"

type libraryValidationPolicy struct {
	Schema    int      `json:"schema"`
	Prompts   []string `json:"prompts"`
	MaxTokens int      `json:"max_tokens"`
}

type libraryRegisterRequest struct {
	Kind      string `json:"kind"`
	Path      string `json:"path,omitzero"`
	Projector string `json:"projector,omitzero"`
	Directory string `json:"directory,omitzero"`
	Name      string `json:"name,omitzero"`
	Modality  string `json:"modality,omitzero"`
}

type libraryValidateRequest struct {
	Path      string `json:"path"`
	Projector string `json:"projector,omitzero"`
}

// libraryModelFiles resolves the model file a register or validate names
// and the projector beside it: a GGUF file, or a downloaded directory
// holding exactly one model GGUF and at most one projector GGUF, each
// sorted by what its own metadata declares. A projector named explicitly
// takes precedence over one found beside the model.
func libraryModelFiles(path, explicitProjector string) (model, projectorPath string, err error) {
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

// libraryRegister answers POST /library/register for a model or a dataset.
func (h *Handler) libraryRegister(response http.ResponseWriter, request *http.Request) {
	var body libraryRegisterRequest
	if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if h.repository == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "library registration needs a store")
		return
	}
	switch body.Kind {
	case "model":
		path, projectorPath, err := libraryModelFiles(body.Path, body.Projector)
		if err != nil {
			writeInvalidRequest(response, err)
			return
		}
		candidate, err := modelintake.PrepareInferenceCandidate(request.Context(), h.repository, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
		if err != nil {
			writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
			return
		}
		if err := modelintake.RegisterCandidate(context.WithoutCancel(request.Context()), h.repository, candidate); err != nil {
			writeGenerationError(response, err)
			return
		}
		receipt := map[string]any{
			"kind": body.Kind, "path": path, "model": candidate.Inventory.Manifest.ID,
			"definition": candidate.Resolved.Document.ID, "recipe": candidate.Definition.ID,
		}
		// A projector beside the model registers as the model's projection
		// candidate: the model's image, audio or video input, servable once
		// validated with the model.
		if projectorPath != "" {
			projection, err := modelintake.PrepareProjectionCandidate(request.Context(), path, projectorPath)
			if err != nil {
				writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
				return
			}
			if err := modelintake.RegisterProjectionCandidate(context.WithoutCancel(request.Context()), h.repository, projection); err != nil {
				writeGenerationError(response, err)
				return
			}
			receipt["projector"], receipt["projection"], receipt["media"] = projectorPath, projection.Definition.ID, projection.Media
		}
		writeJSON(response, http.StatusOK, receipt)
	case "dataset":
		registered, err := dataset.RegisterDirectoryDatasetAs(context.WithoutCancel(request.Context()), h.repository, strings.TrimSpace(body.Name), strings.TrimSpace(body.Directory), body.Modality)
		if err != nil {
			writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
			return
		}
		writeJSON(response, http.StatusOK, map[string]any{
			"kind": body.Kind, "name": body.Name, "dataset": registered.Dataset, "commit": registered.Commit,
			"files": registered.Files, "bytes": registered.Bytes, "changed": registered.Changed,
		})
	default:
		writeInvalidRequest(response, errors.New("library: kind must be model or dataset"))
	}
}

// libraryValidate answers POST /library/validate: the validation runs as an
// operation the strip shows, and its receipt is the gate and run evidence
// the activation binds.
func (h *Handler) libraryValidate(response http.ResponseWriter, request *http.Request) {
	var body libraryValidateRequest
	if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	if h.repository == nil || h.operations == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "library validation needs a store and an operation runtime")
		return
	}
	var policy libraryValidationPolicy
	if err := loadPolicyDocument(libraryValidationPolicyPath, &policy); err != nil || len(policy.Prompts) == 0 || policy.MaxTokens <= 0 {
		writeError(response, http.StatusServiceUnavailable, "library_unavailable", "the library validation policy is absent or incomplete")
		return
	}
	path, projectorPath, err := libraryModelFiles(body.Path, body.Projector)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	revision, err := modelintake.CleanRevision(request.Context())
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
		return
	}
	candidate, err := modelintake.PrepareInferenceCandidate(request.Context(), h.repository, path, modelintake.SessionOverride{}, recipe.ResidencyHybridNative)
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
		return
	}
	var projection *modelintake.ProjectionCandidate
	if projectorPath != "" {
		prepared, err := modelintake.PrepareProjectionCandidate(request.Context(), path, projectorPath)
		if err != nil {
			writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
			return
		}
		projection = &prepared
	}
	id, err := h.operations.Submit(context.WithoutCancel(request.Context()),
		operation.Request{Task: recipe.TaskInference, Recipe: candidate.Definition.ID},
		func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
			return h.validateLibraryModel(ctx, reporter, path, candidate, projection, revision, policy)
		})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusAccepted, map[string]any{
		"operation": id, "path": path, "projector": projectorPath, "recipe": candidate.Definition.ID, "prompts": len(policy.Prompts),
	})
}

// validateLibraryModel is the validation operation: register, record,
// replay, publish, activate, and with a projector the projection's own
// verification and activation, each a progress step the strip shows.
func (h *Handler) validateLibraryModel(ctx context.Context, reporter operation.Reporter, path string, candidate modelintake.Candidate, projection *modelintake.ProjectionCandidate, revision string, policy libraryValidationPolicy) (operation.Completion, error) {
	stages := []string{"register", "record", "replay", "publish", "activate"}
	if projection != nil {
		stages = append(stages, "projector")
	}
	total := uint64(len(stages))
	progress := func(stage string) { reporter.Progress(uint64(slices.Index(stages, stage)), &total) }
	progress("register")
	if err := modelintake.RegisterCandidate(ctx, h.repository, candidate); err != nil {
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
	suite, err := modelintake.RecordExactSuite(ctx, runner, "library/"+filepath.Base(path), policy.Prompts, policy.MaxTokens)
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
	reportID, replayErr := evaluation.EvaluateExactSharded(ctx, h.repository, runner, exactPlan, evaluationPlan, nil)
	progress("publish")
	if replayErr != nil {
		evidence := "contract=exact; plan=" + evaluationPlan.Identity().String() + "; " + strings.ReplaceAll(replayErr.Error(), "\n", " ")
		verification, publishErr := modelintake.PublishFailure(ctx, h.repository, candidate.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence, "exact-mismatch")
		if publishErr != nil {
			return operation.Completion{}, errors.Join(replayErr, publishErr)
		}
		return operation.Completion{Run: verification.Run, Outputs: []artifact.ID{verification.Gate}}, replayErr
	}
	evidence := fmt.Sprintf("contract=exact;plan=%s;report=%s;cases=%d", evaluationPlan.Identity(), reportID, len(suite.Cases))
	verification, err := modelintake.PublishVerification(ctx, h.repository, candidate.Definition, revision, time.Since(started), "cuda:0", "cuda", evidence)
	if err != nil {
		return operation.Completion{}, err
	}
	progress("activate")
	if err := modelrecipe.ActivateCapability(ctx, h.repository, candidate.Definition, verification, recipe.EvidenceVerified,
		"validated from the library: recorded and replayed an exact suite"); err != nil {
		return operation.Completion{}, err
	}
	outputs := []artifact.ID{verification.Gate, reportID}
	if projection != nil {
		progress("projector")
		if err := modelintake.RegisterProjectionCandidate(ctx, h.repository, *projection); err != nil {
			return operation.Completion{}, err
		}
		projected, err := modelintake.VerifyProjection(ctx, h.repository, *projection, revision)
		if err != nil {
			return operation.Completion{}, err
		}
		if err := modelintake.ActivateProjection(ctx, h.repository, *projection, projected,
			"validated from the library: the projector opened and declared its media"); err != nil {
			return operation.Completion{}, err
		}
		outputs = append(outputs, projected.Gate)
	}
	reporter.Publishing()
	reporter.Progress(total, &total)
	return operation.Completion{Run: verification.Run, Outputs: outputs}, nil
}
