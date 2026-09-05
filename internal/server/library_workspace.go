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
	Directory string `json:"directory,omitzero"`
	Name      string `json:"name,omitzero"`
	Modality  string `json:"modality,omitzero"`
}

type libraryValidateRequest struct {
	Path string `json:"path"`
}

// libraryModelPath resolves the model file a register or validate names:
// a GGUF file, or a downloaded directory holding exactly one.
func libraryModelPath(path string) (string, error) {
	path = strings.TrimSpace(path)
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return path, nil
	}
	matches, err := filepath.Glob(filepath.Join(path, "*.gguf"))
	if err != nil {
		return "", err
	}
	if len(matches) != 1 {
		return "", fmt.Errorf("library: %s holds %d GGUF files; name the file to register", path, len(matches))
	}
	return matches[0], nil
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
		path, err := libraryModelPath(body.Path)
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
		writeJSON(response, http.StatusOK, map[string]any{
			"kind": body.Kind, "path": path, "model": candidate.Inventory.Manifest.ID,
			"definition": candidate.Resolved.Document.ID, "recipe": candidate.Definition.ID,
		})
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
	path, err := libraryModelPath(body.Path)
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
	id, err := h.operations.Submit(context.WithoutCancel(request.Context()),
		operation.Request{Task: recipe.TaskInference, Recipe: candidate.Definition.ID},
		func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
			return h.validateLibraryModel(ctx, reporter, path, candidate, revision, policy)
		})
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusAccepted, map[string]any{
		"operation": id, "path": path, "recipe": candidate.Definition.ID, "prompts": len(policy.Prompts),
	})
}

// validateLibraryModel is the validation operation: register, record,
// replay, publish, activate, each a progress step the strip shows.
func (h *Handler) validateLibraryModel(ctx context.Context, reporter operation.Reporter, path string, candidate modelintake.Candidate, revision string, policy libraryValidationPolicy) (operation.Completion, error) {
	stages := []string{"register", "record", "replay", "publish", "activate"}
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
	reporter.Publishing()
	reporter.Progress(total, &total)
	return operation.Completion{Run: verification.Run, Outputs: []artifact.ID{verification.Gate, reportID}}, nil
}
