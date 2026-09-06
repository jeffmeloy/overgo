package server

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
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
//
// The intake itself (internal/libraryintake) arrives from the assembly
// root as LibraryIntake, so this package names no intake package.

// LibraryIntake is the model intake the library routes drive, supplied by
// the assembly root: ModelFiles resolves the model and projector a request
// names, Register publishes the model's facts, candidate recipe and any
// projector and returns the receipt the route answers, and Validate
// prepares a validation under the clean revision and returns the recipe it
// runs under with the operation body the route submits. An error whose
// LibraryRefused method reports true is the request's own fault.
type LibraryIntake struct {
	ModelFiles func(path, explicitProjector string) (model, projectorPath string, err error)
	Register   func(ctx context.Context, repository *overgodb.Store, path, projectorPath string) (map[string]any, error)
	Validate   func(ctx context.Context, repository *overgodb.Store, path, projectorPath string, prompts []string, maxTokens int) (artifact.ID, operation.Executor, error)
	// DeclareProvider declares a hosted provider's models in the store as
	// the provider command does; absent, the provider kind answers that it
	// needs it.
	DeclareProvider func(ctx context.Context, repository *overgodb.Store, declaration ProviderDeclaration) ([]DeclaredProvider, error)
}

// ProviderDeclaration is a hosted provider and the model ids it serves,
// as the library route takes it.
type ProviderDeclaration struct {
	Name           string
	Endpoint       string
	KeyEnvironment string
	Models         []string
	ContextLength  uint32
}

// DeclaredProvider is one declared hosted model the route answers: its
// location, model and recipe identities, and the refusal its key's
// absence carries now.
type DeclaredProvider struct {
	Location string `json:"location"`
	Model    string `json:"model"`
	Recipe   string `json:"recipe"`
	Refusal  string `json:"refusal,omitzero"`
}

func (intake LibraryIntake) assembled() bool {
	return intake.ModelFiles != nil && intake.Register != nil && intake.Validate != nil
}

// libraryValidationPolicyPath names the root document with the prompts a
// validation records and replays.
const libraryValidationPolicyPath = "library_validation.json"

type libraryValidationPolicy struct {
	Schema    int      `json:"schema"`
	Doc       string   `json:"doc,omitzero"`
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
	// A provider declaration: the endpoint, the key's variable, the
	// model ids and the declared context length.
	Endpoint       string   `json:"endpoint,omitzero"`
	KeyEnvironment string   `json:"key_environment,omitzero"`
	Models         []string `json:"models,omitzero"`
	ContextLength  uint32   `json:"context_length,omitzero"`
}

type libraryValidateRequest struct {
	Path      string `json:"path"`
	Projector string `json:"projector,omitzero"`
}

// writeLibraryError answers an intake error: the request's own refusal as
// such, anything else as the generation failure it is.
func writeLibraryError(response http.ResponseWriter, err error) {
	if refused, ok := errors.AsType[interface {
		error
		LibraryRefused() bool
	}](err); ok && refused.LibraryRefused() {
		writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
		return
	}
	writeGenerationError(response, err)
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
		intake := h.config.LibraryIntake
		if !intake.assembled() {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "library registration needs the model intake")
			return
		}
		path, projectorPath, err := intake.ModelFiles(body.Path, body.Projector)
		if err != nil {
			writeInvalidRequest(response, err)
			return
		}
		receipt, err := intake.Register(request.Context(), h.repository, path, projectorPath)
		if err != nil {
			writeLibraryError(response, err)
			return
		}
		receipt["kind"] = body.Kind
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
	case "provider":
		declare := h.config.LibraryIntake.DeclareProvider
		if declare == nil {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "provider declaration needs the launcher's provider intake")
			return
		}
		declared, err := declare(context.WithoutCancel(request.Context()), h.repository, ProviderDeclaration{
			Name: body.Name, Endpoint: body.Endpoint, KeyEnvironment: body.KeyEnvironment, Models: body.Models, ContextLength: body.ContextLength,
		})
		if err != nil {
			writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
			return
		}
		writeJSON(response, http.StatusOK, map[string]any{"kind": body.Kind, "name": body.Name, "declared": declared})
	default:
		writeInvalidRequest(response, errors.New("library: kind must be model, dataset or provider"))
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
	intake := h.config.LibraryIntake
	if h.repository == nil || h.operations == nil || !intake.assembled() {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "library validation needs a store, an operation runtime and the model intake")
		return
	}
	var policy libraryValidationPolicy
	if err := loadPolicyDocument(libraryValidationPolicyPath, &policy); err != nil || len(policy.Prompts) == 0 || policy.MaxTokens <= 0 {
		writeError(response, http.StatusServiceUnavailable, "library_unavailable", "the library validation policy is absent or incomplete")
		return
	}
	path, projectorPath, err := intake.ModelFiles(body.Path, body.Projector)
	if err != nil {
		writeInvalidRequest(response, err)
		return
	}
	recipeID, execute, err := intake.Validate(request.Context(), h.repository, path, projectorPath, policy.Prompts, policy.MaxTokens)
	if err != nil {
		writeLibraryError(response, err)
		return
	}
	id, err := h.operations.Submit(context.WithoutCancel(request.Context()),
		operation.Request{Task: recipe.TaskInference, Recipe: recipeID}, execute)
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	writeJSON(response, http.StatusAccepted, map[string]any{
		"operation": id, "path": path, "projector": projectorPath, "recipe": recipeID, "prompts": len(policy.Prompts),
	})
}
