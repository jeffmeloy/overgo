package server

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"overgo/internal/apimanifest"
	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
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
	// ListProviderModels asks a hosted provider for the models it serves,
	// under the key its variable holds; absent, the listing route answers
	// that it needs it.
	ListProviderModels func(ctx context.Context, endpoint, keyEnvironment string) ([]ProviderModel, error)
	// RetireProvider retires a declared hosted model by its location with
	// a reason, as the provider command does; absent, the retirement route
	// answers that it needs it.
	RetireProvider func(ctx context.Context, repository *overgodb.Store, location, reason string) error
}

// providerRetireRequest: the hosted model to retire and why.
type providerRetireRequest struct {
	Location string `json:"location"`
	Reason   string `json:"reason"`
}

// ProviderModel is one model a hosted provider lists: its id, a display
// name when listed, and the context length it declares (zero unstated).
type ProviderModel struct {
	ID            string `json:"id"`
	Name          string `json:"name,omitzero"`
	ContextLength uint32 `json:"context_length,omitzero"`
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
	libraryRegisterRoute(response, request, h.repository, h.config.LibraryIntake, body)
}

func requireLibraryStore(response http.ResponseWriter, request *http.Request, store *overgodb.Store) bool {
	if store == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "library writes need the store")
		return false
	}
	if err := store.Refresh(request.Context()); err != nil {
		if errors.Is(err, processlock.ErrBusy) {
			writeError(response, http.StatusServiceUnavailable, "store_busy", "another database writer holds access; retry when it releases access")
		} else {
			writeError(response, http.StatusServiceUnavailable, "store_unavailable", err.Error())
		}
		return false
	}
	return true
}

// libraryRegisterRoute shares registration and store admission across served and idle workbenches.
func libraryRegisterRoute(response http.ResponseWriter, request *http.Request, repository *overgodb.Store, intake LibraryIntake, body libraryRegisterRequest) {
	if !requireLibraryStore(response, request, repository) {
		return
	}
	switch body.Kind {
	case "model":
		if intake.ModelFiles == nil || intake.Register == nil {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "library registration needs the model intake")
			return
		}
		path, projectorPath, err := intake.ModelFiles(body.Path, body.Projector)
		if err != nil {
			writeInvalidRequest(response, err)
			return
		}
		receipt, err := intake.Register(request.Context(), repository, path, projectorPath)
		if err != nil {
			writeLibraryError(response, err)
			return
		}
		receipt["kind"] = body.Kind
		writeJSON(response, http.StatusOK, receipt)
	case "dataset":
		registered, err := dataset.RegisterDirectoryDatasetAs(request.Context(), repository, strings.TrimSpace(body.Name), strings.TrimSpace(body.Directory), body.Modality)
		if err != nil {
			writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
			return
		}
		writeJSON(response, http.StatusOK, map[string]any{
			"kind": body.Kind, "name": body.Name, "dataset": registered.Dataset, "commit": registered.Commit,
			"files": registered.Files, "bytes": registered.Bytes, "changed": registered.Changed,
		})
	case "provider":
		declare := intake.DeclareProvider
		if declare == nil {
			writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "provider declaration needs the launcher's provider intake")
			return
		}
		declared, err := declare(request.Context(), repository, ProviderDeclaration{
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

// libraryProviderModels answers GET /library/providers/models: the
// provider named by endpoint and key variable lists its models through
// the launcher's intake, so the page declares from the provider's own
// listing with each model's declared context length.
func (h *Handler) libraryProviderModels(response http.ResponseWriter, request *http.Request) {
	libraryProviderModelsRoute(response, request, h.config.LibraryIntake.ListProviderModels)
}

// libraryProviderModelsRoute: the listing over one intake; the idle shell
// serves it too. The listing sends the named variable as a bearer token to
// the named endpoint, so the GET is admitted as a mutation.
func libraryProviderModelsRoute(response http.ResponseWriter, request *http.Request, list func(context.Context, string, string) ([]ProviderModel, error)) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	if refusal := apimanifest.AdmitCredentiallessEffect(request); refusal != nil {
		writeError(response, http.StatusForbidden, refusal.Type, refusal.Message)
		return
	}
	endpoint, variable := strings.TrimSpace(request.URL.Query().Get("endpoint")), strings.TrimSpace(request.URL.Query().Get("key_environment"))
	if endpoint == "" || variable == "" {
		writeInvalidRequestMessage(response, "library: endpoint and key_environment are required")
		return
	}
	if list == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "provider listing needs the launcher's provider intake")
		return
	}
	models, err := list(request.Context(), endpoint, variable)
	if err != nil {
		writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"endpoint": endpoint, "models": models})
}

// libraryProviderRetire answers POST /library/providers/retire: the
// declared hosted model at the location retires through the launcher's
// intake with the reason (a failed gate record, the alias released), so
// the catalog and the picker stop offering it.
func (h *Handler) libraryProviderRetire(response http.ResponseWriter, request *http.Request) {
	var body providerRetireRequest
	if !requireMethod(response, request, http.MethodPost) || !h.decodeBoundedJSON(response, request, &body) {
		return
	}
	libraryProviderRetireRoute(response, request, h.repository, h.config.LibraryIntake.RetireProvider, body)
}

// libraryProviderRetireRoute: the retirement over one store and intake; the idle shell serves it too.
func libraryProviderRetireRoute(response http.ResponseWriter, request *http.Request, repository *overgodb.Store, retire func(context.Context, *overgodb.Store, string, string) error, body providerRetireRequest) {
	if strings.TrimSpace(body.Location) == "" || strings.TrimSpace(body.Reason) == "" {
		writeInvalidRequestMessage(response, "library: location and reason are required")
		return
	}
	if repository == nil || retire == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "provider retirement needs a store and the launcher's provider intake")
		return
	}
	if !requireLibraryStore(response, request, repository) {
		return
	}
	if err := retire(request.Context(), repository, body.Location, body.Reason); err != nil {
		writeError(response, http.StatusUnprocessableEntity, "library_refused", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"location": body.Location, "retired": true})
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
	if !requireLibraryStore(response, request, h.repository) {
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
