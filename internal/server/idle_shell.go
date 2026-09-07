package server

import (
	"context"
	"net/http"

	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
)

// IdleShell answers the client while the swap proxy runs no child: the
// static shell, health naming no model, the workspace manifest with the
// Library tab enabled and every other tab refused, the store's servable
// catalog so the picker can choose the first model, and the library's
// registration, declaration, listing and retirement routes over the
// store it opens for the write (no child holds the lock); every other
// route refuses with the reason.
type IdleShell struct {
	// Catalog lists the store's activations (the proxy's resolver reads the store read-only).
	Catalog func(ctx context.Context) ([]discovery.CatalogEntry, bool, error)
	// OpenStore opens the store for a library write; closed when the route answers.
	OpenStore func(ctx context.Context) (*overgodb.Store, error)
	// Intake is the library intake the writes go through (validation, which needs a served model, stays absent).
	Intake LibraryIntake
	// WebUIDir serves the client from disk when set (development), as the server's own option does.
	WebUIDir string
}

// idleRefusal: the reason every refused route and every refused manifest tab carries while nothing serves.
const idleRefusal = "no model serves; choose one from the model pill"

// idleTabs: the manifest tabs the idle shell serves; the Library's store routes need no served model.
var idleTabs = map[string]bool{"library": true}

// ServeHTTP implements http.Handler.
func (s *IdleShell) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	switch request.URL.Path {
	case "/health":
		writeJSON(response, http.StatusOK, map[string]any{"status": "ok", "model": ""})
	case "/workspace/manifest":
		s.manifest(response)
	case "/catalog/models":
		s.catalog(response, request)
	case "/providers/key":
		// The proxy took the key before delegating; the next child inherits it.
		writeJSON(response, http.StatusOK, map[string]any{"held": true})
	case "/library/register":
		var body libraryRegisterRequest
		if !requireMethod(response, request, http.MethodPost) || !decodeJSONBounded(response, request, &body, maxRequestBytes) {
			return
		}
		s.withStore(response, request, func(store *overgodb.Store) { libraryRegisterRoute(response, request, store, s.Intake, body) })
	case "/library/providers/models":
		libraryProviderModelsRoute(response, request, s.Intake.ListProviderModels)
	case "/library/providers/retire":
		var body providerRetireRequest
		if !requireMethod(response, request, http.MethodPost) || !decodeJSONBounded(response, request, &body, maxRequestBytes) {
			return
		}
		s.withStore(response, request, func(store *overgodb.Store) {
			libraryProviderRetireRoute(response, request, store, s.Intake.RetireProvider, body)
		})
	default:
		if _, api := resolveRoute(request.URL.Path); api {
			writeError(response, http.StatusServiceUnavailable, "no_model_serves", idleRefusal)
			return
		}
		serveWebUIAssets(response, request, s.WebUIDir)
	}
}

// withStore runs one library write over the store opened for it.
func (s *IdleShell) withStore(response http.ResponseWriter, request *http.Request, route func(store *overgodb.Store)) {
	if s.OpenStore == nil {
		writeError(response, http.StatusNotImplemented, errorCodeUnsupportedOperation, "library writes need the store")
		return
	}
	store, err := s.OpenStore(request.Context())
	if err != nil {
		writeError(response, http.StatusServiceUnavailable, "store_unavailable", err.Error())
		return
	}
	defer store.Close()
	route(store)
}

func (s *IdleShell) manifest(response http.ResponseWriter) {
	declaration, err := parseWorkspaceManifest()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	result := workspaceManifestResponse{Version: declaration.Version, Sections: declaration.Sections, Tabs: make([]workspaceTab, len(declaration.Tabs))}
	for index, tab := range declaration.Tabs {
		served := idleTabs[tab.ID]
		result.Tabs[index] = workspaceTab{ID: tab.ID, Label: tab.Label, Section: tab.Section, Module: tab.module(), Enabled: served}
		if !served {
			result.Tabs[index].Refusal = idleRefusal
		}
	}
	writeJSON(response, http.StatusOK, result)
}

func (s *IdleShell) catalog(response http.ResponseWriter, request *http.Request) {
	if s.Catalog == nil {
		writeError(response, http.StatusServiceUnavailable, "hub_unavailable", "no store catalog is configured")
		return
	}
	entries, truncated, err := s.Catalog(request.Context())
	if err != nil {
		writeError(response, http.StatusInternalServerError, "catalog_error", err.Error())
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"models": catalogListing(entries, evaluation.EvidenceIndex{}), "truncated": truncated, "coverage": map[string]any{}})
}
