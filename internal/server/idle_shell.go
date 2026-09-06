package server

import (
	"context"
	"net/http"

	"overgo/internal/discovery"
	"overgo/internal/evaluation"
)

// IdleShell answers the client while the swap proxy runs no child: the
// static shell, health naming no model, the workspace manifest with every
// tab refused, and the store's servable catalog so the picker can choose
// the first model; every other route refuses with the reason.
type IdleShell struct {
	// Catalog lists the store's activations (the proxy's resolver reads the store read-only).
	Catalog func(ctx context.Context) ([]discovery.CatalogEntry, bool, error)
	// WebUIDir serves the client from disk when set (development), as the server's own option does.
	WebUIDir string
}

// idleRefusal: the reason every refused route and every manifest tab carries while nothing serves.
const idleRefusal = "no model serves; choose one from the model pill"

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
	default:
		if _, api := resolveRoute(request.URL.Path); api {
			writeError(response, http.StatusServiceUnavailable, "no_model_serves", idleRefusal)
			return
		}
		serveWebUIAssets(response, request, s.WebUIDir)
	}
}

func (s *IdleShell) manifest(response http.ResponseWriter) {
	declaration, err := parseWorkspaceManifest()
	if err != nil {
		writeGenerationError(response, err)
		return
	}
	result := workspaceManifestResponse{Version: declaration.Version, Sections: declaration.Sections, Tabs: make([]workspaceTab, len(declaration.Tabs))}
	for index, tab := range declaration.Tabs {
		result.Tabs[index] = workspaceTab{ID: tab.ID, Label: tab.Label, Section: tab.Section, Module: tab.module(), Refusal: idleRefusal}
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
