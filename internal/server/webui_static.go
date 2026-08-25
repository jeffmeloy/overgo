package server

import (
	"embed"
	"io/fs"
	"net/http"
	"path"
	"strings"
)

//go:embed webui
var webuiEmbed embed.FS

// webuiContentSecurityPolicy locks the self-contained client to same-origin
// scripts/connections with no inline scripts, no plugins, and no framing. Style
// keeps 'unsafe-inline' for the inline style attributes el()/SVG rely on.
const webuiContentSecurityPolicy = "default-src 'self'; " +
	"script-src 'self'; " +
	"style-src 'self' 'unsafe-inline'; " +
	"img-src 'self' data:; " +
	"connect-src 'self'; " +
	"font-src 'self'; " +
	"object-src 'none'; " +
	"base-uri 'self'; " +
	"frame-ancestors 'none'; " +
	"form-action 'self'"

// webuiFS: the embedded client rooted at the webui/ directory, so request path
// "/style.css" maps to "style.css".
var webuiFS = mustSubFS(webuiEmbed, "webui")

func mustSubFS(embedded embed.FS, dir string) fs.FS {
	sub, err := fs.Sub(embedded, dir)
	if err != nil {
		panic("server: webui embed sub: " + err.Error())
	}
	return sub
}

// webuiContentTypes: explicit content types for the asset extensions the GUI
// uses, so serving is deterministic regardless of the host's MIME registry.
var webuiContentTypes = map[string]string{
	".html": "text/html; charset=utf-8",
	".css":  "text/css; charset=utf-8",
	".js":   "text/javascript; charset=utf-8",
	".json": "application/json; charset=utf-8",
	".svg":  "image/svg+xml",
	".ico":  "image/x-icon",
	".map":  "application/json; charset=utf-8",
}

// serveWebUI serves the embedded GUI. It is reached only from the router's
// default case, so every API route (all exact-match) takes precedence and can
// never be shadowed. Assets are public — the shell loads without a bearer and
// authenticates its API calls separately. "/" resolves to index.html; the app
// uses hash routing, so no server-side SPA fallback is needed and an unknown
// path returns the same 404 the router used to emit.
func (h *Handler) serveWebUI(response http.ResponseWriter, request *http.Request) {
	if request.URL.Path == "/workspace/manifest" {
		h.workspaceManifest(response, request)
		return
	}
	if request.URL.Path == "/workspace/schema" {
		h.workspaceSchema(response, request)
		return
	}
	if request.URL.Path == "/operations/evidence" {
		if !h.authorized(request) {
			response.Header().Set("WWW-Authenticate", "Bearer")
			writeError(response, http.StatusUnauthorized, "invalid_api_key", "missing or invalid bearer token")
			return
		}
		h.operationEvidence(response, request)
		return
	}
	if strings.HasPrefix(request.URL.Path, "/automations") {
		if !h.authorized(request) {
			response.Header().Set("WWW-Authenticate", "Bearer")
			writeError(response, http.StatusUnauthorized, "invalid_api_key", "missing or invalid bearer token")
			return
		}
		h.automationWorkspace(response, request)
		return
	}
	if strings.HasPrefix(request.URL.Path, "/peers") {
		if !h.authorized(request) {
			response.Header().Set("WWW-Authenticate", "Bearer")
			writeError(response, http.StatusUnauthorized, "invalid_api_key", "missing or invalid bearer token")
			return
		}
		h.peerWorkspace(response, request)
		return
	}
	if request.Method != http.MethodGet && request.Method != http.MethodHead {
		writeError(response, http.StatusNotFound, "not_found", "route not found")
		return
	}
	name := strings.TrimPrefix(request.URL.Path, "/")
	if name == "" {
		name = "index.html"
	}
	name = path.Clean(name)
	// path.Clean on a rooted-then-trimmed name cannot escape, but reject any
	// traversal or absolute remnant defensively before touching the FS.
	if name == "." || strings.HasPrefix(name, "../") || strings.HasPrefix(name, "/") {
		writeError(response, http.StatusNotFound, "not_found", "route not found")
		return
	}
	data, err := fs.ReadFile(webuiFS, name)
	if err != nil {
		writeError(response, http.StatusNotFound, "not_found", "route not found")
		return
	}
	if contentType, ok := webuiContentTypes[strings.ToLower(path.Ext(name))]; ok {
		response.Header().Set("Content-Type", contentType)
	}
	response.Header().Set("Cache-Control", "no-cache")
	// The client is fully self-contained (no external hosts) and carries no inline
	// scripts — index.html's probe lives in probe.js and DOM handlers are attached
	// via addEventListener — so a strict CSP holds: script from same origin only,
	// no plugins, no framing. Inline STYLE attributes are used throughout (el()
	// and SVG), so style keeps 'unsafe-inline' (style injection is far lower risk
	// than script). This is defense in depth over the markdown renderer's own
	// DOM-only, scheme-checked output.
	response.Header().Set("Content-Security-Policy", webuiContentSecurityPolicy)
	response.Header().Set("X-Content-Type-Options", "nosniff")
	response.Header().Set("Referrer-Policy", "no-referrer")
	response.WriteHeader(http.StatusOK)
	if request.Method == http.MethodHead {
		return
	}
	_, _ = response.Write(data)
}
