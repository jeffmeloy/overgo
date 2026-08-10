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
	response.WriteHeader(http.StatusOK)
	if request.Method == http.MethodHead {
		return
	}
	_, _ = response.Write(data)
}
