package modelswap

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
)

// maxRoutedBodyBytes bounds the request body the router buffers to
// read its model field; matches the serving multimodal request bound.
const maxRoutedBodyBytes = 32 << 20

// Resolver maps a requested model name to a launchable servable.
type Resolver interface {
	Resolve(ctx context.Context, name string) (Servable, bool, error)
}

// Proxy routes every request to the child serving the requested
// model: the model field of a JSON body (or the model query
// parameter) picks the servable, the supervisor ensures that child is
// the one running, and the request streams through untouched. A
// request naming no model rides the currently running child, so the
// whole workbench works through the proxy unchanged.
type Proxy struct {
	Supervisor *Supervisor
	Resolver   Resolver
	// Default serves model-less requests when nothing is running yet.
	Default Servable
}

// ServeHTTP implements the swap routing.
func (p *Proxy) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if p.Supervisor == nil || p.Resolver == nil {
		http.Error(response, "model swap proxy is not configured", http.StatusServiceUnavailable)
		return
	}
	servable, body, err := p.routeServable(request)
	if err != nil {
		http.Error(response, err.Error(), http.StatusUnprocessableEntity)
		return
	}
	target, release, err := p.Supervisor.Acquire(request.Context(), servable)
	if err != nil {
		http.Error(response, err.Error(), http.StatusServiceUnavailable)
		return
	}
	defer release()
	upstream, err := url.Parse(target)
	if err != nil {
		http.Error(response, err.Error(), http.StatusInternalServerError)
		return
	}
	if body != nil {
		request.Body = io.NopCloser(bytes.NewReader(body))
		request.ContentLength = int64(len(body))
	}
	proxy := httputil.NewSingleHostReverseProxy(upstream)
	proxy.FlushInterval = -1
	proxy.ServeHTTP(response, request)
}

// routeServable picks the servable a request names: the swap query
// parameter (dedicated to the proxy -- the server's own endpoints
// already use ?model= to name analysis targets, which must never
// trigger a swap), then the JSON body's model field, then whatever
// child is already running, then the default. The body is buffered
// and handed back for replay to the child.
func (p *Proxy) routeServable(request *http.Request) (Servable, []byte, error) {
	var body []byte
	name := request.URL.Query().Get("swap")
	if name == "" && request.Body != nil && request.ContentLength != 0 &&
		strings.Contains(request.Header.Get("Content-Type"), "json") {
		buffered, err := io.ReadAll(io.LimitReader(request.Body, maxRoutedBodyBytes))
		if err != nil {
			return Servable{}, nil, err
		}
		body = buffered
		var envelope struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(buffered, &envelope)
		name = envelope.Model
	}
	if name != "" {
		// A request naming the running child's own identity is already
		// served -- chat requests echo the served model id, and resolving
		// it against the catalog could alias to a different artifact and
		// swap spuriously mid-conversation.
		if running, ok := p.Supervisor.Status(); ok &&
			(strings.EqualFold(name, running.Name) || strings.EqualFold(name, running.Model)) {
			return running, body, nil
		}
		servable, found, err := p.Resolver.Resolve(request.Context(), name)
		if err != nil {
			return Servable{}, nil, err
		}
		if found {
			return servable, body, nil
		}
		// An unresolvable model name falls through to the running child:
		// serving-side model identifiers (the child's own model id) route
		// to the child that owns them.
	}
	if running, ok := p.Supervisor.Status(); ok {
		return running, body, nil
	}
	if p.Default.Name != "" {
		return p.Default, body, nil
	}
	return Servable{}, nil, errors.New("model swap: no model named, none running, and no default configured")
}
