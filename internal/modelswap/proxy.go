package modelswap

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"

	"overgo/internal/apimanifest"
	"overgo/internal/httpstream"
	"overgo/internal/processcontrol"
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
	// Keys takes a hosted provider's key for this process, so every child
	// launched from now on inherits it; the request then rides on to the
	// running child, which takes it for itself. Nil refuses the route.
	Keys ProviderKeys
	// Idle answers a model-less request while no child runs and no default
	// is set (the cold start: the shell, its boot routes, the catalog the
	// picker chooses from). Nil refuses such a request instead.
	Idle http.Handler
}

// errNothingServes: a model-less request with no child running and no default.
var errNothingServes = errors.New("model swap: no model named, none running, and no default configured")

// ProviderKeys places a hosted provider's key in the proxy's process for
// the model a reference names.
type ProviderKeys interface {
	SetKey(ctx context.Context, reference, key string) error
}

// providerKeyPath is the server's provider-key route, intercepted here
// so the proxy holds the key before the child does.
const providerKeyPath = "/providers/key"

// providerKeyRequest mirrors the server route's body.
type providerKeyRequest struct {
	Location string `json:"location"`
	Key      string `json:"key"`
}

// ServeHTTP implements the swap routing.
func (p *Proxy) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	if p.Supervisor == nil || p.Resolver == nil {
		http.Error(response, "model swap proxy is not configured", http.StatusServiceUnavailable)
		return
	}
	// The proxy carries no credential and mutates its own process (a key, a launch) before the child
	// sees the request, so it admits exactly as the credential-less server does, first.
	admit := apimanifest.AdmitCredentialless
	if request.URL.Query().Get("swap") != "" {
		// The swap query launches or replaces a child on a GET, so it is admitted as a mutation.
		admit = apimanifest.AdmitCredentiallessEffect
	}
	if refusal := admit(request); refusal != nil {
		response.Header().Set("Content-Type", "application/json")
		response.WriteHeader(http.StatusForbidden)
		_ = json.NewEncoder(response).Encode(map[string]any{"error": map[string]string{"message": refusal.Message, "type": refusal.Type}})
		return
	}
	mediaType, _, _ := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if mediaType == "application/x-ndjson" {
		release, err := httpstream.OpenDuplex(response, request)
		if err != nil {
			http.Error(response, "bidirectional proxy transport is unavailable", http.StatusNotImplemented)
			return
		}
		defer release()
	}
	if request.URL.Path == providerKeyPath && request.Method == http.MethodPost {
		if p.Keys == nil {
			http.Error(response, "model swap proxy takes no provider keys", http.StatusNotImplemented)
			return
		}
		raw, err := io.ReadAll(io.LimitReader(request.Body, maxRoutedBodyBytes))
		if err != nil {
			http.Error(response, err.Error(), http.StatusBadRequest)
			return
		}
		var entry providerKeyRequest
		if err := json.Unmarshal(raw, &entry); err != nil {
			http.Error(response, err.Error(), http.StatusBadRequest)
			return
		}
		if err := p.Keys.SetKey(request.Context(), entry.Location, entry.Key); err != nil {
			http.Error(response, err.Error(), http.StatusUnprocessableEntity)
			return
		}
		request.Body = io.NopCloser(bytes.NewReader(raw))
		request.ContentLength = int64(len(raw))
	}
	servable, err := p.routeServable(request)
	if errors.Is(err, errNothingServes) && p.Idle != nil {
		// The header names the proxy with no model, so the shell shows the proxy present and nothing served.
		response.Header().Set("X-Overgo-Swap-Proxy", "")
		p.Idle.ServeHTTP(response, request)
		return
	}
	if err != nil {
		http.Error(response, err.Error(), http.StatusUnprocessableEntity)
		return
	}
	target, release, err := p.Supervisor.Acquire(request.Context(), servable)
	if err != nil {
		code := "model_load_failed"
		if errors.Is(err, processcontrol.ErrResourceBusy) {
			code = "resource_busy"
		} else if errors.Is(err, processcontrol.ErrDeviceMemory) {
			code = "insufficient_memory"
		}
		response.Header().Set("Content-Type", "application/json")
		response.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(response).Encode(map[string]any{"error": map[string]string{"code": code, "message": err.Error()}})
		return
	}
	defer release()
	upstream, err := url.Parse(target)
	if err != nil {
		http.Error(response, err.Error(), http.StatusInternalServerError)
		return
	}
	// Every proxied answer names the proxy, so a shell behind it can show the
	// swap capability and a shell served directly can say it is absent.
	response.Header().Set("X-Overgo-Swap-Proxy", servable.Name)
	proxy := httputil.NewSingleHostReverseProxy(upstream)
	proxy.FlushInterval = -1
	proxy.ErrorHandler = writeProxyError
	proxy.ServeHTTP(response, request)
}

// A cancelled client no longer receives a gateway response. Preserve normal
// upstream failure diagnostics and status for requests that are still live.
func writeProxyError(response http.ResponseWriter, request *http.Request, err error) {
	if errors.Is(request.Context().Err(), context.Canceled) {
		return
	}
	log.Printf("http: proxy error: %v", err)
	http.Error(response, http.StatusText(http.StatusBadGateway), http.StatusBadGateway)
}

// routeServable picks the servable a request names: the swap query
// parameter (dedicated to the proxy -- the server's own endpoints
// already use ?model= to name analysis targets, which must never
// trigger a swap), then the JSON body's model field, then whatever
// child is already running, then the default. NDJSON routing reads only
// its opening record; the remaining stream is passed to the child unread.
func (p *Proxy) routeServable(request *http.Request) (Servable, error) {
	name := request.URL.Query().Get("swap")
	explicitSwap := name != ""
	if name == "" && request.Body != nil && request.ContentLength != 0 &&
		strings.Contains(request.Header.Get("Content-Type"), "json") {
		buffered, err := routingEnvelope(request)
		if err != nil {
			return Servable{}, err
		}
		var envelope struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(buffered, &envelope); err != nil {
			return Servable{}, fmt.Errorf("model swap: invalid routing envelope: %w", err)
		}
		name = envelope.Model
	}
	if name != "" {
		// A request naming the running child's own identity is already
		// served -- chat requests echo the served model id, and resolving
		// it against the catalog could alias to a different artifact and
		// swap spuriously mid-conversation.
		if running, ok := p.Supervisor.Status(); ok &&
			(strings.EqualFold(name, running.Name) || strings.EqualFold(name, running.Model)) {
			return running, nil
		}
		servable, found, err := p.Resolver.Resolve(request.Context(), name)
		if err != nil {
			return Servable{}, err
		}
		if found {
			return servable, nil
		}
		if explicitSwap {
			return Servable{}, fmt.Errorf("model swap: requested model %q is unavailable", name)
		}
		// An unresolvable model name falls through to the running child:
		// serving-side model identifiers (the child's own model id) route
		// to the child that owns them.
	}
	if running, ok := p.Supervisor.Status(); ok {
		return running, nil
	}
	if p.Default.Name != "" {
		return p.Default, nil
	}
	return Servable{}, errNothingServes
}

// replayBody retains the original body's cancellation and close ownership.
type replayBody struct {
	io.Reader
	io.Closer
}

func routingEnvelope(request *http.Request) ([]byte, error) {
	original := request.Body
	reader := bufio.NewReader(original)
	var prefix []byte
	mediaType, _, _ := mime.ParseMediaType(request.Header.Get("Content-Type"))
	if mediaType == "application/x-ndjson" {
		for {
			fragment, err := reader.ReadSlice('\n')
			if len(prefix)+len(fragment) > maxRoutedBodyBytes {
				return nil, errors.New("model swap: routing envelope exceeds request bound")
			}
			prefix = append(prefix, fragment...)
			if errors.Is(err, bufio.ErrBufferFull) {
				continue
			}
			if err != nil && !errors.Is(err, io.EOF) {
				return nil, err
			}
			break
		}
	} else {
		var err error
		prefix, err = io.ReadAll(io.LimitReader(reader, maxRoutedBodyBytes+1))
		if err != nil {
			return nil, err
		}
		if len(prefix) > maxRoutedBodyBytes {
			return nil, errors.New("model swap: routing envelope exceeds request bound")
		}
	}
	request.Body = replayBody{Reader: io.MultiReader(bytes.NewReader(prefix), reader), Closer: original}
	return prefix, nil
}
