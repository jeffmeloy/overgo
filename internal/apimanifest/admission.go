package apimanifest

import (
	"net"
	"net/http"
	"net/url"
	"strings"
)

// Refusal says why a credential-less request is not admitted, with the
// error type the API envelope carries.
type Refusal struct {
	Type    string
	Message string
}

// Error implements error.
func (r *Refusal) Error() string { return r.Message }

// RefusalHost is the refusal type for a Host that is not loopback on a real listener.
const RefusalHost = "invalid_host"

// RefusalOrigin is the refusal type for a request the browser's cross-origin protection refuses.
const RefusalOrigin = "cross_origin_request"

// AdmitCredentialless admits a request on a transport that carries no
// bearer credential: a loopback Host on a real listener (a matching Origin
// alone cannot stop DNS rebinding; an embedded handler has no listener
// authority) and the browser's cross-origin protection. The server applies
// it to every credential-less route and the swap proxy applies it before
// any mutation of its own, so a foreign origin can neither reach a route
// nor turn a proxy-side key or launch.
func AdmitCredentialless(request *http.Request) *Refusal {
	if request.Context().Value(http.LocalAddrContextKey) != nil {
		host := (&url.URL{Host: request.Host}).Hostname()
		if !strings.EqualFold(host, "localhost") && !net.ParseIP(host).IsLoopback() {
			return &Refusal{Type: RefusalHost, Message: "credential-less requests require a loopback Host"}
		}
	}
	var protection http.CrossOriginProtection
	if err := protection.Check(request); err != nil {
		return &Refusal{Type: RefusalOrigin, Message: err.Error()}
	}
	return nil
}

// AdmitCredentiallessEffect admits a request whose handling has a side
// effect whatever its method: the proxy's swap on the health probe and the
// provider listing that sends a key to the named endpoint are GETs, and
// the browser's cross-origin protection reads fetch metadata and Origin
// only for unsafe methods, so the check runs over the request as a POST.
// A loopback client without fetch metadata still passes.
func AdmitCredentiallessEffect(request *http.Request) *Refusal {
	if refusal := AdmitCredentialless(request); refusal != nil {
		return refusal
	}
	unsafe := request.Clone(request.Context())
	unsafe.Method = http.MethodPost
	var protection http.CrossOriginProtection
	if err := protection.Check(unsafe); err != nil {
		return &Refusal{Type: RefusalOrigin, Message: err.Error()}
	}
	return nil
}
