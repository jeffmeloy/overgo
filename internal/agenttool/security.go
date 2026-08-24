package agenttool

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
)

// Endpoint reachability is decided at dial time against resolved
// addresses, not at parse time against host names: a name that
// re-resolves to a private address between validation and connection
// would otherwise walk straight through the policy.

// refuseRedirect refuses every redirect on both executors: a manual
// binds one endpoint, and a response steering the call elsewhere is a
// different endpoint the manual never declared.
func refuseRedirect(request *http.Request, _ []*http.Request) error {
	return fmt.Errorf("agent tool: endpoint redirected to %q; manuals bind one endpoint", request.URL)
}

// privateAddress reports whether the address is loopback, private,
// link-local, or unspecified -- the ranges a server-side fetch must
// not reach on behalf of a manual.
func privateAddress(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() || ip.IsUnspecified()
}

// guardedDialContext resolves the host and dials only an address the
// policy admits, pinning the connection to the checked address so a
// rebinding name cannot swap targets after the check.
func guardedDialContext(allowPrivate bool) func(context.Context, string, string) (net.Conn, error) {
	dialer := &net.Dialer{Timeout: invokeTimeout}
	return func(ctx context.Context, network, address string) (net.Conn, error) {
		host, port, err := net.SplitHostPort(address)
		if err != nil {
			return nil, err
		}
		resolved, err := net.DefaultResolver.LookupIPAddr(ctx, host)
		if err != nil {
			return nil, err
		}
		var refused error
		for _, candidate := range resolved {
			if !allowPrivate && privateAddress(candidate.IP) {
				refused = fmt.Errorf(
					"agent tool: endpoint %q resolves to the private address %s; private networks are refused",
					host, candidate.IP)
				continue
			}
			return dialer.DialContext(ctx, network, net.JoinHostPort(candidate.IP.String(), port))
		}
		if refused != nil {
			return nil, refused
		}
		return nil, errors.New("agent tool: endpoint resolved to no address")
	}
}

func newTransportClient(allowPrivate bool) *http.Client {
	return &http.Client{
		CheckRedirect: refuseRedirect,
		Transport:     &http.Transport{DialContext: guardedDialContext(allowPrivate)},
		Timeout:       invokeTimeout,
	}
}
