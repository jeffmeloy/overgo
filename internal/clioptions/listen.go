package clioptions

import (
	"fmt"
	"net"
)

// RequireLoopbackWithoutCredential refuses a listen address that reaches
// beyond this host while the serving routes carry no bearer credential.
// Loopback binds stay open for local operation; every other bind -- an
// explicit interface address, a hostname, or an all-interfaces bind with an
// empty host -- requires a credential so unauthenticated routes are never
// exposed past the machine. A hostname other than localhost is refused
// rather than resolved: resolution is environment-dependent, and the guard
// only accepts what it can prove.
func RequireLoopbackWithoutCredential(address, credential string) error {
	if credential != "" {
		return nil
	}
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		return fmt.Errorf("listen address %q: %w", address, err)
	}
	if host == "localhost" {
		return nil
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return nil
	}
	return fmt.Errorf(
		"listen address %q reaches beyond this host and no route credential is configured; set a bearer credential or bind a loopback address",
		address,
	)
}
