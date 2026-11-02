package agenttool

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
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

// PublicAddress reports whether a server-side fetch may reach the address: global unicast and in no refused network. A
// tool's endpoint and a media source answer to this one list.
func PublicAddress(address netip.Addr) bool {
	address = address.Unmap()
	if !address.IsValid() || !address.IsGlobalUnicast() {
		return false
	}
	for _, prefix := range refusedNetworks {
		if prefix.Contains(address) {
			return false
		}
	}
	return true
}

var refusedNetworks = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"), netip.MustParsePrefix("10.0.0.0/8"),
	netip.MustParsePrefix("100.64.0.0/10"), netip.MustParsePrefix("127.0.0.0/8"),
	netip.MustParsePrefix("169.254.0.0/16"), netip.MustParsePrefix("172.16.0.0/12"),
	netip.MustParsePrefix("192.0.0.0/24"), netip.MustParsePrefix("192.0.2.0/24"),
	netip.MustParsePrefix("192.168.0.0/16"), netip.MustParsePrefix("198.18.0.0/15"),
	netip.MustParsePrefix("198.51.100.0/24"), netip.MustParsePrefix("203.0.113.0/24"),
	netip.MustParsePrefix("224.0.0.0/4"), netip.MustParsePrefix("240.0.0.0/4"),
	netip.MustParsePrefix("::/128"), netip.MustParsePrefix("::1/128"),
	netip.MustParsePrefix("64:ff9b::/96"), netip.MustParsePrefix("100::/64"),
	netip.MustParsePrefix("2001::/23"), netip.MustParsePrefix("2001:db8::/32"),
	netip.MustParsePrefix("fc00::/7"), netip.MustParsePrefix("fe80::/10"),
	netip.MustParsePrefix("ff00::/8"),
}

// guardedDialContext resolves the host and dials only an address the
// policy admits, pinning the connection to the checked address so a
// rebinding name cannot swap targets after the check.
func guardedDialContext(allowPrivate bool) func(context.Context, string, string) (net.Conn, error) {
	dialer := &net.Dialer{}
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
			address, parsed := netip.AddrFromSlice(candidate.IP)
			if !allowPrivate && (!parsed || !PublicAddress(address)) {
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
	}
}
