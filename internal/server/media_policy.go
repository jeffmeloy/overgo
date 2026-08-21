package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"
)

const (
	maxRemoteMediaHostBytes  = 253
	maxRemoteMediaLabelBytes = 63
	defaultHTTPPort          = "80"
	defaultHTTPSPort         = "443"
)

type RemoteMediaPolicy struct {
	Enabled               bool
	AllowedSchemes        []string
	AllowedHosts          []string
	AllowedPorts          []int
	AllowPrivateNetworks  bool
	MaxRedirects          int
	MaxConcurrentFetches  int
	ConnectTimeout        time.Duration
	ResponseHeaderTimeout time.Duration
	TotalTimeout          time.Duration
	MaxResponseBytes      int64
}

type mediaPolicyDocument struct {
	Schema int `json:"schema"`
	Remote struct {
		Enabled               bool     `json:"enabled"`
		AllowedSchemes        []string `json:"allowed_schemes"`
		AllowedHosts          []string `json:"allowed_hosts"`
		AllowedPorts          []int    `json:"allowed_ports"`
		AllowPrivateNetworks  bool     `json:"allow_private_networks"`
		MaxRedirects          int      `json:"max_redirects"`
		MaxConcurrentFetches  int      `json:"max_concurrent_fetches"`
		ConnectTimeout        string   `json:"connect_timeout"`
		ResponseHeaderTimeout string   `json:"response_header_timeout"`
		TotalTimeout          string   `json:"total_timeout"`
		MaxResponseBytes      int64    `json:"max_response_bytes"`
	} `json:"remote_media"`
}

func LoadRemoteMediaPolicy(path string) (*RemoteMediaPolicy, error) {
	var document mediaPolicyDocument
	if err := loadPolicyDocument(path, &document); err != nil {
		return nil, fmt.Errorf("load remote media policy: %w", err)
	}
	if document.Schema != policyDocumentSchemaVersion {
		return nil, fmt.Errorf(
			"remote media policy schema = %d, want %d", document.Schema, policyDocumentSchemaVersion,
		)
	}
	connect, err := time.ParseDuration(document.Remote.ConnectTimeout)
	if err != nil {
		return nil, fmt.Errorf("remote media policy connect_timeout: %w", err)
	}
	header, err := time.ParseDuration(document.Remote.ResponseHeaderTimeout)
	if err != nil {
		return nil, fmt.Errorf("remote media policy response_header_timeout: %w", err)
	}
	total, err := time.ParseDuration(document.Remote.TotalTimeout)
	if err != nil {
		return nil, fmt.Errorf("remote media policy total_timeout: %w", err)
	}
	policy := &RemoteMediaPolicy{
		Enabled:               document.Remote.Enabled,
		AllowedSchemes:        slices.Clone(document.Remote.AllowedSchemes),
		AllowedHosts:          slices.Clone(document.Remote.AllowedHosts),
		AllowedPorts:          slices.Clone(document.Remote.AllowedPorts),
		AllowPrivateNetworks:  document.Remote.AllowPrivateNetworks,
		MaxRedirects:          document.Remote.MaxRedirects,
		MaxConcurrentFetches:  document.Remote.MaxConcurrentFetches,
		ConnectTimeout:        connect,
		ResponseHeaderTimeout: header,
		TotalTimeout:          total,
		MaxResponseBytes:      document.Remote.MaxResponseBytes,
	}
	if err := validateRemoteMediaPolicy(policy); err != nil {
		return nil, err
	}
	return policy, nil
}

func validateRemoteMediaPolicy(policy *RemoteMediaPolicy) error {
	if policy == nil {
		return nil
	}
	if len(policy.AllowedSchemes) == 0 {
		return errors.New("remote media policy requires an allowed scheme")
	}
	for index, scheme := range policy.AllowedSchemes {
		scheme = strings.ToLower(strings.TrimSpace(scheme))
		if scheme != "https" && scheme != "http" {
			return fmt.Errorf("remote media policy scheme %q is unsupported", scheme)
		}
		policy.AllowedSchemes[index] = scheme
	}
	slices.Sort(policy.AllowedSchemes)
	policy.AllowedSchemes = slices.Compact(policy.AllowedSchemes)
	if policy.Enabled && len(policy.AllowedHosts) == 0 {
		return errors.New("enabled remote media policy requires an explicit host allowlist")
	}
	for index, pattern := range policy.AllowedHosts {
		normalized, err := normalizeAllowedHost(pattern)
		if err != nil {
			return fmt.Errorf("remote media policy host %d: %w", index, err)
		}
		policy.AllowedHosts[index] = normalized
	}
	if len(policy.AllowedPorts) == 0 {
		return errors.New("remote media policy requires an allowed port")
	}
	for _, port := range policy.AllowedPorts {
		if port < 1 || port > math.MaxUint16 {
			return fmt.Errorf("remote media policy port %d is invalid", port)
		}
	}
	slices.Sort(policy.AllowedPorts)
	policy.AllowedPorts = slices.Compact(policy.AllowedPorts)
	if policy.MaxRedirects < 0 {
		return errors.New("remote media policy max_redirects must be non-negative")
	}
	if policy.MaxConcurrentFetches < 1 {
		return errors.New("remote media policy max_concurrent_fetches must be positive")
	}
	for name, value := range map[string]time.Duration{
		"connect_timeout":         policy.ConnectTimeout,
		"response_header_timeout": policy.ResponseHeaderTimeout,
		"total_timeout":           policy.TotalTimeout,
	} {
		if value <= 0 {
			return fmt.Errorf("remote media policy %s must be positive", name)
		}
	}
	if policy.TotalTimeout < policy.ConnectTimeout {
		return errors.New("remote media policy total_timeout is shorter than connect_timeout")
	}
	if policy.MaxResponseBytes <= 0 {
		return errors.New("remote media policy max_response_bytes must be positive")
	}
	return nil
}

func normalizeAllowedHost(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	if strings.HasSuffix(value, ".") {
		value = strings.TrimSuffix(value, ".")
	}
	if address, err := netip.ParseAddr(value); err == nil {
		return address.Unmap().String(), nil
	}
	if strings.HasPrefix(value, "*.") {
		if strings.Count(value, "*") != 1 {
			return "", errors.New("wildcard must be one leading '*.'")
		}
		value = "*." + strings.TrimPrefix(value, "*.")
	}
	host := strings.TrimPrefix(value, "*.")
	if host == "" || len(host) > maxRemoteMediaHostBytes || strings.ContainsAny(host, "/:@?#%") {
		return "", errors.New("host pattern is invalid")
	}
	for _, label := range strings.Split(host, ".") {
		if label == "" || len(label) > maxRemoteMediaLabelBytes || label[0] == '-' || label[len(label)-1] == '-' {
			return "", errors.New("host label is invalid")
		}
		for _, character := range label {
			if (character < 'a' || character > 'z') && (character < '0' || character > '9') && character != '-' {
				return "", errors.New("host must use ASCII DNS labels")
			}
		}
	}
	return value, nil
}

type remoteMediaFetcher struct {
	policy RemoteMediaPolicy
	client *http.Client
	slots  chan struct{}
}

func newRemoteMediaFetcher(policy *RemoteMediaPolicy) (*remoteMediaFetcher, error) {
	if policy == nil {
		return nil, nil
	}
	copyPolicy := *policy
	copyPolicy.AllowedSchemes = slices.Clone(policy.AllowedSchemes)
	copyPolicy.AllowedHosts = slices.Clone(policy.AllowedHosts)
	copyPolicy.AllowedPorts = slices.Clone(policy.AllowedPorts)
	if err := validateRemoteMediaPolicy(&copyPolicy); err != nil {
		return nil, err
	}
	fetcher := &remoteMediaFetcher{policy: copyPolicy, slots: make(chan struct{}, copyPolicy.MaxConcurrentFetches)}
	dialer := &net.Dialer{Timeout: copyPolicy.ConnectTimeout, KeepAlive: -1}
	transport := &http.Transport{
		Proxy: nil,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(address)
			if err != nil {
				return nil, fmt.Errorf("remote media dial address: %w", err)
			}
			addresses, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
			if err != nil || len(addresses) == 0 {
				return nil, fmt.Errorf("remote media DNS lookup failed")
			}
			for _, address := range addresses {
				if !copyPolicy.AllowPrivateNetworks && !publicMediaIP(address) {
					return nil, errors.New("remote media DNS resolved to a prohibited network")
				}
			}
			var dialErr error
			for _, resolved := range addresses {
				conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(resolved.String(), port))
				if err == nil {
					return conn, nil
				}
				dialErr = err
			}
			return nil, fmt.Errorf("remote media connection failed: %w", dialErr)
		},
		DisableKeepAlives:     true,
		DisableCompression:    true,
		ResponseHeaderTimeout: copyPolicy.ResponseHeaderTimeout,
		TLSHandshakeTimeout:   copyPolicy.ConnectTimeout,
	}
	fetcher.client = &http.Client{
		Transport: transport,
		Timeout:   copyPolicy.TotalTimeout,
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			if len(via) > copyPolicy.MaxRedirects {
				return errors.New("remote media redirect limit exceeded")
			}
			return fetcher.validateURL(request.URL)
		},
	}
	return fetcher, nil
}

func (f *remoteMediaFetcher) validateURL(target *url.URL) error {
	if f == nil || !f.policy.Enabled {
		return errors.New("remote media URLs are disabled")
	}
	if target == nil || !slices.Contains(f.policy.AllowedSchemes, strings.ToLower(target.Scheme)) {
		return errors.New("remote media URL scheme is not allowed")
	}
	if target.User != nil || target.Fragment != "" {
		return errors.New("remote media URL credentials and fragments are prohibited")
	}
	host := strings.ToLower(strings.TrimSuffix(target.Hostname(), "."))
	if !f.hostAllowed(host) {
		return errors.New("remote media URL host is not allowlisted")
	}
	port := target.Port()
	if port == "" {
		if strings.EqualFold(target.Scheme, "https") {
			port = defaultHTTPSPort
		} else {
			port = defaultHTTPPort
		}
	}
	portNumber, err := strconv.Atoi(port)
	if err != nil || !slices.Contains(f.policy.AllowedPorts, portNumber) {
		return errors.New("remote media URL port is not allowed")
	}
	return nil
}

func (f *remoteMediaFetcher) hostAllowed(host string) bool {
	for _, pattern := range f.policy.AllowedHosts {
		if strings.HasPrefix(pattern, "*.") {
			suffix := strings.TrimPrefix(pattern, "*")
			if strings.HasSuffix(host, suffix) && host != strings.TrimPrefix(suffix, ".") {
				return true
			}
		} else if host == pattern {
			return true
		}
	}
	return false
}

func (f *remoteMediaFetcher) fetch(ctx context.Context, source, mediaType string) ([]byte, error) {
	data, _, err := f.fetchTyped(ctx, source, mediaType+"/*", func(contentType string) bool {
		return strings.HasPrefix(contentType, mediaType+"/")
	})
	return data, err
}

func (f *remoteMediaFetcher) fetchDocument(ctx context.Context, source string) ([]byte, string, error) {
	return f.fetchTyped(ctx, source, "text/*, application/json, application/xml", supportedResponseTextType)
}

func (f *remoteMediaFetcher) fetchTyped(
	ctx context.Context,
	source, accept string,
	acceptType func(string) bool,
) ([]byte, string, error) {
	source = strings.TrimSpace(source)
	target, err := url.Parse(source)
	if err != nil {
		return nil, "", errors.New("remote media URL is invalid")
	}
	if err := f.validateURL(target); err != nil {
		return nil, "", err
	}
	select {
	case f.slots <- struct{}{}:
		defer func() { <-f.slots }()
	case <-ctx.Done():
		return nil, "", ctx.Err()
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return nil, "", errors.New("remote media request is invalid")
	}
	request.Header.Set("Accept", accept)
	request.Header.Set("User-Agent", "overgo-media/1")
	response, err := f.client.Do(request)
	if err != nil {
		return nil, "", fmt.Errorf("fetch remote media: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, "", fmt.Errorf("remote media returned HTTP status %d", response.StatusCode)
	}
	contentType := strings.ToLower(strings.TrimSpace(strings.Split(response.Header.Get("Content-Type"), ";")[0]))
	if !acceptType(contentType) {
		return nil, "", fmt.Errorf("remote media content type %q is unsupported", contentType)
	}
	if response.ContentLength > f.policy.MaxResponseBytes {
		return nil, "", errors.New("remote media response exceeds byte limit")
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, f.policy.MaxResponseBytes+1))
	if err != nil {
		return nil, "", fmt.Errorf("read remote media: %w", err)
	}
	if len(data) == 0 {
		return nil, "", errors.New("remote media response is empty")
	}
	if int64(len(data)) > f.policy.MaxResponseBytes {
		return nil, "", errors.New("remote media response exceeds byte limit")
	}
	return data, contentType, nil
}

func publicMediaIP(address netip.Addr) bool {
	address = address.Unmap()
	if !address.IsValid() || !address.IsGlobalUnicast() {
		return false
	}
	for _, prefix := range prohibitedMediaNetworks {
		if prefix.Contains(address) {
			return false
		}
	}
	return true
}

var prohibitedMediaNetworks = []netip.Prefix{
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
