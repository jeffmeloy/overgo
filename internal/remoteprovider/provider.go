// Package remoteprovider declares remote inference providers in the
// store. A provider document names the endpoint, the environment variable
// holding the key, and the model id a hosted service serves under the
// OpenAI-compatible chat completions protocol. Declaring one commits a
// model manifest whose only component is that document, recorded at a
// remote location, and activates a remote inference recipe over it, so
// the servable catalog lists the model beside local ones and the server
// relays to it. Nothing about the hosted model is reproducible from the
// store: its environment records the remote backend, and every run under
// it carries that mark.
package remoteprovider

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

const (
	// MediaType is the provider document's media type.
	MediaType = "application/vnd.overgo.remote-provider+json"
	// Schema is the provider document's schema identity.
	Schema = "overgo/remote-provider/v1"
	// Protocol names the wire protocol every declared provider speaks: the
	// OpenAI chat completions request and its streamed chunks, which
	// OpenRouter serves for every model it lists.
	Protocol = "openai-chat-completions"
	// locationScheme prefixes the remote location a declared model is
	// served by and listed under.
	locationScheme = "remote://"
)

// Provider is one hosted model: the service name, its API endpoint (the
// base the chat completions path joins), the environment variable whose
// value is the key, the model id the service routes on, and the context
// length the service declares for it (zero when unstated).
type Provider struct {
	Version        uint16      `json:"version"`
	Name           string      `json:"name"`
	Endpoint       string      `json:"endpoint"`
	KeyEnvironment string      `json:"key_environment"`
	Model          string      `json:"model"`
	ContextLength  uint32      `json:"context_length,omitzero"`
	ID             artifact.ID `json:"-"`
}

var codec = artifact.JSONDocumentCodec(
	"remote provider", artifact.KindProfile, MediaType, Schema, canonicalize,
	func(value Provider) artifact.ID { return value.ID },
	func(value *Provider, id artifact.ID) { value.ID = id }, nil,
)

func canonicalize(provider *Provider) error {
	if provider == nil || provider.Version != artifact.InitialDocumentVersion {
		return errors.New("remote provider: invalid document version")
	}
	provider.Name = strings.TrimSpace(provider.Name)
	provider.Endpoint = strings.TrimRight(strings.TrimSpace(provider.Endpoint), "/")
	provider.KeyEnvironment = strings.TrimSpace(provider.KeyEnvironment)
	provider.Model = strings.TrimSpace(provider.Model)
	if provider.Name == "" || strings.ContainsAny(provider.Name, "/ ") {
		return errors.New("remote provider: name must be one word without slashes")
	}
	endpoint, err := url.Parse(provider.Endpoint)
	if err != nil || endpoint.Host == "" || (endpoint.Scheme != "https" && endpoint.Scheme != "http") {
		return fmt.Errorf("remote provider: endpoint %q must be an http(s) URL", provider.Endpoint)
	}
	if provider.KeyEnvironment == "" || provider.Model == "" {
		return errors.New("remote provider: key environment variable and model id are required")
	}
	return nil
}

// New validates a declaration and derives its identity.
func New(provider Provider) (Provider, error) {
	return codec.NewInitial(provider)
}

// Location names the remote model the way the servable catalog lists it
// and the server is told to serve it.
func Location(provider Provider) string {
	return locationScheme + provider.Name + "/" + provider.Model
}

// IsRemoteLocation reports whether a serving reference names a hosted
// model rather than bytes on disk.
func IsRemoteLocation(reference string) bool {
	return strings.HasPrefix(reference, locationScheme)
}

// SetKey places a declared model's provider key in this process's
// environment, never on disk: from now on the catalog lists the model
// servable and the relay signs with the key. The reference names the
// model by location or identity; the provider is returned so the caller
// can name the variable that now holds the key.
func SetKey(ctx context.Context, store *overgodb.Store, limit int, reference, key string) (Provider, error) {
	if strings.TrimSpace(key) == "" {
		return Provider{}, errors.New("remote provider: key is empty")
	}
	provider, _, remote, err := Reference(ctx, store, limit, reference)
	if err != nil {
		return Provider{}, err
	}
	if !remote {
		return Provider{}, fmt.Errorf("remote provider: %s names no declared model", reference)
	}
	if err := os.Setenv(provider.KeyEnvironment, key); err != nil {
		return Provider{}, err
	}
	return provider, nil
}

// Refusal names why the provider cannot serve now: its key is absent from
// the environment. An empty refusal means the key is present.
func Refusal(provider Provider) string {
	if os.Getenv(provider.KeyEnvironment) == "" {
		return provider.KeyEnvironment + " is not set"
	}
	return ""
}

// Key reads the provider's key from its environment variable.
func Key(provider Provider) (string, error) {
	if refusal := Refusal(provider); refusal != "" {
		return "", errors.New("remote provider: " + refusal)
	}
	return os.Getenv(provider.KeyEnvironment), nil
}

// Environment is the execution environment every run at the provider
// records: the endpoint host, the remote backend, the provider as the
// driver and the protocol as the runtime, so the record says the run is
// not reproducible from the store.
func Environment(provider Provider) (runrecord.Environment, error) {
	endpoint, err := url.Parse(provider.Endpoint)
	if err != nil {
		return runrecord.Environment{}, err
	}
	return runrecord.NewEnvironment(runrecord.Environment{
		Host: endpoint.Host, OS: runrecord.BackendRemote, Arch: runrecord.BackendRemote,
		Device: provider.Model, Backend: runrecord.BackendRemote, Driver: provider.Name, Runtime: Protocol,
	})
}
