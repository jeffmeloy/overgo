package remoteprovider

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/url"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// The record of a variable the operator named for providers.
const (
	variableMediaType = "application/vnd.overgo.remote-provider-variable+json"
	variableSchema    = "overgo/remote-provider-variable/v1"
)

// namedVariable is an environment variable the operator named for providers: a declaration binds no other, so a caller
// who holds the server's API key cannot have the value of another of the host's variables sent to an endpoint of its
// choosing. Only cmd/remote-provider names one, which no HTTP route reaches.
type namedVariable struct {
	Version uint16      `json:"version"`
	Name    string      `json:"name"`
	ID      artifact.ID `json:"-"`
}

var variableCodec = artifact.JSONDocumentCodec(
	"remote provider variable", artifact.KindProfile, variableMediaType, variableSchema,
	func(value *namedVariable) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion {
			return errors.New("remote provider: invalid variable version")
		}
		value.Name = strings.TrimSpace(value.Name)
		if value.Name == "" || strings.ContainsAny(value.Name, " =") {
			return errors.New("remote provider: a variable is one name")
		}
		if slices.Contains(serverSecrets, value.Name) {
			return fmt.Errorf("remote provider: %s is the server's own secret", value.Name)
		}
		return nil
	},
	func(value namedVariable) artifact.ID { return value.ID },
	func(value *namedVariable, id artifact.ID) { value.ID = id },
)

// NameVariable records name as a variable the operator names for providers' keys; naming one twice changes nothing.
func NameVariable(ctx context.Context, store artifact.Repository, name string) error {
	named, err := variableCodec.NewInitial(namedVariable{Name: name})
	if err != nil {
		return err
	}
	content, err := variableCodec.Content(named)
	if err != nil {
		return err
	}
	_, err = artifact.Publish(ctx, store, artifact.Batch{Key: "remote-provider/variable/" + named.Name, Contents: []artifact.Content{content}})
	return err
}

// namedVariables are the variables the operator named for providers.
func namedVariables(ctx context.Context, store *overgodb.Store) ([]string, error) {
	var names []string
	_, err := overgodb.VisitDecodedDocuments(ctx, store, overgodb.DocumentQuery{
		Contracts: []artifact.DocumentContract{{Kind: artifact.KindProfile, MediaType: variableMediaType, Schema: variableSchema}},
		Order:     overgodb.DocumentOldestFirst,
	}, variableCodec.Parse, func(_ overgodb.DocumentView, named namedVariable) error {
		if !slices.Contains(names, named.Name) {
			names = append(names, named.Name)
		}
		return nil
	})
	return names, err
}

// Admitted refuses a variable the operator has not named for providers, by its name and how to name it.
func Admitted(ctx context.Context, store *overgodb.Store, name string) error {
	named, err := namedVariables(ctx, store)
	if err != nil {
		return err
	}
	if !slices.Contains(named, strings.TrimSpace(name)) {
		return fmt.Errorf("remote provider: %s is not a variable the operator named for providers' keys; name it with go run ./cmd/remote-provider -name-variable %s", name, name)
	}
	return nil
}

// KeyTransport refuses an endpoint a key would cross the network to in the clear: a key goes over https, or over http
// to this machine alone.
func KeyTransport(endpoint string) error {
	parsed, err := url.Parse(CanonicalEndpoint(endpoint))
	if err != nil {
		return err
	}
	if parsed.Scheme == "https" {
		return nil
	}
	host := parsed.Hostname()
	if ip := net.ParseIP(host); parsed.Scheme == "http" && (host == "localhost" || ip != nil && ip.IsLoopback()) {
		return nil
	}
	return fmt.Errorf("remote provider: a key goes to %s only over https, or over http to this machine", endpoint)
}
