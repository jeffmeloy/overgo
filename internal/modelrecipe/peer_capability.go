package modelrecipe

import (
	"context"
	"errors"
	"net/url"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const remotePeerCapabilityVersion uint16 = 1

var remotePeerCapabilityCodec = artifact.JSONDocumentCodec(
	"remote peer capability", artifact.KindEvidence,
	"application/vnd.overgo.remote-peer-capability+json", "overgo/remote-peer-capability/v1",
	canonicalizeRemotePeerCapability,
	func(value RemotePeerCapability) artifact.ID { return value.ID },
	func(value *RemotePeerCapability, id artifact.ID) { value.ID = id },
	func(value RemotePeerCapability) RemotePeerCapability {
		value.Tasks = slices.Clone(value.Tasks)
		return value
	},
)

// RemotePeerCapability defines one peer's declared task transport.
type RemotePeerCapability struct {
	Version     uint16        `json:"version"`
	Environment artifact.ID   `json:"environment"`
	Tasks       []recipe.Task `json:"tasks"`
	Endpoint    string        `json:"endpoint"`
	ID          artifact.ID   `json:"-"`
}

// NewRemotePeerCapability identifies one immutable declared task transport.
func NewRemotePeerCapability(value RemotePeerCapability) (RemotePeerCapability, error) {
	return remotePeerCapabilityCodec.NewInitial(value)
}

// RequireRemotePeerCapability returns one validated immutable declaration.
func RequireRemotePeerCapability(
	ctx context.Context,
	reader artifact.Reader,
	id artifact.ID,
) (RemotePeerCapability, error) {
	return remotePeerCapabilityCodec.Require(ctx, reader, id)
}

func canonicalizeRemotePeerCapability(value *RemotePeerCapability) error {
	if value == nil || value.Version != remotePeerCapabilityVersion ||
		value.Environment.Kind() != artifact.KindEvidence || len(value.Tasks) == 0 {
		return errors.New("model recipe: invalid remote peer capability")
	}
	for _, task := range value.Tasks {
		if !task.Valid() {
			return errors.New("model recipe: invalid remote peer capability task")
		}
	}
	slices.Sort(value.Tasks)
	if len(slices.Compact(slices.Clone(value.Tasks))) != len(value.Tasks) {
		return errors.New("model recipe: duplicate remote peer capability task")
	}
	return validateRemoteEndpoint(value.Endpoint)
}

func validateRemoteEndpoint(raw string) error {
	endpoint, err := url.ParseRequestURI(raw)
	if err != nil || (endpoint.Scheme != "http" && endpoint.Scheme != "https") || endpoint.Host == "" ||
		endpoint.User != nil || endpoint.Fragment != "" {
		return errors.New("model recipe: invalid remote peer endpoint")
	}
	return nil
}
