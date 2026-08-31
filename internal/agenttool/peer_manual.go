package agenttool

import (
	"errors"

	"overgo/internal/artifact"
)

// PeerCapabilityManual derives the exact UTCP manual one remote peer
// capability is invoked through: a sequenced http-json-stream transport
// bound to the published endpoint, whose content-addressed identity closes
// over the exact capability. Internal Go never routes through this manual
// -- it is the outward boundary, and only the boundary.
func PeerCapabilityManual(endpoint string, capability artifact.ID, task string) (Manual, error) {
	if endpoint == "" || !capability.Valid() || task == "" {
		return Manual{}, errors.New("agent tool: peer capability manual requires an endpoint, capability, and task")
	}
	// The capability identity rides the manual's own content: an http
	// transport binds no target, so the description carries the exact
	// capability the manual identity closes over.
	return NewManual(Manual{
		Name:        "peer." + task,
		Description: "Invoke the compatibility-admitted remote peer capability " + capability.String() + ".",
		Effect:      EffectInspection,
		Transport:   Transport{Kind: TransportHTTPJSONStream, URL: endpoint},
	})
}
