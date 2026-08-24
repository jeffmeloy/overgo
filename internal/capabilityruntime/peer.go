package capabilityruntime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"

	"overgo/internal/modelrecipe"
)

const (
	peerModelHeader         = "X-Overgo-Model"
	peerRecipeHeader        = "X-Overgo-Recipe"
	peerResourcesHeader     = "X-Overgo-Resources"
	peerCompatibilityHeader = "X-Overgo-Compatibility"
)

// ExecuteRemotePeer: stream one compatibility-admitted request.
func ExecuteRemotePeer(
	ctx context.Context,
	client *http.Client,
	selection modelrecipe.CapabilityEvidenceSelection,
	input io.Reader,
	output io.Writer,
) (err error) {
	if ctx == nil || selection.Session != modelrecipe.SessionSpillover || !selection.Peer.ID.Valid() ||
		selection.Peer.Capability.ID != selection.Peer.PeerCapability ||
		selection.Peer.Model != selection.Activation.Definition.Model ||
		selection.Peer.Recipe != selection.Activation.Definition.ID ||
		selection.Peer.Resources != selection.Resources.Identity || input == nil || output == nil {
		return errors.New("capability runtime: incomplete remote peer execution")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, selection.Peer.Capability.Endpoint, input)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(peerModelHeader, selection.Activation.Definition.Model.String())
	request.Header.Set(peerRecipeHeader, selection.Activation.Definition.ID.String())
	request.Header.Set(peerResourcesHeader, selection.Resources.Identity.String())
	request.Header.Set(peerCompatibilityHeader, selection.Peer.ID.String())
	if client == nil {
		client = http.DefaultClient
	}
	peerClient := *client
	peerClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err := peerClient.Do(request)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, response.Body.Close()) }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return fmt.Errorf("capability runtime: remote peer status %s", response.Status)
	}
	_, err = io.Copy(output, response.Body)
	return err
}
