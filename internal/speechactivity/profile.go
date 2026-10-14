package speechactivity

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
)

// Profile binds operation declarations to exact model, inventory and license
// artifacts. Exactly one offline or streaming boundary policy is permitted.
type Profile struct {
	Version   uint16                  `json:"version"`
	Model     artifact.ID             `json:"model"`
	Inventory artifact.ID             `json:"inventory"`
	License   artifact.ID             `json:"license"`
	Frontend  audiodsp.FrontendConfig `json:"frontend"`
	Network   Declaration             `json:"network"`
	Offline   *OfflineConfig          `json:"offline,omitzero"`
	Streaming *BoundaryConfig         `json:"streaming,omitzero"`
	ID        artifact.ID             `json:"-"`
}

var profileCodec = artifact.JSONDocumentCodec("speech activity profile", artifact.KindProfile,
	"application/vnd.overgo.speech-activity-profile+json", "overgo/speech-activity-profile/v1",
	func(p *Profile) error {
		if p.Version != artifact.InitialDocumentVersion || p.Model.Kind() != artifact.KindModel || p.Inventory.Kind() != artifact.KindTensorInventory || p.License.Kind() != artifact.KindFile ||
			(p.Offline == nil) == (p.Streaming == nil) || len(p.Network.Blocks) == 0 || len(p.Network.Dense) == 0 {
			return errors.New("speech activity: incomplete profile")
		}
		return nil
	}, func(p Profile) artifact.ID { return p.ID }, func(p *Profile, id artifact.ID) { p.ID = id })

// RequireProfile reads a canonical, content-addressed execution declaration.
func RequireProfile(ctx context.Context, reader artifact.Reader, id artifact.ID) (Profile, error) {
	return profileCodec.Require(ctx, reader, id)
}
