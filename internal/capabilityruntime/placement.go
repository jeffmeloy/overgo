package capabilityruntime

import (
	"context"
	"errors"
	"runtime"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
)

// ExactCapabilityPlacement binds one verified model selection to the exact
// runnable implementation that may execute it.
type ExactCapabilityPlacement struct {
	Capability runrecord.CapabilityIdentity
	Selection  modelrecipe.CapabilityEvidenceSelection
	ID         artifact.ID
}

// ResolveExactCapabilityPlacement loads and validates one capability-bound
// local or peer model selection.
func ResolveExactCapabilityPlacement(
	ctx context.Context,
	reader artifact.Reader,
	capabilityID artifact.ID,
	selection modelrecipe.CapabilityEvidenceSelection,
) (ExactCapabilityPlacement, error) {
	if ctx == nil || reader == nil {
		return ExactCapabilityPlacement{}, errors.New("capability runtime: placement authority is absent")
	}
	selection, err := modelrecipe.RefreshCapabilityExecution(ctx, reader, selection)
	if err != nil {
		return ExactCapabilityPlacement{}, err
	}
	capability, err := runrecord.RequireCapabilityIdentity(ctx, reader, capabilityID)
	if err != nil {
		return ExactCapabilityPlacement{}, err
	}
	placement := ExactCapabilityPlacement{Capability: capability, Selection: selection}
	if err := placement.validate(); err != nil {
		return ExactCapabilityPlacement{}, err
	}
	placement.ID, err = artifact.JSONID(artifact.KindProfile, struct {
		Capability artifact.ID `json:"capability"`
		Selection  artifact.ID `json:"selection"`
	}{Capability: capability.ID, Selection: selection.Identity})
	return placement, err
}

func (placement ExactCapabilityPlacement) validate() error {
	definition := placement.Selection.Program.Definition()
	capability := placement.Capability
	if capability.ValidateIdentity() != nil || !placement.Selection.Identity.Valid() ||
		capability.Implementation != definition.Model || placement.Selection.Resources.Recipe != definition.ID ||
		capability.Resources.HostBytes < placement.Selection.Resources.ArtifactBytes {
		return errors.New("capability runtime: capability differs from selected model execution")
	}
	if placement.Selection.Session == modelrecipe.SessionSpillover {
		if capability.Transport.Kind != runrecord.CapabilityTransportPeer || !placement.Selection.Peer.ID.Valid() ||
			capability.Transport.Endpoint != placement.Selection.Peer.Capability.Endpoint {
			return errors.New("capability runtime: peer capability differs from placement")
		}
		return nil
	}
	if capability.Transport.Kind == runrecord.CapabilityTransportPeer ||
		capability.Platform.OS != runtime.GOOS || capability.Platform.Arch != runtime.GOARCH ||
		capability.Resources.CPUThreads > uint32(runtime.NumCPU()) {
		return errors.New("capability runtime: local host does not satisfy capability placement")
	}
	return nil
}

// ExecutePlaced executes only after revalidating the exact capability and
// model selection against the current repository.
func (catalog ExecutorCatalog) ExecutePlaced(
	ctx context.Context,
	store artifact.Repository,
	path string,
	placement ExactCapabilityPlacement,
	raw string,
) (any, error) {
	current, err := ResolveExactCapabilityPlacement(ctx, store, placement.Capability.ID, placement.Selection)
	if err != nil || current.ID != placement.ID {
		return nil, errors.Join(errors.New("capability runtime: exact placement changed"), err)
	}
	return catalog.Execute(ctx, store, path, current.Selection, raw)
}
