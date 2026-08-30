package capabilityruntime

import (
	"context"
	"sync"

	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// DeltaSource is the canonical store surface a head-bound view reconciles
// against.
type DeltaSource interface {
	Head() (artifact.CommitID, uint64)
	DeltasSince(
		ctx context.Context,
		previousHead artifact.CommitID,
		previousSequence uint64,
		projectionVersion string,
		maxCommits int,
	) (overgodb.HeadBoundDelta, bool, error)
}

// headBoundDeltaWindow bounds one reconciliation walk; a busier tail simply
// reports changed and lets the consumer re-derive.
const headBoundDeltaWindow = 64

// HeadBoundView tracks one consumer's exact reconciled store coordinate so a
// repeated freshness check stops re-resolving authority when the canonical
// store has not moved. It never holds derived state itself: the consumer owns
// its view and this type owns only the continuity coordinate.
type HeadBoundView struct {
	mu         sync.Mutex
	head       artifact.CommitID
	sequence   uint64
	projection string
}

// Reconcile reports how the canonical store moved relative to this view.
// changed=false means the consumer's derived state is exactly current and
// re-derivation may be skipped. changed=true hands over the ordered coalesced
// delta and advances the view to the delta's resulting head. resync=true
// means no delta can honestly bridge — an unseeded view, broken continuity,
// or a foreign projection contract — and the consumer must rebuild its
// derived state from the current snapshot; the view advances to that head.
func (v *HeadBoundView) Reconcile(
	ctx context.Context,
	source DeltaSource,
) (changed, resync bool, delta overgodb.HeadBoundDelta, err error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.projection == "" {
		v.projection = overgodb.ProjectionContractVersion()
	}
	delta, resync, err = source.DeltasSince(ctx, v.head, v.sequence, v.projection, headBoundDeltaWindow)
	if err != nil {
		return false, false, overgodb.HeadBoundDelta{}, err
	}
	if resync {
		v.head, v.sequence = source.Head()
		v.projection = overgodb.ProjectionContractVersion()
		return true, true, overgodb.HeadBoundDelta{}, nil
	}
	if len(delta.Commits) == 0 && !delta.Truncated {
		return false, false, delta, nil
	}
	v.head, v.sequence = delta.Head, delta.Sequence
	return true, false, delta, nil
}
