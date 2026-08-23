package plan

import (
	"context"
	"errors"

	"overgo/internal/artifact"
)

const (
	leaseOutcomeVersion uint16 = 1
	// LeaseOutcomeMediaType identifies lease outcomes.
	LeaseOutcomeMediaType = "application/vnd.overgo.lease-outcome+json"
	// LeaseOutcomeSchema identifies the lease-outcome schema.
	LeaseOutcomeSchema = "overgo/lease-outcome/v1"
)

// LeaseOutcome: predicted and measured lease resources.
type LeaseOutcome struct {
	Version                 uint16      `json:"version"`
	Lease                   artifact.ID `json:"lease"`
	Predicted               Resources   `json:"predicted"`
	Actual                  Resources   `json:"actual"`
	PredictedWallNS         uint64      `json:"predicted_wall_ns"`
	ActualWallNS            uint64      `json:"actual_wall_ns"`
	PredictedInterferenceNS uint64      `json:"predicted_interference_ns"`
	ActualInterferenceNS    uint64      `json:"actual_interference_ns"`
	Collision               bool        `json:"collision"`
	Abandoned               bool        `json:"abandoned"`
	RecoveryNS              uint64      `json:"recovery_ns"`
	ID                      artifact.ID `json:"-"`
}

var leaseOutcomeCodec = artifact.JSONDocumentCodec("lease outcome", artifact.KindEvidence, LeaseOutcomeMediaType, LeaseOutcomeSchema,
	canonicalizeLeaseOutcome, func(value LeaseOutcome) artifact.ID { return value.ID },
	func(value *LeaseOutcome, id artifact.ID) { value.ID = id }, nil)

// RecordLeaseOutcome: normalize, bind lease, commit.
func RecordLeaseOutcome(ctx context.Context, repository artifact.Repository, data []byte) (LeaseOutcome, error) {
	value, _, err := leaseOutcomeCodec.Normalize(data)
	if err != nil {
		return LeaseOutcome{}, err
	}
	_, ok, err := ReadWorkLease(ctx, repository, value.Lease)
	if err != nil || !ok {
		return LeaseOutcome{}, errors.New("plan: lease outcome references no work lease")
	}
	batch, err := leaseOutcomeCodec.Batch("automation/lease-outcome/"+value.ID.String(), value,
		[]artifact.Lineage{{Child: value.ID, Parent: value.Lease, Relation: artifact.RelationDerivedFrom}}, nil)
	if err != nil {
		return LeaseOutcome{}, err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return value, err
}

// ParseLeaseOutcome: committed measurement decoder.
func ParseLeaseOutcome(content []byte) (LeaseOutcome, error) {
	return leaseOutcomeCodec.Parse(content)
}

func ReadLeaseOutcome(ctx context.Context, reader artifact.Reader, id artifact.ID) (LeaseOutcome, bool, error) {
	return readTypedDocument(ctx, reader, id, leaseOutcomeCodec.Contract, leaseOutcomeCodec.Read)
}

func canonicalizeLeaseOutcome(value *LeaseOutcome) error {
	validResources := func(resources Resources) bool {
		return resources.CPUThreads > 0 && resources.HostRAMGiB > 0 && resources.VRAMGiB >= 0 &&
			(!resources.GPUExclusive || resources.VRAMGiB > 0)
	}
	if value == nil || value.Version != leaseOutcomeVersion || value.Lease.Kind() != artifact.KindEvidence ||
		!validResources(value.Predicted) || !validResources(value.Actual) ||
		value.PredictedWallNS == 0 || value.ActualWallNS == 0 ||
		value.PredictedInterferenceNS > value.PredictedWallNS || value.ActualInterferenceNS > value.ActualWallNS ||
		(value.Collision || value.Abandoned) && value.RecoveryNS == 0 {
		return errors.New("plan: invalid lease outcome")
	}
	return nil
}
