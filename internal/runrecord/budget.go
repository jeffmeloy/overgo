package runrecord

import (
	"errors"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	budgetMediaType = "application/vnd.overgo.budget-grant+json"
	budgetSchema    = "overgo/budget-grant/v1"
)

// BudgetGrant is an immutable limit. Consumers derive usage from evidence and
// ask Allows; the grant never becomes a mutable counter.
type BudgetGrant struct {
	Owner     artifact.ID `json:"owner"`
	Subject   artifact.ID `json:"subject"`
	Unit      string      `json:"unit"`
	Issued    uint64      `json:"issued"`
	ExpiresAt string      `json:"expires_at"`
	ID        artifact.ID `json:"-"`
}

var budgetCodec = artifact.JSONDocumentCodec(
	"budget grant", artifact.KindEvidence, budgetMediaType, budgetSchema, canonicalizeBudget,
	func(value BudgetGrant) artifact.ID { return value.ID },
	func(value *BudgetGrant, id artifact.ID) { value.ID = id }, nil,
)

func NewBudgetGrant(value BudgetGrant) (BudgetGrant, error) {
	return budgetCodec.New(value)
}

// Allows checks a derived consumed amount without creating another authority.
func (value BudgetGrant) Allows(subject artifact.ID, consumed, requested uint64, now time.Time) error {
	if err := budgetCodec.ValidateIdentity(value); err != nil {
		return err
	}
	if subject != value.Subject {
		return errors.New("run record: budget subject differs")
	}
	expires, _ := time.Parse(time.RFC3339Nano, value.ExpiresAt)
	if requested == 0 || consumed > value.Issued || requested > value.Issued-consumed {
		return errors.New("run record: budget exhausted")
	}
	if !now.Before(expires) {
		return errors.New("run record: budget expired")
	}
	return nil
}

func (value BudgetGrant) Batch(key string) (artifact.Batch, error) {
	return artifact.DependencyDocumentBatch(key, budgetCodec, value, value.Owner, value.Subject)
}

func canonicalizeBudget(value *BudgetGrant) error {
	if value == nil || value.Owner.Kind() != artifact.KindEvidence || !value.Subject.Valid() ||
		!textcheck.LowerIdentifier(value.Unit, maxLabelBytes) || value.Issued == 0 {
		return errors.New("run record: invalid budget grant")
	}
	expires, err := time.Parse(time.RFC3339Nano, value.ExpiresAt)
	if err != nil {
		return errors.New("run record: invalid budget expiry")
	}
	value.ExpiresAt = expires.UTC().Format(time.RFC3339Nano)
	return nil
}
