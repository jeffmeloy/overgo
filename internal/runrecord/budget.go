package runrecord

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

const (
	SplitPartitionMediaType = "application/vnd.overgo.split-partition+json"
	SplitPartitionSchema    = "overgo/split-partition/v1"
	BudgetMediaType         = "application/vnd.overgo.budget+json"
	BudgetSchema            = "overgo/budget/v1"
	BudgetChargeMediaType   = "application/vnd.overgo.budget-charge+json"
	BudgetChargeSchema      = "overgo/budget-charge/v1"
)

// SplitRole names the four isolated evaluation datasets. Isolation is the
// control: immutable data still leaks through repeated queries, so promotion
// and audit results are blinded from the proposer and every query is charged.
type SplitRole string

// Only the blinded roles need names today; development and selection are
// identified by their partition fields and gain constants with their first
// consumer.
const (
	SplitPromotion SplitRole = "promotion"
	SplitAudit     SplitRole = "audit"
)

// SplitPartition binds one dataset to four DISTINCT shards, one per role, and
// names the proposer the blinding rule applies to.
type SplitPartition struct {
	Version     uint16      `json:"version"`
	Dataset     artifact.ID `json:"dataset"`
	Development artifact.ID `json:"development"`
	Selection   artifact.ID `json:"selection"`
	Promotion   artifact.ID `json:"promotion"`
	Audit       artifact.ID `json:"audit"`
	Proposer    artifact.ID `json:"proposer"`
	ID          artifact.ID `json:"-"`
}

// Budget is an immutable query-budget grant for one split. Consumption is
// derived from committed charges; there is no mutable counter to drift.
type Budget struct {
	Version   uint16      `json:"version"`
	Unit      string      `json:"unit"`
	Split     artifact.ID `json:"split"`
	Issued    uint64      `json:"issued"`
	Authority artifact.ID `json:"authority"`
	ID        artifact.ID `json:"-"`
}

// BudgetCharge consumes part of a grant for one named consumer and purpose.
type BudgetCharge struct {
	Version  uint16      `json:"version"`
	Budget   artifact.ID `json:"budget"`
	Amount   uint64      `json:"amount"`
	Consumer artifact.ID `json:"consumer"`
	Purpose  string      `json:"purpose"`
	ID       artifact.ID `json:"-"`
}

var splitPartitionCodec = artifact.JSONDocumentCodec(
	"split partition", artifact.KindEvidence, SplitPartitionMediaType, SplitPartitionSchema,
	canonicalizeSplitPartition,
	func(value SplitPartition) artifact.ID { return value.ID },
	func(value *SplitPartition, id artifact.ID) { value.ID = id }, nil,
)

var budgetCodec = artifact.JSONDocumentCodec(
	"budget", artifact.KindEvidence, BudgetMediaType, BudgetSchema,
	canonicalizeBudget,
	func(value Budget) artifact.ID { return value.ID },
	func(value *Budget, id artifact.ID) { value.ID = id }, nil,
)

var budgetChargeCodec = artifact.JSONDocumentCodec(
	"budget charge", artifact.KindEvidence, BudgetChargeMediaType, BudgetChargeSchema,
	canonicalizeBudgetCharge,
	func(value BudgetCharge) artifact.ID { return value.ID },
	func(value *BudgetCharge, id artifact.ID) { value.ID = id }, nil,
)

func ParseSplitPartition(data []byte) (SplitPartition, error) {
	return splitPartitionCodec.Parse(data)
}

func ParseBudget(data []byte) (Budget, error) {
	return budgetCodec.Parse(data)
}

func ParseBudgetCharge(data []byte) (BudgetCharge, error) {
	return budgetChargeCodec.Parse(data)
}

func NewBudget(unit string, split artifact.ID, issued uint64, authority artifact.ID) (Budget, error) {
	return budgetCodec.New(Budget{
		Version: artifact.InitialDocumentVersion, Unit: unit, Split: split, Issued: issued, Authority: authority,
	})
}

func NewBudgetCharge(budget artifact.ID, amount uint64, consumer artifact.ID, purpose string) (BudgetCharge, error) {
	return budgetChargeCodec.New(BudgetCharge{
		Version: artifact.InitialDocumentVersion, Budget: budget, Amount: amount, Consumer: consumer, Purpose: purpose,
	})
}

func (value Budget) Content() (artifact.Content, error) { return budgetCodec.Content(value) }

func (value BudgetCharge) Content() (artifact.Content, error) {
	return budgetChargeCodec.Content(value)
}

func (value Budget) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Split, value.Authority)
}

func (value BudgetCharge) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Budget, value.Consumer)
}

func (value Budget) Batch(key string) (artifact.Batch, error) {
	return budgetCodec.Batch(key, value, value.Lineage(), nil)
}

func (value BudgetCharge) Batch(key string) (artifact.Batch, error) {
	return budgetChargeCodec.Batch(key, value, value.Lineage(), nil)
}

// BudgetBalance derives the remaining grant from committed charges. Charges
// against other budgets are refused rather than skipped, and consumption
// beyond the grant is an error naming the overrun: the balance can prove
// exhaustion but can never go silently negative.
func BudgetBalance(budget Budget, charges []BudgetCharge) (uint64, error) {
	if err := budget.ValidateIdentity(); err != nil {
		return 0, err
	}
	consumed := uint64(0)
	for _, charge := range charges {
		if err := charge.ValidateIdentity(); err != nil {
			return 0, err
		}
		if charge.Budget != budget.ID {
			return 0, fmt.Errorf("run record: charge %s belongs to budget %s, not %s", charge.ID, charge.Budget, budget.ID)
		}
		if charge.Amount > budget.Issued-consumed {
			return 0, fmt.Errorf("run record: budget %s exhausted: issued %d, next charge %d exceeds remaining %d",
				budget.ID, budget.Issued, charge.Amount, budget.Issued-consumed)
		}
		consumed += charge.Amount
	}
	return budget.Issued - consumed, nil
}

func (value SplitPartition) ValidateIdentity() error {
	return splitPartitionCodec.ValidateIdentity(value)
}

func (value Budget) ValidateIdentity() error { return budgetCodec.ValidateIdentity(value) }

func (value BudgetCharge) ValidateIdentity() error { return budgetChargeCodec.ValidateIdentity(value) }

// ValidateBlinding enforces the blinding rule at the only place observations
// are recorded: the charge ledger. A query against a split is an observation
// of its results, so a charge by the partition's proposer against the
// promotion or audit split is refused by name. Budgets on splits this
// partition does not govern validate vacuously.
func ValidateBlinding(partition SplitPartition, budget Budget, charges []BudgetCharge) error {
	if err := partition.ValidateIdentity(); err != nil {
		return err
	}
	var role SplitRole
	switch budget.Split {
	case partition.Promotion:
		role = SplitPromotion
	case partition.Audit:
		role = SplitAudit
	default:
		return nil
	}
	for _, charge := range charges {
		if charge.Consumer == partition.Proposer {
			return fmt.Errorf("run record: proposer %s is blinded from %s results (charge %s)",
				partition.Proposer, role, charge.ID)
		}
	}
	return nil
}

func canonicalizeSplitPartition(value *SplitPartition) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Dataset.Kind() != artifact.KindDataset ||
		value.Proposer.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid split partition envelope")
	}
	splits := []artifact.ID{value.Development, value.Selection, value.Promotion, value.Audit}
	for _, split := range splits {
		if split.Kind() != artifact.KindDatasetShard {
			return errors.New("run record: split partition role is not a dataset shard")
		}
	}
	if !distinctIDs(splits...) {
		return errors.New("run record: split partition roles must be distinct shards")
	}
	return nil
}

func canonicalizeBudget(value *Budget) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Unit == "" ||
		value.Split.Kind() != artifact.KindDatasetShard || value.Issued == 0 ||
		value.Authority.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid budget grant")
	}
	return nil
}

func canonicalizeBudgetCharge(value *BudgetCharge) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || !value.Budget.Valid() ||
		value.Amount == 0 || value.Consumer.Kind() != artifact.KindEvidence || value.Purpose == "" {
		return errors.New("run record: invalid budget charge")
	}
	return nil
}
