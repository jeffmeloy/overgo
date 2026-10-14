package evaluation

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
	"overgo/internal/runrecord"
)

const (
	poolConfirmationMedia  = "application/vnd.overgo.heldout-confirmation+json"
	poolConfirmationSchema = "overgo/heldout-confirmation/v1"

	// The answers a confirmation has, which poolVerdicts counts.
	poolGain      = "gain"
	poolLoss      = "loss"
	poolUndecided = "not enough evidence"

	// poolKey prefixes the key of everything a pool publishes, before its
	// acceptance's identity.
	poolKey = "training/heldout-pool/"
	// poolConfirmer names the judge of an objective's confirmations, the
	// consumer their charges are made for. It is not the proposer, which the
	// promotion part is blinded from.
	poolConfirmer = "overgo/change-confirmer/"
)

// partDocuments names the documents of one part of a pool, in its
// membership's order.
func partDocuments(ctx context.Context, reader artifact.Reader, shard artifact.ID) ([]string, error) {
	membership, found, err := dataset.LoadMembership(ctx, reader, shard)
	if err != nil || !found {
		return nil, errors.Join(fmt.Errorf("evaluation: the pool's part %s is not committed", shard), err)
	}
	documents := make([]string, len(membership.Records))
	for index, record := range membership.Records {
		documents[index] = record.ID
	}
	return documents, nil
}

// PoolConfirmation is one confirmation as a pool records it: the acceptance
// it was judged by, the candidate, the charge it cost and the verdict, a gain,
// a loss or not enough evidence. The statistic the verdict was read from is
// recorded nowhere: the acceptance divides its false-gain rate over what a
// proposer told verdicts could reach, and one that could read the statistic
// would be told more.
type PoolConfirmation struct {
	ID         artifact.ID `json:"-"`
	Version    uint16      `json:"version"`
	Acceptance artifact.ID `json:"acceptance"`
	Candidate  artifact.ID `json:"candidate"`
	Charge     artifact.ID `json:"charge"`
	Verdict    string      `json:"verdict"`
}

var poolConfirmationCodec = artifact.JSONDocumentCodec(
	"held-out confirmation", artifact.KindEvaluation, poolConfirmationMedia, poolConfirmationSchema,
	func(value *PoolConfirmation) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Acceptance.Kind() != artifact.KindEvidence ||
			!value.Candidate.Valid() || value.Charge.Kind() != artifact.KindEvidence ||
			!slices.Contains([]string{poolGain, poolLoss, poolUndecided}, value.Verdict) {
			return errors.New("evaluation: invalid held-out confirmation")
		}
		return nil
	},
	func(value PoolConfirmation) artifact.ID { return value.ID },
	func(value *PoolConfirmation, id artifact.ID) { value.ID = id }, nil,
)

// chargeAttempts bounds how often a confirmation reads again because the
// store moved under it.
const chargeAttempts = 8

// errStoreMoved says the store was committed to between a confirmation's
// first reading and its charge: what it read no longer decides, and nothing
// was charged.
var errStoreMoved = errors.New("evaluation: the store moved under the confirmation")

// oneDecision makes a confirmation's readings and its charge one decision.
// confirm reads everything that says whether a charge is owed under the head
// it is given, and its charge is committed only if that is still the store's
// head (confirmOnPart). Whatever was committed in between, another
// candidate's charge on the same grant or a part registered over the same
// documents, was read in part or not at all; the confirmation then charges
// nothing and reads again from the start, so what landed is counted before a
// charge is made. A store that is still moving after chargeAttempts readings
// refuses the confirmation, which has charged and read nothing.
func oneDecision(store artifact.Repository, confirm func(head artifact.CommitID) (PoolConfirmation, uint64, error)) (PoolConfirmation, uint64, error) {
	for range chargeAttempts {
		head, _ := store.Head()
		if confirmation, remaining, err := confirm(head); !errors.Is(err, errStoreMoved) {
			return confirmation, remaining, err
		}
	}
	return PoolConfirmation{}, 0, fmt.Errorf("%w during each of %d readings of what its documents have answered; nothing was charged or read, and it can be asked again", errStoreMoved, chargeAttempts)
}

// confirmOnPart charges one confirmation to the promotion part of a partition
// and publishes its verdict: the part of ConfirmOnPool that is the same for
// every acceptance a part is confirmed under. The acceptance grants issued
// confirmations on a part it fixed with documents documents; owner is whose
// confirmations they are, and purpose what this one is for, which names its
// charge. judge is handed the part's documents once the charge is committed
// and returns the verdict, and is not called for a confirmation recorded
// before or one the part's documents cannot pay for. head is the store's head
// from before the caller's first reading: the charge is committed against it
// and against no later one, and a store that has moved gives errStoreMoved
// with nothing charged (oneDecision).
func confirmOnPart(ctx context.Context, store artifact.Repository, head artifact.CommitID, partition runrecord.SplitPartition, acceptance, owner, candidate artifact.ID, documents, issued uint64, purpose string, judge func(promotion []string) (string, error)) (PoolConfirmation, uint64, error) {
	membership, found, err := dataset.LoadMembership(ctx, store, partition.Promotion)
	if err != nil || !found || uint64(len(membership.Records)) != documents {
		return PoolConfirmation{}, 0, errors.Join(fmt.Errorf("evaluation: the promotion part of pool %s holds %d documents, committed %t, and its acceptance was fixed for %d", acceptance, len(membership.Records), found, documents), err)
	}
	promotion := make([]string, len(membership.Records))
	for index, record := range membership.Records {
		promotion[index] = record.ID
	}
	// The grant is the one the acceptance published: named by what it grants, and loaded with its charges.
	granted, err := runrecord.NewBudget(poolUnit, partition.Promotion, issued, acceptance)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	budget, charges, err := runrecord.BudgetCharges(ctx, store, granted.ID)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	confirmer, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(poolConfirmer+owner.String()))
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	charge, err := runrecord.NewBudgetCharge(budget.ID, 1, confirmer, purpose)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	remaining, err := runrecord.BudgetBalance(budget, charges)
	if err == nil {
		err = runrecord.ValidateBlinding(partition, budget, charges)
	}
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	// What other grants have charged for on these documents is spent here too.
	earlier, charged, scope, err := exposure(ctx, store, partition, budget.ID, membership)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	remaining -= min(remaining, earlier)
	if slices.ContainsFunc(charges, func(committed runrecord.BudgetCharge) bool { return committed.ID == charge.ID }) {
		verdicts, err := store.Children(ctx, charge.ID)
		if err != nil {
			return PoolConfirmation{}, 0, err
		}
		for _, edge := range verdicts {
			if recorded, err := poolConfirmationCodec.Require(ctx, store, edge.Child); err == nil {
				return recorded, remaining, nil
			}
		}
	} else {
		if remaining == 0 {
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: the promotion part of pool %s has answered the %d confirmations its %d documents allow, %d of them under other acceptances; a further one needs a fresh partition",
				acceptance, issued, documents, earlier)
		}
		// A part published before its contents were named cannot be found from
		// another registration of its documents, so it is charged nothing more.
		if published, err := partContents(ctx, store, partition.Promotion); err != nil {
			return PoolConfirmation{}, 0, err
		} else if len(published) == 0 {
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: the promotion part of pool %s was published without the names of what it holds, so what its documents answer could not be found where they are registered again; nothing was charged or read, and a fresh acceptance names them", acceptance)
		}
		// The key counts the charge among all on the parts of the datasets
		// that share documents with this one, under whichever partition and
		// acceptance. The store holds one content under a key, so of two
		// confirmations that would each be the next, on one part or on two
		// that share documents, of one dataset or of two, one is charged and
		// the other is refused by the store and reads nothing.
		batch, err := charge.Batch(fmt.Sprintf("%s%s/confirmation/%d", poolKey, scope, charged+1))
		if err != nil {
			return PoolConfirmation{}, 0, err
		}
		if _, held, err := store.Artifact(ctx, confirmer); err != nil {
			return PoolConfirmation{}, 0, err
		} else if !held {
			batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: confirmer})
		}
		batch.ExpectedHead = &head
		if _, err := artifact.Publish(ctx, store, batch); errors.Is(err, artifact.ErrCommitPrecondition) {
			return PoolConfirmation{}, 0, errStoreMoved
		} else if err != nil {
			return PoolConfirmation{}, 0, fmt.Errorf("evaluation: the confirmation was not charged, so nothing was read: %w", err)
		}
		remaining--
	}
	verdict, err := judge(promotion)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	confirmation, err := poolConfirmationCodec.NewInitial(PoolConfirmation{Acceptance: acceptance, Candidate: candidate, Charge: charge.ID, Verdict: verdict})
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	batch, err := poolConfirmationCodec.Batch(poolKey+acceptance.String()+"/verdict/"+charge.ID.String(), confirmation,
		artifact.DependencyLineage(confirmation.ID, acceptance, charge.ID), nil)
	if err != nil {
		return PoolConfirmation{}, 0, err
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return PoolConfirmation{}, 0, err
	}
	return confirmation, remaining, nil
}

// exposure counts the confirmations other grants have charged for on the
// documents of a pool's promotion part. Such a grant is another acceptance's
// on the part itself, or one on the promotion or audit part of another
// partition of the pool's dataset, that
// holds one of the part's documents by record or by content. So an acceptance
// replaced, in its gain, its basis, its window, its base or its partition, is
// charged what its documents have answered, and only documents no grant has
// charged for start a budget whole. A charge stays with the grant it was made
// against.
//
// A part of the dataset's own partitions holds one of the documents when it
// holds its record, a record of its group, which is the unit a part is
// divided by, or a record of its content. A part of any dataset holds one
// when both were published as containing its content (containContents), so
// bytes registered again as another dataset are found. A dataset that records
// no content for its records, as a benchmark's import does not, is followed
// by what the part itself was published as containing.
//
// Datasets whose held-out parts share a content are numbered together: scope
// names them, the least of the datasets reached from this one through shared
// contents, and charged is every confirmation charged on a promotion or audit
// part of any of them, this grant's among them. Both are the same from
// whichever of those datasets they are read, so two confirmations that would
// each be the next on parts that share documents, under one dataset or under
// two, ask the store for one key.
func exposure(ctx context.Context, reader artifact.Reader, partition runrecord.SplitPartition, own artifact.ID, promotion dataset.Membership) (earlier, charged uint64, scope artifact.ID, err error) {
	contents, err := recordContents(ctx, reader, partition.Dataset)
	if err != nil {
		return 0, 0, scope, err
	}
	// held are the documents by record, group and content; names are their contents as every dataset names them.
	held, names := map[string]bool{}, map[artifact.ID]bool{}
	for _, record := range promotion.Records {
		held[record.ID], held[record.Group], held[contents[record.ID]] = true, true, true
		if contents[record.ID] == "" {
			continue
		}
		named, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(poolContent+contents[record.ID]))
		if err != nil {
			return 0, 0, scope, err
		}
		names[named] = true
	}
	delete(held, "")
	if promotion.ID.Valid() {
		published, err := partContents(ctx, reader, promotion.ID)
		if err != nil {
			return 0, 0, scope, err
		}
		for _, named := range published {
			names[named] = true
		}
	}
	// numbered are the parts whose charges number the next one, and
	// overlapping those of them that hold a document of this part.
	datasets, numbered, overlapping := []artifact.ID{partition.Dataset}, map[artifact.ID]bool{}, map[artifact.ID]bool{}
	for at := 0; at < len(datasets); at++ {
		partitions, err := reader.Children(ctx, datasets[at])
		if err != nil {
			return 0, 0, scope, err
		}
		for _, edge := range partitions {
			if descriptor, _, err := reader.Artifact(ctx, edge.Child); err != nil {
				return 0, 0, scope, err
			} else if descriptor.MediaType != runrecord.SplitPartitionMediaType {
				continue
			}
			content, _, err := artifact.ReadContent(ctx, reader, edge.Child)
			if err != nil {
				return 0, 0, scope, err
			}
			other, err := runrecord.ParseSplitPartition(content.Data)
			if err != nil {
				return 0, 0, scope, err
			}
			for _, shard := range []artifact.ID{other.Promotion, other.Audit} {
				membership, found, err := dataset.LoadMembership(ctx, reader, shard)
				if err != nil || !found {
					return 0, 0, scope, errors.Join(fmt.Errorf("evaluation: the pool's part %s is not committed", shard), err)
				}
				numbered[shard] = true
				if datasets[at] == partition.Dataset && slices.ContainsFunc(membership.Records, func(record dataset.Record) bool {
					return held[record.ID] || held[record.Group] || held[contents[record.ID]]
				}) {
					overlapping[shard] = true
				}
				published, err := partContents(ctx, reader, shard)
				if err != nil {
					return 0, 0, scope, err
				}
				for _, named := range published {
					if names[named] {
						overlapping[shard] = true
					}
					// Whoever else holds the content brings its dataset into the count.
					holders, err := reader.Parents(ctx, named)
					if err != nil {
						return 0, 0, scope, err
					}
					for _, holder := range holders {
						if holder.Relation != artifact.RelationContains || numbered[holder.Parent] {
							continue
						}
						part, found, err := dataset.LoadMembership(ctx, reader, holder.Parent)
						if err != nil || !found {
							return 0, 0, scope, errors.Join(fmt.Errorf("evaluation: the part %s that holds a document of the pool is not committed", holder.Parent), err)
						}
						if !slices.Contains(datasets, part.Source) {
							datasets = append(datasets, part.Source)
						}
					}
				}
			}
		}
	}
	for shard := range numbered {
		grants, err := reader.Children(ctx, shard)
		if err != nil {
			return 0, 0, scope, err
		}
		for _, grant := range grants {
			if descriptor, _, err := reader.Artifact(ctx, grant.Child); err != nil {
				return 0, 0, scope, err
			} else if descriptor.MediaType != runrecord.BudgetMediaType {
				continue
			}
			budget, charges, err := runrecord.BudgetCharges(ctx, reader, grant.Child)
			if err != nil {
				return 0, 0, scope, err
			}
			for _, charge := range charges {
				if budget.Unit != poolUnit {
					continue
				}
				charged += charge.Amount
				if overlapping[shard] && grant.Child != own {
					earlier += charge.Amount
				}
			}
		}
	}
	return earlier, charged, slices.MinFunc(datasets, func(left, right artifact.ID) int { return cmp.Compare(left.String(), right.String()) }), nil
}

// partContents are the names a part was published as containing: its
// documents' contents, as every dataset names them.
func partContents(ctx context.Context, reader artifact.Reader, shard artifact.ID) ([]artifact.ID, error) {
	edges, err := reader.Children(ctx, shard)
	var names []artifact.ID
	for _, edge := range edges {
		if edge.Relation == artifact.RelationContains {
			names = append(names, edge.Child)
		}
	}
	return names, err
}
