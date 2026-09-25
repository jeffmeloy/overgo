package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataset"
)

// Split partition names: the objective trains on one and is judged on the other.
const (
	trainPartition   = "train"
	heldoutPartition = "heldout"
)

// splitGroup is the unit a split keeps whole: an asset (all of one asset's
// records land in one partition, so related records never straddle it) or a
// record (each record assigned alone, which a single-asset dataset needs).
// Either way the membership lists every record by the identity the
// materializer reads it under.
type splitGroup string

const (
	splitByAsset  splitGroup = "asset"
	splitByRecord splitGroup = "record"
)

// splitWeights are the declared train and held-out partition weights.
type splitWeights struct{ train, heldout uint64 }

// parseSplitWeights reads train=<w>,heldout=<w>; both are required and
// positive, so the ratio is declared, never defaulted.
func parseSplitWeights(text string) (splitWeights, error) {
	var weights splitWeights
	seen := map[string]bool{}
	for field := range strings.SplitSeq(text, ",") {
		name, value, found := strings.Cut(strings.TrimSpace(field), "=")
		weight, err := strconv.ParseUint(value, 10, 64)
		if !found || err != nil || weight == 0 || seen[name] {
			return splitWeights{}, fmt.Errorf("training-objective: -split-weights %q: want train=<w>,heldout=<w> with positive weights", text)
		}
		seen[name] = true
		switch name {
		case trainPartition:
			weights.train = weight
		case heldoutPartition:
			weights.heldout = weight
		default:
			return splitWeights{}, fmt.Errorf("training-objective: -split-weights names unknown partition %q", name)
		}
	}
	if weights.train == 0 || weights.heldout == 0 {
		return splitWeights{}, fmt.Errorf("training-objective: -split-weights %q: both train and heldout are required", text)
	}
	return weights, nil
}

// splitDataset partitions the registered dataset's records into a training
// and a held-out membership. The seed derives from the dataset identity, so
// one dataset always splits the same way.
func splitDataset(ctx context.Context, reader artifact.Reader, datasetID artifact.ID, group splitGroup, weights splitWeights) (dataset.SplitPlan, dataset.Membership, error) {
	document, found, err := dataset.Load(ctx, reader, datasetID)
	if err != nil {
		return dataset.SplitPlan{}, dataset.Membership{}, err
	}
	if !found || len(document.Assets) == 0 {
		return dataset.SplitPlan{}, dataset.Membership{}, fmt.Errorf("training-objective: dataset %s holds no assets to split", datasetID)
	}
	if group != splitByAsset && group != splitByRecord {
		return dataset.SplitPlan{}, dataset.Membership{}, fmt.Errorf("training-objective: -split-group %q: want asset or record", group)
	}
	var records []dataset.Record
	for _, asset := range document.Assets {
		for ordinal := range asset.Records {
			id := dataset.RecordID(datasetID, asset.Name, ordinal)
			unit := id
			if group == splitByAsset {
				unit = asset.Name
			}
			records = append(records, dataset.Record{ID: id, Group: unit})
		}
	}
	digest := sha256.Sum256([]byte(datasetID.String()))
	plan, err := dataset.BuildGroupSplit(datasetID, records, binary.BigEndian.Uint64(digest[:]), []dataset.SplitPartition{
		{Name: trainPartition, Weight: weights.train}, {Name: heldoutPartition, Weight: weights.heldout},
	})
	if err != nil {
		return dataset.SplitPlan{}, dataset.Membership{}, err
	}
	var training dataset.Membership
	for _, membership := range plan.Memberships {
		if len(membership.Records) == 0 {
			return dataset.SplitPlan{}, dataset.Membership{}, fmt.Errorf(
				"training-objective: the %s split of %d %s groups leaves %q empty; declare finer groups or other weights",
				datasetID, len(records), group, membership.Partition)
		}
		if membership.Partition == trainPartition {
			training = membership
		}
	}
	if training.ID == (artifact.ID{}) {
		return dataset.SplitPlan{}, dataset.Membership{}, errors.New("training-objective: split produced no train membership")
	}
	return plan, training, nil
}
