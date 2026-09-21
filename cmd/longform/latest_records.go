package main

import (
	"context"
	"io"
	"slices"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
)

// latestRecordsVerb lists the newest long-form record of every weights
// location. It is a verb of its own, read before the measurement options:
// those describe a pass over models, and this loads none.
const latestRecordsVerb = "latest-records"

// LatestRecord is the newest long-form record of one weights location: the
// inference surface it measured and whether it met its floors. A reader that
// knows the surface the code is at now can say whether the record is current
// without linking the device stack this command measures with -- which is why
// the validation planner asks here instead of reading the records itself.
type LatestRecord struct {
	Location string   `json:"location"`
	Surface  string   `json:"surface"`
	Commit   string   `json:"commit"`
	Passed   bool     `json:"passed"`
	Reasons  []string `json:"reasons,omitempty"`
}

// latestRecords prints every location's newest record, ordered by location.
func latestRecords(ctx context.Context, repository string, output io.Writer) error {
	root, err := dataroot.StoreRoot(repository)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(root)
	if err != nil {
		return err
	}
	defer store.Close()
	latest := longform.LatestByLocation(ctx, store, 0)
	records := make([]LatestRecord, 0, len(latest))
	for location, summary := range latest {
		records = append(records, LatestRecord{
			Location: location, Surface: summary.Result.Surface, Commit: summary.Result.Commit,
			Passed: summary.Result.Verdict.Passed, Reasons: summary.Result.Verdict.Reasons,
		})
	}
	slices.SortFunc(records, func(left, right LatestRecord) int { return strings.Compare(left.Location, right.Location) })
	return clioptions.WritePrettyJSON(output, records)
}
