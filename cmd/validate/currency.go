package main

import (
	"context"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/inferencesurface"
	"overgo/internal/overgodb"
)

// acceptedRecords is what the one current-evidence decision of the plan, -run
// and the inventory reads, once per invocation and only as the cells need it:
// the inference surface now with every location's newest long-form record,
// one acceptance receipt per package, and each DNA model's smoke-run verdict.
type acceptedRecords struct {
	surface  string
	records  map[string]guardRecord
	receipts map[string]testReport
	smoke    map[artifact.ID]error
}

// readAcceptedRecords reads what the cells' owners judge by. A read that
// fails fails the command: an unread record is neither current nor a reason
// to spend a device re-acquiring it.
func readAcceptedRecords(ctx context.Context, root, storeRoot string, store *overgodb.Store, cells []ModelValidation) (acceptedRecords, error) {
	read := acceptedRecords{records: map[string]guardRecord{}, receipts: map[string]testReport{}, smoke: map[artifact.ID]error{}}
	for _, cell := range cells {
		pkg := surfacePackages[cell.Surface]
		_, received := read.receipts[pkg]
		_, smoked := read.smoke[cell.Model]
		var err error
		switch {
		case dnaGuard(cell) && !smoked:
			run, found, err := latestSmokeRun(ctx, store, cell.Model, cell.Recipe)
			if err != nil {
				return acceptedRecords{}, err
			}
			read.smoke[cell.Model] = smokeCurrent(ctx, root, cell.Model, run, found)
		case cell.Validation == guardValidation && !dnaGuard(cell) && read.surface == "":
			if read.surface, err = inferencesurface.Digest(ctx, root); err == nil {
				read.records, err = latestGuardRecords(ctx, root, storeRoot)
			}
		case pkg != "" && !received:
			read.receipts[pkg], err = packageTests(ctx, root, pkg)
		}
		if err != nil {
			return acceptedRecords{}, err
		}
	}
	return read, nil
}

// current is the one decision on whether a cell's accepted evidence holds at
// this tree. The activation's recipe only finds the record; the record's own
// binding decides: a text guard record is current when it measured the
// inference surface digest the code is at now and met its floors, a DNA
// model's smoke run when it succeeded and no inference surface source moved
// since its commit, and media, speech and specialized evidence when its
// acceptance package's receipt holds the media runtime identity at this tree.
// It returns nil for current evidence, errCellNotJudged for a cell no owner
// judges, and otherwise why the evidence is not current.
func (read acceptedRecords) current(cell ModelValidation) error {
	switch {
	case dnaGuard(cell):
		if refusal, found := read.smoke[cell.Model]; found {
			return refusal
		}
		return fmt.Errorf("no smoke run was read for this model; "+smokeRerun, cell.Model)
	case cell.Validation == guardValidation:
		return guardCurrent(read.records, cell.Location, read.surface)
	}
	return mediaCurrent(read.receipts[surfacePackages[cell.Surface]], cell.Surface)
}
