package main

import (
	"context"
	"errors"
	"slices"

	"overgo/internal/inferencesurface"
	"overgo/internal/landing"
	"overgo/internal/overgodb"
)

// cycleSuites lists the improvement cycle's lanes, which run every cycle:
// go test's cache replays the packages no change reached since their pass.
func cycleSuites() []SelectedSuite {
	suites := []SelectedSuite{}
	for _, lane := range landing.CycleLanes() {
		suites = append(suites, SelectedSuite{Lane: lane.Name, Arguments: lane.Arguments, Environment: lane.Environment})
	}
	return suites
}

// errEveryCycle is why a serving cell and a DNA model's smoke run run: each
// is re-acquired every cycle, a serving attempt because it binds no commit
// its currency could be judged by, the smoke run because it costs seconds.
var errEveryCycle = errors.New("re-acquired every cycle")

// acceptedRecords is what the one current-evidence decision of the plan, -run
// and the inventory reads, once per invocation and only as the cells need it:
// the inference surface now with every location's newest long-form record,
// one acceptance receipt per package, and each commit-bound cell's verdict
// with the surface packages and catalog evaluations it was judged by.
type acceptedRecords struct {
	surface     string
	records     map[string]guardRecord
	receipts    map[string]testReport
	bound       map[boundKey]error
	directories map[SurfaceID][]string
	benchmarks  *benchmarkEvidence
}

// readAcceptedRecords reads what the cells' owners judge by. A read that
// fails fails the command: an unread record is neither current nor a reason
// to spend a device re-acquiring it. It also gives each commit-bound cell, in
// place, the protocol, cost and grounding its re-acquisition takes.
func readAcceptedRecords(ctx context.Context, root, storeRoot string, store *overgodb.Store, cells []ModelValidation) (acceptedRecords, error) {
	read := acceptedRecords{
		records: map[string]guardRecord{}, receipts: map[string]testReport{}, bound: map[boundKey]error{}, directories: map[SurfaceID][]string{},
	}
	for index, cell := range cells {
		pkg := surfacePackages[cell.Surface]
		_, received := read.receipts[pkg]
		var err error
		switch {
		case cell.Validation == guardValidation && !dnaGuard(cell) && read.surface == "":
			if read.surface, err = inferencesurface.Digest(ctx, root); err == nil {
				read.records, err = latestGuardRecords(ctx, root, storeRoot)
			}
		case pkg != "" && !received:
			read.receipts[pkg], err = packageTests(ctx, root, pkg)
		case slices.Contains(boundSurfaces, cell.Surface):
			cells[index], err = read.judgeBound(ctx, root, store, cell)
		}
		if err != nil {
			return acceptedRecords{}, err
		}
	}
	if err := groundTraining(ctx, root, store, cells); err != nil {
		return acceptedRecords{}, err
	}
	return read, groundProjection(ctx, root, store, cells)
}

// current is the one decision on whether a cell's accepted evidence holds at
// this tree. The activation's recipe only finds the record; the record's own
// binding decides: a text guard record is current when it measured the
// inference surface digest the code is at now and met its floors, a
// benchmark or proof record when its run succeeded and no change since its
// commit moves the cell's surface, and media, speech and specialized
// evidence when its acceptance package's receipt holds the media runtime
// identity at this tree. A serving cell and a DNA model's smoke run are
// never current. It returns nil for current evidence, errCellNotJudged for a
// cell no owner judges, errCellInapplicable for one its model's declaration
// excludes, and otherwise why the evidence is not current.
func (read acceptedRecords) current(cell ModelValidation) error {
	switch {
	case dnaGuard(cell), cell.Validation == servingValidation:
		return errEveryCycle
	case cell.Validation == guardValidation:
		return guardCurrent(read.records, cell.Location, read.surface)
	case slices.Contains(boundSurfaces, cell.Surface):
		if verdict, found := read.bound[boundKey{cell.Recipe, cell.Validation, cell.Protocol}]; found {
			return verdict
		}
		return errors.New("no record was read for this cell")
	}
	return mediaCurrent(read.receipts[surfacePackages[cell.Surface]], cell.Surface)
}
