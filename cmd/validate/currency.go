package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/gitauthority"
	"overgo/internal/inferencesurface"
	"overgo/internal/landing"
	"overgo/internal/overgodb"
	"overgo/internal/scorecard"
)

// cycleSuites decides each of the improvement cycle's lanes at root the way
// a cell is decided: it runs when the commit range's changed paths reach it,
// or when changes since the promoted commit do, since promotion waited on
// its pass there; otherwise that pass still holds and it is reused.
func cycleSuites(ctx context.Context, root, head string, changed []string) ([]SelectedSuite, error) {
	lanes, err := landing.CycleLanes(root)
	if err != nil {
		return nil, err
	}
	held, err := gitauthority.Query(ctx, root, "for-each-ref", "--format=%(objectname)", scorecard.PromotedBranch)
	if err != nil {
		return nil, err
	}
	promoted := strings.TrimSpace(string(held))
	var since []string
	if promoted != "" {
		if since, err = changedPaths(ctx, root, promoted, head); err != nil {
			return nil, err
		}
	}
	suites := []SelectedSuite{}
	for _, lane := range lanes {
		suite := SelectedSuite{Lane: lane.Name, Run: true, Arguments: lane.Arguments, Environment: lane.Environment}
		switch {
		case lane.Reaches(changed):
			suite.Reason = "run: the commit range's changes reach it"
		case promoted == "":
			suite.Reason = "run: no " + scorecard.PromotedBranch + " commit holds its pass yet"
		case lane.Reaches(since):
			suite.Reason = fmt.Sprintf("run: changes since the promoted commit %.*s reach it", surfaceShown, promoted)
		default:
			suite.Run, suite.Reason = false, "reuse: its pass holds at the promoted commit and no change since reaches it"
		}
		suites = append(suites, suite)
	}
	return suites, nil
}

// acceptedRecords is what the one current-evidence decision of the plan, -run
// and the inventory reads, once per invocation and only as the cells need it:
// the inference surface now with every location's newest long-form record,
// one acceptance receipt per package, each DNA model's smoke-run verdict, the
// recipes a serving attempt measured, and each commit-bound cell's verdict
// with the surface packages and catalog evaluations it was judged by.
type acceptedRecords struct {
	surface     string
	records     map[string]guardRecord
	receipts    map[string]testReport
	smoke       map[artifact.ID]error
	serving     map[artifact.ID]bool
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
		records: map[string]guardRecord{}, receipts: map[string]testReport{}, smoke: map[artifact.ID]error{},
		bound: map[boundKey]error{}, directories: map[SurfaceID][]string{},
	}
	for index, cell := range cells {
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
			read.smoke[cell.Model] = smokeCurrent(ctx, root, storeRoot, cell.Model, run, found)
		case cell.Validation == guardValidation && !dnaGuard(cell) && read.surface == "":
			if read.surface, err = inferencesurface.Digest(ctx, root); err == nil {
				read.records, err = latestGuardRecords(ctx, root, storeRoot)
			}
		case cell.Validation == servingValidation && read.serving == nil:
			read.serving, err = measuredRecipes(ctx, store)
		case pkg != "" && !received:
			read.receipts[pkg], err = packageTests(ctx, root, pkg)
		case slices.Contains(boundSurfaces, cell.Surface):
			cells[index], err = read.judgeBound(ctx, root, store, cell)
		}
		if err != nil {
			return acceptedRecords{}, err
		}
	}
	return read, groundTraining(ctx, root, store, cells)
}

// current is the one decision on whether a cell's accepted evidence holds at
// this tree. The activation's recipe only finds the record; the record's own
// binding decides: a text guard record is current when it measured the
// inference surface digest the code is at now and met its floors, a DNA
// model's smoke run when it succeeded and no inference surface source moved
// since its commit, a serving cell when a measured attempt of its recipe timed
// its load, a benchmark or proof record when its run succeeded and no change
// since its commit moves the cell's surface, and media, speech and
// specialized evidence when its acceptance package's receipt holds the media
// runtime identity at this tree. It returns nil for current evidence,
// errCellNotJudged for a cell no owner judges, errCellInapplicable for one its
// model's declaration excludes, and otherwise why the evidence is not current.
func (read acceptedRecords) current(cell ModelValidation) error {
	switch {
	case dnaGuard(cell):
		if refusal, found := read.smoke[cell.Model]; found {
			return refusal
		}
		return errors.New("no smoke run was read for this model")
	case cell.Validation == guardValidation:
		return guardCurrent(read.records, cell.Location, read.surface)
	case cell.Validation == servingValidation:
		if read.serving[cell.Recipe] {
			return nil
		}
		return errUnmeasured
	case slices.Contains(boundSurfaces, cell.Surface):
		if verdict, found := read.bound[boundKey{cell.Recipe, cell.Validation}]; found {
			return verdict
		}
		return errors.New("no record was read for this cell")
	}
	return mediaCurrent(read.receipts[surfacePackages[cell.Surface]], cell.Surface)
}
