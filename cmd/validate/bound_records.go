package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// A benchmark or proof cell's accepted record is a run bound to the commit it
// ran at. Its producer refuses to record from a dirty tree, so that commit is
// exactly the code it measured. A benchmark cell's record is its recipe's
// newest evaluation of its family's suite as the active catalog compiles it
// now: a suite imported again since is another dataset, which no older record
// measured. A proof cell's record is the verified run its activation accepted.
// The record is current when its run succeeded and no change since its commit
// moves the cell's surface, by the rule a commit range moves one: the
// evaluation surface holds the inference and scoring code a benchmark ran
// through, the training surface the trainers a proof ran.

// benchmarkFamilies are the derived-suite families an inference activation
// owes an accuracy benchmark on.
var benchmarkFamilies = []string{"mmlu-pro", "bbh", "ifeval"}

// templateFamilies score the answer a model generates to an instruction, which
// means something only through the model's own chat template. A first
// measurement of one runs under it; one of any other family takes the
// raw-completion anchor, the only protocol the catalog names BBH under.
var templateFamilies = []string{"ifeval"}

// boundSurfaces are the surfaces whose cells' records are runs bound to the
// commit they ran at.
var boundSurfaces = []SurfaceID{"evaluation", "training"}

// errCellInapplicable marks a cell its model's own declaration excludes.
var errCellInapplicable = errors.New("the model's declared evaluation domains admit no suite of this benchmark")

// boundKey names one commit-bound cell: a recipe's validation.
type boundKey struct {
	recipe     artifact.ID
	validation string
}

// boundRecord is what a commit-bound cell is judged by: its record's run, and
// the protocol and cost its re-acquisition takes. refusal says why no record
// can stand for the cell where one would be looked for.
type boundRecord struct {
	run      runrecord.Run
	found    bool
	protocol string
	cost     int64
	refusal  error
}

// benchmarkEvidence is the active catalog compiled without binding a model,
// as the scorecard compiles it, and the store's newest evaluation of each
// recipe and suite under each protocol.
type benchmarkEvidence struct {
	suites   []evaluation.CompiledSuite
	evidence evaluation.EvidenceIndex
}

// readBenchmarkEvidence compiles the active catalog and indexes the
// evaluations of its suites. A catalog that compiles no suite indexes none,
// and each benchmark cell says so; its command meets the same refusal.
func readBenchmarkEvidence(ctx context.Context, store *overgodb.Store) benchmarkEvidence {
	suites, _, _ := evaluation.DeriveStoreSuites(ctx, store, evaluation.ListingAuthorities())
	names := map[artifact.ID]string{}
	for _, suite := range suites {
		if descriptor := suite.Descriptor(); descriptor.Dataset.Valid() {
			names[descriptor.Dataset] = descriptor.Source
		}
	}
	return benchmarkEvidence{suites: suites, evidence: evaluation.LatestEvidence(ctx, store, names, 0)}
}

// record finds a benchmark cell's record: its recipe's newest evaluation of
// the family's suite, under whichever protocol ran last, which is the one the
// scorecard reads. A family no record measures yet takes its first protocol.
// The cell is inapplicable when its model declares evaluation domains that
// admit no suite of the family, as evaluate's own routing finds.
func (benchmarks benchmarkEvidence) record(ctx context.Context, store *overgodb.Store, cell ModelValidation) (boundRecord, error) {
	family := func(source string) bool { return strings.HasSuffix(source, "/"+cell.Validation) }
	compiles := func(suite evaluation.CompiledSuite) bool { return family(suite.Descriptor().Source) }
	switch domains := strings.Split(cell.Modality, ","); {
	case !slices.ContainsFunc(benchmarks.suites, compiles):
		return boundRecord{refusal: fmt.Errorf("the active catalog compiles no %s suite", cell.Validation)}, nil
	case !slices.ContainsFunc(evaluation.FilterSuitesForDomains(benchmarks.suites, domains, cell.Modality != ""), compiles):
		return boundRecord{refusal: fmt.Errorf("%w: it declares %s", errCellInapplicable, cell.Modality)}, nil
	}
	for _, summary := range benchmarks.evidence.EvaluationsByRecipe[cell.Recipe] {
		if !family(summary.Suite) {
			continue
		}
		evaluated, err := runrecord.RequireEvaluation(ctx, store, summary.Record)
		if err != nil {
			return boundRecord{}, err
		}
		run, err := runrecord.RequireRun(ctx, store, evaluated.Run)
		return boundRecord{run: run, found: true, protocol: summary.Prompting, cost: int64(run.MeasuredNS)}, err
	}
	protocol := evaluation.PromptingRawCompletion.Label()
	if slices.Contains(templateFamilies, cell.Validation) {
		protocol = evaluation.PromptingChatTemplate.Label()
	}
	return boundRecord{protocol: protocol, cost: int64(benchmarkBudget)}, nil
}

// proofRecord finds a proof cell's record: the verified run its activation
// accepted.
func proofRecord(ctx context.Context, store *overgodb.Store, cell ModelValidation) (boundRecord, error) {
	activation, found, err := modelrecipe.ActiveRecord(ctx, store, cell.Model, cell.Kind)
	if err != nil || !found {
		return boundRecord{}, err
	}
	verified, err := runrecord.VerifyEvidence(ctx, store, activation.Definition.ID, activation.Event.Evidence)
	return boundRecord{run: verified.Run, found: err == nil, cost: int64(verified.Run.MeasuredNS)}, err
}

// recordCurrent says why a commit-bound cell's record is not current at this
// tree: none can stand for the cell, there is none, its run did not succeed,
// its commit cannot be compared with the tree, or a change since its commit
// moves the cell's surface, whose package directories are directories. Nil
// means the record stands for the code as it is now.
func recordCurrent(ctx context.Context, root string, directories []string, cell ModelValidation, record boundRecord) error {
	run := record.run
	switch {
	case record.refusal != nil:
		return record.refusal
	case !record.found:
		return fmt.Errorf("no %s record for this recipe yet", cell.Validation)
	case run.Outcome != runrecord.OutcomeSucceeded:
		return fmt.Errorf("its %s run at commit %.*s did not succeed (%s)", cell.Validation, surfaceShown, run.CodeCommit, run.Failure)
	}
	changed, err := changedSince(ctx, root, run.CodeCommit)
	if err != nil {
		return fmt.Errorf("its %s record is bound to commit %.*s, which this tree cannot be compared with: %w", cell.Validation, surfaceShown, run.CodeCommit, err)
	}
	if moved := surfaceMoves(directories, changed); moved.Affected {
		return fmt.Errorf("its %s record ran at commit %.*s and the %s surface moved since: %s", cell.Validation, surfaceShown, run.CodeCommit, cell.Surface, moved.Reason)
	}
	return nil
}

// judgeBound judges one commit-bound cell and returns it with the protocol and
// cost its re-acquisition takes. It lists the cell's surface packages and
// reads the catalog's evaluations the first time a cell needs them.
func (read *acceptedRecords) judgeBound(ctx context.Context, root string, store *overgodb.Store, cell ModelValidation) (ModelValidation, error) {
	if _, listed := read.directories[cell.Surface]; !listed {
		directories, err := surfaceDirectories(ctx, root, modalitySurfaces[cell.Surface].roots)
		if err != nil {
			return cell, err
		}
		read.directories[cell.Surface] = directories
	}
	var record boundRecord
	var err error
	if slices.Contains(benchmarkFamilies, cell.Validation) {
		if read.benchmarks == nil {
			benchmarks := readBenchmarkEvidence(ctx, store)
			read.benchmarks = &benchmarks
		}
		record, err = read.benchmarks.record(ctx, store, cell)
	} else {
		record, err = proofRecord(ctx, store, cell)
	}
	if err != nil {
		return cell, err
	}
	cell.Protocol, cell.Cost = record.protocol, record.cost
	read.bound[boundKey{cell.Recipe, cell.Validation}] = recordCurrent(ctx, root, read.directories[cell.Surface], cell, record)
	return cell, nil
}
