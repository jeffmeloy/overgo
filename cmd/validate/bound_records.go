package main

import (
	"cmp"
	"context"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// A benchmark cell is re-acquired every cycle under the protocol of its
// recipe's newest evaluation of its family's suite as the active catalog
// compiles it now (a suite imported again since is another dataset, which no
// older record measured). Evaluate reads each case's answer from the store
// where the code that answers now gave it before, and asks the model the rest.

// benchmarkFamilies are the derived-suite families every inference activation
// owes an accuracy benchmark on, besides the family the scorecard
// compares for it.
var benchmarkFamilies = []string{"mmlu-pro", "bbh", "ifeval"}

// templateFamilies score the answer a model generates to an instruction, which
// means something only through the model's own chat template. A first
// measurement of one runs under it; one of any other family takes the
// raw-completion anchor, the only protocol the catalog names BBH under.
var templateFamilies = []string{"ifeval"}

// benchmarkEvidence is the active catalog compiled without binding a model,
// as the scorecard compiles it, and the store's newest evaluation of each
// recipe and suite under each protocol.
type benchmarkEvidence struct {
	suites   []evaluation.CompiledSuite
	evidence evaluation.EvidenceIndex
}

// readBenchmarkEvidence compiles the active catalog and indexes the
// evaluations of its suites. A catalog that compiles no suite indexes none,
// and each benchmark cell's command meets that refusal.
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

// bind gives a benchmark cell the protocol and cost of its
// recipe's newest evaluation of the family's suite, under the protocol the
// cell names when the scorecard compares it, else under whichever
// protocol ran last. A family no record measures yet takes the named protocol
// or its first, and runs in full. The cell is inapplicable when its model
// declares evaluation domains that admit no suite of the family, as
// evaluate's own routing finds.
func (benchmarks benchmarkEvidence) bind(ctx context.Context, store *overgodb.Store, cell ModelValidation) (ModelValidation, error) {
	compiles := func(suite evaluation.CompiledSuite) bool { return ofFamily(suite.Descriptor().Source, cell.Validation) }
	domains := strings.Split(cell.Modality, ",")
	if slices.ContainsFunc(benchmarks.suites, compiles) && !slices.ContainsFunc(evaluation.FilterSuitesForDomains(benchmarks.suites, domains, cell.Modality != ""), compiles) {
		cell.Inapplicable = "the model's declared evaluation domains admit no suite of this benchmark: it declares " + cell.Modality
		return cell, nil
	}
	for _, summary := range benchmarks.evidence.EvaluationsByRecipe[cell.Recipe] {
		if !summary.ExecutionKnown || !ofFamily(summary.Suite, cell.Validation) || cell.Protocol != "" && summary.Prompting != cell.Protocol {
			continue
		}
		evaluated, err := runrecord.RequireEvaluation(ctx, store, summary.Record)
		if err != nil {
			return cell, err
		}
		run, err := runrecord.RequireRun(ctx, store, evaluated.Run)
		cell.Protocol, cell.Cost = summary.Prompting, int64(run.MeasuredNS)
		return cell, err
	}
	protocol := evaluation.PromptingRawCompletion.Label()
	if slices.Contains(templateFamilies, cell.Validation) {
		protocol = evaluation.PromptingChatTemplate.Label()
	}
	cell.Protocol, cell.Cost = cmp.Or(cell.Protocol, protocol), int64(benchmarkBudget)
	return cell, nil
}

// ofFamily reports whether a suite, named by its source, is of a benchmark
// family: the catalog names a family's suites by it after the source's root.
func ofFamily(source, family string) bool { return strings.HasSuffix(source, "/"+family) }

// benchmark reports whether a cell is an accuracy benchmark: of an inference
// activation's cells, every one but its guard and serving cell.
func benchmark(cell ModelValidation) bool {
	return cell.Kind == recipe.TaskInference && cell.Validation != guardValidation && cell.Validation != servingValidation
}

// proofCost is the wall of the verified run a proof cell's activation
// accepted, which orders the proofs smallest-first; zero when the active
// recipe is not the selected one or none is accepted.
func proofCost(ctx context.Context, store *overgodb.Store, cell ModelValidation) (int64, error) {
	activation, found, err := modelrecipe.ActiveRecord(ctx, store, cell.Model, cell.Kind)
	if err != nil || !found || activation.Definition.ID != cell.Recipe {
		return 0, err
	}
	return int64(activation.Verified.Run.MeasuredNS), nil
}

// bindCells gives each cell, in place, what its re-acquisition takes: a
// benchmark the protocol and cost of its newest evaluation, or
// its exclusion by its model's declared domains; a training or projection
// proof the wall of the run its activation accepted, then the grounding the
// training and projection routes take, then what the files its command is given
// are (consumedFiles). A read of a record that fails fails the command:
// an unread record is no reason to spend a device re-acquiring blind.
func bindCells(ctx context.Context, root string, store *overgodb.Store, cells []ModelValidation, execution string) error {
	var benchmarks *benchmarkEvidence
	for index, cell := range cells {
		var err error
		switch {
		case benchmark(cell):
			if benchmarks == nil {
				read := readBenchmarkEvidence(ctx, store)
				benchmarks = &read
			}
			cells[index], err = benchmarks.bind(ctx, store, cell)
		case cell.Validation == trainingProof || cell.Validation == projectionProof:
			cells[index].Cost, err = proofCost(ctx, store, cell)
		}
		if err != nil {
			return err
		}
	}
	if err := groundTraining(ctx, root, store, cells); err != nil {
		return err
	}
	if err := groundProjection(ctx, root, store, cells); err != nil {
		return err
	}
	// Last, since the grounding decides which paths a cell's command is given.
	listed := map[string]string{}
	for index, cell := range cells {
		cells[index].Consumed = consumedFiles(ctx, store, cell, listed, execution)
	}
	return nil
}
