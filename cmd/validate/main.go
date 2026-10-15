// validate runs the end-to-end validation of the whole system for one commit:
// it accounts for every registered model, derives the appropriate validation
// for each declared function, and re-acquires every activated applicable cell,
// a benchmark asking its model only what its code has not answered, then runs the improvement
// cycle's lanes, the real-model suites no landing runs. The default dry run
// computes and prints the plan without loading a model or a
// device; -run executes the selected cells and lanes. -claims holds the
// claims compatibility.json and the training specifications make to the
// sources they bind and checks the matrices published from them; a claim whose
// source moved is rebound only after -claims -run re-runs its verify. It also
// renders the model and media reports from the store on demand, exports the
// media samples, publishes, checks and accepts the capability census, and
// commits model-verification records. It also runs the smoke oracles against
// the active inference models, the reference parity comparisons, the hermetic
// short test lane and the test receipts reviewed media deltas bind.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/gitauthority"
	"overgo/internal/inferencesurface"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/scorecard"
	"overgo/internal/testevidence"
	"overgo/internal/trainingworkflow"
)

// catalogLimit bounds the registered-model listing; a truncated denominator is
// refused rather than read as complete.
const catalogLimit = 4096

func main() { clioptions.MainNamed("validate", run) }

// run ends the work in progress, rather than the process, on an interrupt,
// so a child test run or a model being served is stopped by its owner.
func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	return runArgs(ctx, os.Args[1:], os.Stdout)
}

func runArgs(ctx context.Context, args []string, output io.Writer) error {
	flags := flag.NewFlagSet("validate", flag.ContinueOnError)
	root := flags.String("root", ".", "repository root: the grounding files are read from it, and the commands run in it")
	repository := flags.String("repo", "", "OvergoDB store; empty resolves through the data-root contract")
	execute := flags.Bool("run", false, "execute the selected validations the scorecard waits on and the cycle lanes (default: dry run prints the plan only)")
	background := flags.Bool("background", false, "with -run: execute the background validations instead, the suites the scorecard does not compare, which the scorecard never waits on")
	cellsOnly := flags.Bool("cells", false, "with -run: execute only the cells the scorecard waits on, leaving the cycle lanes to -run -lanes, so a cycle can publish its scorecard between them")
	lanesOnly := flags.Bool("lanes", false, "with -run: execute only the cycle lanes, which go test's cache replays for the packages no change reached")
	inventory := flags.String("inventory", "", "comma-separated model roots: report each directory beneath them as activated, registered, unregistered or not a model, with the commands that would advance it; loads no model")
	onboardPath := flags.String("onboard", "", "a model directory or GGUF: convert a Hugging Face directory, register, record and replay an exact suite, activate, and run the long-form guard, in one command")
	scorecardView := flags.Bool("scorecard", false, "print the user scorecard: for every activated model and task, time to first output, throughput, held-out quality, failures, load time, peak device memory and setup steps, each read from the store or listed as a gap with its reason, and the share of served models supported by declaration alone; runs nothing")
	publish := flags.Bool("publish", false, "with -scorecard: write it as the newest record in the scorecards directory beside the store and list what moved since the previous record")
	gap := flags.String("gap", "", "with -scorecard: fail while the gap this key names is open on the scorecard the store holds, compared with the measurements published before it; the verifier of the row the loop proposed for it")
	target := flags.Float64("target", math.NaN(), "with -gap, for the key of a decline: the value its cell is to be restored to, which the row carries from the cycle that found the decline")
	claims := flags.Bool("claims", false, "check compatibility.json's claims and the training specifications against their bound sources and the matrices published from them; with -run, re-run the verify of every claim whose bound source moved, rebind the claims that pass, and write the matrices again")
	models := flags.Bool("models", false, "print the nested model compatibility report: the docs/verification typed specifications joined against the store")
	media := flags.String("media", "", "print the media capability report rendered from store activations and the verification specifications: markdown, or json for its typed projection; its links resolve from docs/")
	exportSamples := flags.Bool("export-samples", false, "export healthy media activations' generation outputs as decodable files under docs/media_samples")
	mediaScope := flags.String("media-scope", "all", "with -export-samples: the tasks exported, all or image-video; the media report always renders every media task")
	publishCensus := flags.Bool("publish-census", false, "publish the complete registered capability denominator from clean code")
	checkCensus := flags.String("check-census", "", "check one exact immutable capability census against the live registered denominator")
	var censusBindings []string
	flags.Func("check-census-binding", "check a binding's exact accepted census checkpoint and disposition without changing it or checking a new denominator; repeat for multiple binding files", func(path string) error {
		censusBindings = append(censusBindings, path)
		return nil
	})
	acceptPath := flags.String("accept-census", "", "a census binding file, cmd/validate/testdata/accepted_census.json or media_census.json: print what the registered models add to, remove from or change in the census it accepts; with -reason, from clean code, publish the live census with a finding closed for that reason and bind the file to them")
	reason := flags.String("reason", "", "with -accept-census: why the changed census is accepted; without it nothing is written")
	recordVerification := flags.String("record-verification", "", "commit a typed model-verification record from a JSON spec (model, name, evidenced capability claims) into the store")
	smoke := flags.Bool("smoke", false, "serve every active inference model, or the one -model names, against its declared oracle in docs/verification/smoke-oracles.json and record a bound run into the store")
	budget := flags.Duration("budget", 0, "with -smoke: the total discovery and execution budget; required")
	smokeModel := flags.String("model", "", "with -smoke: the exact model identity to serve; omitted models receive no evidence credit")
	diagnostic := flags.Bool("diagnostic", false, "with -smoke: inspect behavior without recording a run")
	parityRun := flags.Bool("parity", false, "run the reference comparison the first argument names, with the flags after it: audio-oracle, sensenova or vqa")
	testLane := flags.Bool("tests", false, "run the hermetic short test lane over the packages after the flags (default ./...): a failed, unfinished or unclassified skipped test refuses it")
	receipt := flags.Bool("receipt", false, "run go test -json, one package at a time, with the go test arguments after the flags, and keep the passing stream in the store as a receipt")
	command := clioptions.Command{
		Name:     "validate",
		Purpose:  "select the minimal sufficient end-to-end validation for a commit and, with -run, execute it",
		Audience: "the master-lead session and its unattended driver validating the whole system per commit",
		Constraints: []string{
			"the default dry run itself loads no model and opens no device: it reads the registered models and the newest benchmark evaluations and prints the plan as JSON",
			"every activated cell the model's declaration does not exclude runs every cycle, whatever it recorded before, but a training proof of a kind of objective declared as the operator's to run, which -run names with its reason and does not run; a benchmark runs under the protocol of its newest record, and evaluate puts to the model only the cases the store holds no answer to by the code that answers now",
			"the scorecard waits on every guard, serving cell, proof and cycle lane, and on a benchmark that re-measures the held-out contract the scorecard beside the store compares; any other benchmark is background, and -run -background runs those",
			"every registered model is accounted for; each available model is dispatched by its declared functions",
			"-inventory starts from the directories on disk, so a model nothing has registered is reported; it ties a converted model to its directory by the content of the config files the conversion recorded, never by a name",
			"-scorecard reads the store and the serving source: it loads no model and opens no device",
			"-onboard runs on a clean tree: the activation binds the commit it validated under",
			"-claims reads the checkout only; a claim is rebound only by its own verify passing, never by hand",
			"-models, -media, -check-census and -check-census-binding read the store; -export-samples writes only under docs/media_samples; -publish-census and -record-verification write the store; -accept-census reads it, and with -reason writes it and the binding file",
			"-smoke loads each selected model on the device and, without -diagnostic, records its run at the checked-out commit into the store",
			"-parity, -tests and -receipt take their arguments after the flags; -receipt alone writes the store",
			"help opens no store and reads nothing",
		},
	}
	if handled, err := command.ParseForHelp(flags, args, output); err != nil || handled {
		return err
	}
	switch {
	case *parityRun:
		return runParity(flags.Args(), output)
	case *testLane:
		return runTests(flags.Args(), output, func(packages []string) (testevidence.GoTestReport, error) { return runGoTest(ctx, packages) })
	case *receipt:
		return runReceipt(ctx, output, *repository, flags.Args())
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("validate: unexpected arguments %v", flags.Args())
	}
	if *smoke {
		return runSmoke(ctx, output, *repository, *budget, *smokeModel, *diagnostic)
	}
	if *budget != 0 || *smokeModel != "" || *diagnostic {
		return errors.New("-budget, -model and -diagnostic go with -smoke")
	}
	if *onboardPath != "" {
		steps, err := newOnboarding(*root, *repository, output)
		if err != nil {
			return err
		}
		return steps.onboard(ctx, output, *onboardPath)
	}
	if *scorecardView {
		storeRoot, err := dataroot.StoreRoot(*repository)
		if err != nil {
			return err
		}
		store, err := overgodb.OpenReadOnly(storeRoot)
		if err != nil {
			return err
		}
		defer store.Close()
		records := scorecard.Directory(storeRoot)
		if *gap != "" {
			var restoreTo *float64
			if !math.IsNaN(*target) {
				restoreTo = target
			}
			return openGap(ctx, store, records, *gap, restoreTo, output)
		}
		if !*publish {
			records = ""
		}
		return writeScorecard(ctx, store, *root, records, output)
	}
	if *claims {
		return publishedClaims(ctx, output, *root, *execute)
	}
	if *media != "" && *media != "markdown" && *media != "json" || *mediaScope != "all" && !*exportSamples {
		return errors.New("-media renders markdown or json, and -media-scope narrows -export-samples alone; the media report always renders every media task")
	}
	if *models || *media != "" || *exportSamples || *publishCensus || *checkCensus != "" || len(censusBindings) != 0 || *acceptPath != "" || *recordVerification != "" {
		storeRoot, err := dataroot.StoreRoot(*repository)
		if err != nil {
			return err
		}
		switch {
		case len(censusBindings) != 0:
			store, err := overgodb.OpenReadOnly(storeRoot)
			if err != nil {
				return err
			}
			defer store.Close()
			var failures []error
			for _, path := range censusBindings {
				if err := checkCensusBinding(ctx, store, path); err != nil {
					failures = append(failures, fmt.Errorf("census binding %s requires explicit reacceptance before replacement: %w", path, err))
				}
			}
			return errors.Join(failures...)
		case *publishCensus || *checkCensus != "":
			return runCapabilityCensus(storeRoot, *publishCensus, *checkCensus, output)
		case *acceptPath != "":
			// Only an acceptance publishes a census, which names the commit that
			// built it, so only it needs clean code; a reading of what differs
			// publishes nothing and runs from any checkout.
			resolve := runrecord.HeadCommit
			if *reason != "" {
				resolve = runrecord.ExecutableCodeCommit
			}
			commit, err := resolve(".")
			if err != nil {
				return err
			}
			return acceptCensus(storeRoot, commit, *acceptPath, *reason, output)
		case *recordVerification != "":
			return runRecordVerification(*recordVerification, storeRoot, output)
		case *exportSamples:
			scope, err := resolveMediaReportScope(*mediaScope)
			if err != nil {
				return err
			}
			return exportMediaSamples(*root, storeRoot, scope)
		case *media != "":
			report, err := generateMediaReport(*root, storeRoot)
			if err != nil {
				return err
			}
			rendered := report.Markdown
			if *media == "json" {
				rendered = report.JSON
			}
			_, err = output.Write(rendered)
			return err
		}
		store, err := overgodb.OpenReadOnly(storeRoot)
		if err != nil {
			return err
		}
		report, err := generateModelsReport(*root, store)
		if err == nil {
			_, err = output.Write(report)
		}
		return errors.Join(err, store.Close())
	}

	// One store, resolved once and absolute: the store this command reads is
	// the one every producer it launches at the repository root writes.
	storeRoot, err := dataroot.StoreRoot(*repository)
	if err != nil {
		return err
	}
	if storeRoot, err = filepath.Abs(storeRoot); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		return err
	}
	defer store.Close()
	var scope runScope
	if *execute {
		if scope, err = runScopeOf(*background, *cellsOnly, *lanesOnly); err != nil {
			return err
		}
		if scope == scopeLanes {
			// The lanes need no model resolved: only they run.
			return executeRun(ctx, output, *root, storeRoot, store, Plan{Suites: cycleSuites()}, scope)
		}
	}
	registered, err := registeredCells(ctx, store)
	if err != nil {
		return err
	}
	compared, err := lastRecord(storeRoot)
	if err != nil {
		return err
	}
	registered = compareContracts(registered, compared)
	execution, _ := inferencesurface.Execution("overgo/internal/mediacapability")
	if err := bindCells(ctx, *root, store, registered, execution); err != nil {
		return err
	}
	if *inventory != "" {
		directories, err := modelDirectories(strings.Split(*inventory, ","))
		if err != nil {
			return err
		}
		converted, err := convertedSources(ctx, store)
		if err != nil {
			return err
		}
		return clioptions.WritePrettyJSON(output, buildInventory(directories, registered, converted, storeRoot))
	}

	plan := SelectValidation(Inputs{Registered: registered, Scorecard: compared})
	plan.Suites = cycleSuites()
	if *execute {
		return executeRun(ctx, output, *root, storeRoot, store, plan, scope)
	}
	return clioptions.WritePrettyJSON(output, plan)
}

// registeredCells enumerates every registered model and derives its validation
// obligations from its declared functions, generically: a model with no
// activation, or absent bytes, is one registered-inactive cell; each activated
// function maps to the appropriate validation for its kind. No per-model
// knowledge participates.
func registeredCells(ctx context.Context, store *overgodb.Store) ([]ModelValidation, error) {
	memo := discovery.LoadMemo(ctx, store)
	entries, truncated, err := discovery.RegisteredCatalog(ctx, store, catalogLimit, memo)
	if err != nil {
		return nil, err
	}
	if truncated {
		return nil, fmt.Errorf("validate: registered catalog exceeds listing bound %d; raise the bound", catalogLimit)
	}
	entries = currentIdentities(entries, trainingworkflow.WeightsIdentity)
	var cells []ModelValidation
	for _, entry := range entries {
		name := ""
		if entry.Location != "" {
			name = filepath.Base(entry.Location)
		}
		if !entry.Present || len(entry.Capabilities) == 0 {
			cells = append(cells, ModelValidation{Model: entry.Model, ModelName: name, Location: entry.Location})
			continue
		}
		modality := declaredModality(ctx, store, entry.Model)
		// A capability resolves the live directory recorded for the model,
		// never the weights file, shard or head file the catalog found it by;
		// none recorded leaves its measurement unbound.
		directory, _ := discovery.ExecutionDirectory(ctx, store, entry.Model)
		for _, capability := range entry.Capabilities {
			validations := kindValidation(capability.Task)
			if len(validations) == 0 {
				cells = append(cells, ModelValidation{Model: entry.Model, ModelName: name, Location: entry.Location, Kind: capability.Task, Modality: modality})
				continue
			}
			for _, validation := range validations {
				// The activation's recipe finds a benchmark's newest record.
				cells = append(cells, ModelValidation{
					Model: entry.Model, ModelName: name, Location: entry.Location, Directory: directory,
					Kind: capability.Task, Modality: modality, Recipe: capability.Recipe,
					Validation: validation, Stale: capability.Stale,
				})
			}
		}
	}
	return cells, nil
}

// currentIdentities drops, at a weights location that more than one model
// identity answers to, the training capability of the identities the weights
// no longer resolve to when the identity they do resolve to carries a training
// capability of its own that is not stale. A file registered under the
// whole-file hash an older bootstrap recorded and under the complete-manifest
// identity a training run binds now has a training activation under each, and
// the superseded one's proof can never hand off: its run is recorded under the
// current identity's recipe, so it would run, fail the cycle every
// cycle. Only the training route's own resolution is proven to match the
// identity a proof binds, so no other task is dropped; a location where one
// identity alone holds training, or whose identity cannot be resolved, is left
// as the catalog lists it.
func currentIdentities(entries []discovery.CatalogEntry, resolve func(location string) (artifact.ID, error)) []discovery.CatalogEntry {
	training := func(entry discovery.CatalogEntry) bool {
		return slices.ContainsFunc(entry.Capabilities, func(capability discovery.Capability) bool { return capability.Task == recipe.TaskTraining })
	}
	byLocation := map[string][]int{}
	for index, entry := range entries {
		if entry.Location != "" && training(entry) {
			byLocation[entry.Location] = append(byLocation[entry.Location], index)
		}
	}
	entries = slices.Clone(entries)
	for location, group := range byLocation {
		if len(group) <= 1 {
			continue
		}
		current, err := resolve(location)
		if err != nil {
			continue
		}
		at := slices.IndexFunc(group, func(index int) bool { return entries[index].Model == current })
		if at < 0 || slices.ContainsFunc(entries[group[at]].Capabilities, func(capability discovery.Capability) bool {
			return capability.Task == recipe.TaskTraining && capability.Stale != ""
		}) {
			continue
		}
		for _, index := range group {
			if index != group[at] {
				entries[index].Capabilities = slices.DeleteFunc(slices.Clone(entries[index].Capabilities), func(capability discovery.Capability) bool { return capability.Task == recipe.TaskTraining })
			}
		}
	}
	return entries
}

// kindValidation lists the appropriate validations for one declared task.
// Dispatch is by task, never by model name: an inference model earns a guard, a
// measured serving attempt and the accuracy benchmarks, an image-gen model its
// generation proof, and so on.
func kindValidation(task recipe.Task) []string {
	switch string(task) {
	case "inference":
		return append([]string{guardValidation, servingValidation}, benchmarkFamilies...)
	case "projection":
		return []string{projectionProof}
	case "image-gen":
		return []string{"image-proof"}
	case "video-gen":
		return []string{"video-proof"}
	case "speech", "transcription":
		return []string{"speech-proof"}
	case "vqa":
		return []string{"vqa-proof"}
	case "seq2seq", "forecast", "tabular", "image-embedding", "reconstruction", "generation":
		return []string{"specialized-proof"}
	case "training":
		return []string{trainingProof}
	default:
		return nil
	}
}

// declaredModality joins a model's declared evaluation domains for display; an
// undeclared or unreadable model contributes no modality label.
func declaredModality(ctx context.Context, store *overgodb.Store, model artifact.ID) string {
	domains, declared, err := modelartifact.EvalDomains(ctx, store, model)
	if err != nil || !declared || len(domains) == 0 {
		return ""
	}
	return strings.Join(domains, ",")
}

// revParse resolves a git revision to its object id through the read-only git owner.
func revParse(ctx context.Context, root, revision string) (string, error) {
	out, err := gitauthority.Query(ctx, root, "rev-parse", revision)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
