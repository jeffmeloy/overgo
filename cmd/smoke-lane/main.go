// smoke-lane checks declared behavior for every active inference model.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

func main() { clioptions.MainNamed("smoke-lane", func() error { return run(os.Args[1:]) }) }

func run(args []string) error {
	flags := flag.NewFlagSet("smoke-lane", flag.ContinueOnError)
	budget := flags.Duration("budget", 0, "total discovery and execution budget (required)")
	diagnostic := flags.Bool("diagnostic", false, "inspect behavior without publishing evidence")
	model := flags.String("model", "", "exact model identity; omitted models receive no evidence credit")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("smoke: unexpected arguments")
	}
	if *model != "" {
		id, err := artifact.ParseID(*model)
		if err != nil || id.Kind() != artifact.KindModel {
			return errors.New("smoke: -model requires an exact model identity")
		}
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if *budget > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeoutCause(ctx, *budget, errors.New("smoke: execution budget exhausted"))
		defer cancel()
	}
	reader, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		return err
	}
	began := time.Now()
	fmt.Println("[smoke] checking active recipes and artifact bytes")
	entries, err := discovery.Servable(ctx, reader, 10_000)
	fmt.Printf("[smoke] discovery=%.1fs models=%d\n", time.Since(began).Seconds(), len(entries))
	if err != nil {
		return errors.Join(err, reader.Close())
	}
	if len(entries) == 0 {
		return errors.Join(runrecord.LaneError(runrecord.LaneEmpty, "servable matrix has no models"), reader.Close())
	}
	if *budget <= 0 {
		return errors.Join(errors.New("smoke: explicit positive -budget required"), reader.Close())
	}
	oracles, err := readSmokeOracles(smokeOraclePath, entries)
	if err != nil {
		return errors.Join(err, reader.Close())
	}
	references := make(map[artifact.ID][]smokeReferenceCase)
	for _, entry := range entries {
		oracle := oracles[entry.Model]
		domains, declared, domainErr := evaluation.EvalDomains(ctx, reader, entry.Model)
		if domainErr != nil || (!declared && oracle.Domain != evaluation.DomainText) || (declared && (len(domains) != 1 || domains[0] != oracle.Domain)) {
			return errors.Join(fmt.Errorf("smoke: domain binding differs for %s: %v", entry.Model, domainErr), reader.Close())
		}
		if oracle.ReferenceFile != "" {
			references[entry.Model], err = readSmokeReference(oracle)
			if err != nil {
				return errors.Join(err, reader.Close())
			}
		}
	}
	if err := reader.Close(); err != nil {
		return err
	}
	commit := ""
	if !*diagnostic {
		commit, err = runrecord.VerifyingCommit(".")
		if err != nil {
			return err
		}
	}
	started, passed := 0, 0
	var failure error
	for _, entry := range entries {
		if *model != "" && *model != entry.Model.String() {
			continue
		}
		if err := ctx.Err(); err != nil {
			failure = err
			break
		}
		started++
		began := time.Now()
		observation, serveErr := serve(ctx, roots.Store, entry, oracles[entry.Model], references[entry.Model])
		wall := time.Since(began)
		if !*diagnostic {
			serveErr = errors.Join(serveErr, recordSmokeTransaction(context.WithoutCancel(ctx), roots.Store, entry, oracles[entry.Model], observation, serveErr, wall, commit))
		}
		fmt.Printf("[smoke] %s wall=%.1fs error=%v\n", entry.Model, wall.Seconds(), serveErr)
		if serveErr != nil {
			failure = serveErr
			break
		}
		passed++
	}
	fmt.Printf("audit: inference catalog=%d started=%d passed=%d failed=%d not_started=%d diagnostic=%t; no benchmark or modality promotion\n", len(entries), started, passed, started-passed, len(entries)-started, *diagnostic)
	if failure != nil || started == 0 {
		return runrecord.LaneError(runrecord.LaneFailed, fmt.Sprintf("smoke incomplete: %v", failure))
	}
	return nil
}

func serve(ctx context.Context, store string, entry discovery.Entry, oracle smokeOracle, reference []smokeReferenceCase) (smokeObservation, error) {
	runner, err := clioptions.OpenRunner(ctx, store, entry.Location, inference.OpenOptions{})
	if err != nil {
		return smokeObservation{}, err
	}
	description, err := runner.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil || description.Identity.Model != entry.Model || description.Identity.Recipe != entry.Recipe {
		return smokeObservation{}, errors.Join(fmt.Errorf("smoke: runtime activation differs: %v", err), runner.Close())
	}
	observation, err := executeSmoke(ctx, runner, oracle, reference)
	return observation, errors.Join(err, runner.Close())
}

// Publish after runner close; the writer never spans model execution.
func recordSmokeTransaction(ctx context.Context, path string, entry discovery.Entry, oracle smokeOracle, observation smokeObservation, serveErr error, wall time.Duration, commit string) error {
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		return err
	}
	environmentContent, err := environment.Content()
	if err != nil {
		return err
	}
	oracleData, err := json.Marshal(oracle)
	if err != nil {
		return err
	}
	oracleID, err := artifact.IdentifyBytes(artifact.KindProfile, oracleData)
	if err != nil {
		return err
	}
	observation.Oracle = oracleID
	outcome, failure := runrecord.OutcomeSucceeded, ""
	if serveErr != nil {
		outcome, failure = runrecord.OutcomeFailed, "smoke_failed"
	}
	resultData, err := json.Marshal(struct {
		Model       artifact.ID       `json:"model"`
		Observation smokeObservation  `json:"observation"`
		Outcome     runrecord.Outcome `json:"outcome"`
		WallNS      int64             `json:"wall_ns"`
	}{entry.Model, observation, outcome, wall.Nanoseconds()})
	if err != nil {
		return err
	}
	resultID, err := artifact.IdentifyBytes(artifact.KindEvidence, resultData)
	if err != nil {
		return err
	}
	contents := []artifact.Content{environmentContent,
		{Descriptor: artifact.Descriptor{ID: oracleID, Size: uint64(len(oracleData))}, Data: oracleData},
		{Descriptor: artifact.Descriptor{ID: resultID, Size: uint64(len(resultData))}, Data: resultData},
	}
	inputs := []artifact.ID{entry.Model, oracleID}
	if oracle.Reference.Valid() {
		data, err := os.ReadFile(oracle.ReferenceFile)
		if err != nil {
			return err
		}
		id, err := artifact.IdentifyBytes(artifact.KindEvidence, data)
		if err != nil || id != oracle.Reference {
			return errors.New("smoke: native reference changed during execution")
		}
		contents = append(contents, artifact.Content{Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(data))}, Data: data})
		inputs = append(inputs, id)
	}
	run, err := runrecord.NewBoundRun(entry.Recipe, outcome, inputs, []artifact.ID{resultID}, failure, commit, environment.ID, uint64(wall.Nanoseconds()), nil)
	if err != nil {
		return err
	}
	runContent, err := run.Content()
	if err != nil {
		return err
	}
	contents = append(contents, runContent)
	lineage := run.Lineage()
	if outcome == runrecord.OutcomeSucceeded {
		workload, err := artifact.IdentifyBytes(artifact.KindDataset, oracleData)
		if err != nil {
			return err
		}
		measurement, err := runrecord.NewEvaluation(entry.Recipe, run.ID, workload, []runrecord.Metric{{Name: "smoke_wall_ns", Value: float64(wall.Nanoseconds()), Unit: "ns", Direction: runrecord.DirectionMinimize}})
		if err != nil {
			return err
		}
		content, err := measurement.Content()
		if err != nil {
			return err
		}
		contents = append(contents, artifact.Content{Descriptor: artifact.Descriptor{ID: workload, Size: uint64(len(oracleData))}, Data: oracleData}, content)
		lineage = append(lineage, measurement.Lineage()...)
	}
	batch, err := artifact.NewDocumentBatch("smoke/"+run.ID.String(), contents, lineage, nil)
	if err != nil {
		return err
	}
	store, err := overgodb.OpenContext(ctx, path)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, store, batch)
	return errors.Join(err, store.Close())
}
