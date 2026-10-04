package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelartifact"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// runSmoke is validate -smoke: it checks the declared behavior of every
// active inference model, or of the one model names, in the store repository
// names, and records each run there unless diagnostic. budget bounds
// discovery and execution together and must be positive.
func runSmoke(ctx context.Context, output io.Writer, repository string, budget time.Duration, model string, diagnostic bool) error {
	if model != "" {
		id, err := artifact.ParseID(model)
		if err != nil || id.Kind() != artifact.KindModel {
			return errors.New("smoke: -model requires an exact model identity")
		}
	}
	storeRoot, err := dataroot.StoreRoot(repository)
	if err != nil {
		return err
	}
	if budget > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeoutCause(ctx, budget, errors.New("smoke: execution budget exhausted"))
		defer cancel()
	}
	reader, err := overgodb.OpenReadOnly(storeRoot)
	if err != nil {
		return err
	}
	began := time.Now()
	fmt.Fprintln(output, "[smoke] checking active recipes and artifact bytes")
	scope := "catalog"
	var selection []string
	if model != "" {
		scope = "selected"
		selection = append(selection, model)
	}
	entries, err := discovery.ServableWithMemo(ctx, reader, 10_000, nil, selection...)
	fmt.Fprintf(output, "[smoke] discovery=%.1fs models=%d\n", time.Since(began).Seconds(), len(entries))
	if err != nil {
		return errors.Join(err, reader.Close())
	}
	if len(entries) == 0 {
		return errors.Join(runrecord.LaneError(runrecord.LaneEmpty, "servable matrix has no models"), reader.Close())
	}
	if budget <= 0 {
		return errors.Join(errors.New("smoke: explicit positive -budget required"), reader.Close())
	}
	oracles, err := readSmokeOracles(smokeOraclePath, entries, model)
	if err != nil {
		return errors.Join(err, reader.Close())
	}
	references := make(map[artifact.ID][]smokeReferenceCase)
	for _, entry := range entries {
		oracle := oracles[entry.Model]
		domains, declared, domainErr := modelartifact.EvalDomains(ctx, reader, entry.Model)
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
	if !diagnostic {
		commit, err = runrecord.VerifyingCommit(".")
		if err != nil {
			return err
		}
	}
	started, passed := 0, 0
	var failure error
	for _, entry := range entries {
		if model != "" && model != entry.Model.String() {
			continue
		}
		if err := ctx.Err(); err != nil {
			failure = err
			break
		}
		started++
		began := time.Now()
		observation, serveErr := serve(ctx, storeRoot, entry, oracles[entry.Model], references[entry.Model])
		wall := time.Since(began)
		if !diagnostic {
			serveErr = errors.Join(serveErr, recordSmokeTransaction(context.WithoutCancel(ctx), storeRoot, entry, oracles[entry.Model], observation, serveErr, wall, commit))
		}
		fmt.Fprintf(output, "[smoke] %s wall=%.1fs error=%v\n", entry.Model, wall.Seconds(), serveErr)
		if serveErr != nil {
			failure = serveErr
			break
		}
		passed++
	}
	fmt.Fprintf(output, "audit: inference scope=%s discovered=%d started=%d passed=%d failed=%d not_started=%d diagnostic=%t; no evidence for models outside scope; no benchmark or modality promotion\n", scope, len(entries), started, passed, started-passed, len(entries)-started, diagnostic)
	if failure != nil || started == 0 {
		return runrecord.LaneError(runrecord.LaneFailed, fmt.Sprintf("smoke incomplete: %v", failure))
	}
	return nil
}

func serve(ctx context.Context, store string, entry discovery.Entry, oracle smokeOracle, reference []smokeReferenceCase) (smokeObservation, error) {
	runner, err := modelcli.OpenRunner(ctx, store, entry.Location, inference.OpenOptions{})
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
