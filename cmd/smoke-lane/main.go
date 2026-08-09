// smoke-lane: the store-derived model smoke matrix (Automation Doctrine
// Layer 3 target, now built). The matrix is not a file: it is the servable
// predicate — every model with an active inference recipe and present bytes
// gets one short serve, and every serve records a run + wall-time evaluation
// to the store against the model's own recipe. Absent models are
// UNAVAILABLE, listed, never green.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
)

const smokePrompt = "The capital of France is"

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "smoke-lane: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	working, err := os.Getwd()
	if err != nil {
		return err
	}
	roots, err := dataroot.Resolve(working)
	if err != nil {
		return err
	}
	// Discovery reads through a read-only open; each record is its own short
	// writer transaction so serves never contend with the lane's lock.
	ctx := context.Background()
	reader, err := repodb.OpenReadOnly(roots.Store)
	if err != nil {
		return err
	}
	entries, err := discovery.Servable(ctx, reader, 10_000)
	closeErr := reader.Close()
	if err != nil || closeErr != nil {
		return errorsJoin(err, closeErr)
	}
	if len(entries) == 0 {
		fmt.Println("smoke-lane: 0 servable models; nothing to smoke (honest empty, not green)")
		return nil
	}
	unavailable, failed, passed := 0, 0, 0
	for _, entry := range entries {
		if !entry.Present {
			unavailable++
			fmt.Printf("[smoke] %s UNAVAILABLE (recorded location missing: %s)\n", entry.Model, entry.Location)
			continue
		}
		began := time.Now()
		serveErr := serve(entry.Location)
		wall := time.Since(began)
		outcome := runrecord.OutcomeSucceeded
		failure := ""
		if serveErr != nil {
			outcome = runrecord.OutcomeFailed
			failure = serveErr.Error()
			failed++
		} else {
			passed++
		}
		if err := recordSmokeTransaction(ctx, roots.Store, entry, outcome, failure, wall); err != nil {
			return err
		}
		fmt.Printf("[smoke] %s %-9s %6.1fs tier=%s\n", entry.Model, outcome, wall.Seconds(), entry.Tier)
		if serveErr != nil {
			fmt.Printf("        %s\n", tail(failure, 400))
		}
	}
	fmt.Printf("=== SMOKE %d passed / %d failed / %d unavailable of %d servable ===\n",
		passed, failed, unavailable, len(entries))
	fmt.Println("honesty: matrix derived from the servable predicate; runs recorded to the store per model recipe")
	if failed > 0 || unavailable > 0 {
		return fmt.Errorf("%d failed, %d unavailable", failed, unavailable)
	}
	return nil
}

// serve runs one short generation through the recipe-authorized path. A
// subprocess keeps the lane honest: it exercises exactly what a user runs.
func serve(location string) error {
	cmd := exec.Command("go", "run", "./cmd/generate", "-n", "4", location, smokePrompt)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%v: %s", err, tail(string(out), 800))
	}
	var payload struct {
		Text string `json:"text"`
	}
	trimmed := string(out)
	if index := strings.Index(trimmed, "{"); index >= 0 {
		trimmed = trimmed[index:]
	}
	if json.Unmarshal([]byte(trimmed), &payload) != nil || strings.TrimSpace(payload.Text) == "" {
		return fmt.Errorf("serve produced no text")
	}
	return nil
}

func errorsJoin(errs ...error) error {
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// recordSmokeTransaction opens a short-lived writer, commits, and closes:
// the writer lock is held only for the record, never across a serve.
func recordSmokeTransaction(
	ctx context.Context,
	storePath string,
	entry discovery.Entry,
	outcome runrecord.Outcome,
	failure string,
	wall time.Duration,
) error {
	store, err := repodb.Open(storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	return recordSmoke(ctx, store, entry, outcome, failure, wall)
}

func recordSmoke(
	ctx context.Context,
	store *repodb.Store,
	entry discovery.Entry,
	outcome runrecord.Outcome,
	failure string,
	wall time.Duration,
) error {
	environmentDoc, err := json.Marshal(map[string]string{
		"os": runtime.GOOS, "arch": runtime.GOARCH, "go": runtime.Version(), "host": hostname(),
	})
	if err != nil {
		return err
	}
	environmentID, err := artifact.IdentifyBytes(artifact.KindEvidence, environmentDoc)
	if err != nil {
		return err
	}
	codeCommit := "unknown"
	if out, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
		codeCommit = strings.TrimSpace(string(out))
	}
	// Run records want structured facts: failure is a CODE (detail already
	// printed), and a successful run's output is the smoke-result document.
	failureCode := ""
	if failure != "" {
		failureCode = "serve_failed"
	}
	resultDoc, err := json.Marshal(map[string]string{
		"model": entry.Model.String(), "prompt": smokePrompt,
		"wall_ns": fmt.Sprintf("%d", wall.Nanoseconds()), "outcome": string(outcome),
	})
	if err != nil {
		return err
	}
	resultID, err := artifact.IdentifyBytes(artifact.KindEvidence, resultDoc)
	if err != nil {
		return err
	}
	var outputs []artifact.ID
	if outcome == runrecord.OutcomeSucceeded {
		outputs = []artifact.ID{resultID}
	}
	run, err := runrecord.NewBoundRun(
		entry.Recipe, outcome, []artifact.ID{entry.Model}, outputs, failureCode, codeCommit,
		environmentID, uint64(wall.Nanoseconds()), nil,
	)
	if err != nil {
		return err
	}
	runContent, err := run.Content()
	if err != nil {
		return err
	}
	batch := artifact.Batch{
		Key:       "smoke/" + entry.Model.String() + "/" + codeCommit,
		Artifacts: []artifact.Descriptor{{ID: environmentID}},
		Contents: []artifact.Content{runContent, {
			Descriptor: artifact.Descriptor{ID: resultID, Size: uint64(len(resultDoc))},
			Data:       resultDoc,
		}},
		Lineage: run.Lineage(),
	}
	if outcome == runrecord.OutcomeSucceeded {
		workloadID, err := artifact.IdentifyBytes(artifact.KindDataset, []byte("overgo-smoke-workload/v1"))
		if err != nil {
			return err
		}
		evaluation, err := runrecord.NewEvaluation(entry.Recipe, run.ID, workloadID, []runrecord.Metric{{
			Name: "smoke_wall_ns", Value: float64(wall.Nanoseconds()),
			Unit: "ns", Direction: runrecord.DirectionMinimize,
		}})
		if err != nil {
			return err
		}
		evaluationContent, err := evaluation.Content()
		if err != nil {
			return err
		}
		batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: workloadID})
		batch.Contents = append(batch.Contents, evaluationContent)
		batch.Lineage = append(batch.Lineage, evaluation.Lineage()...)
	}
	_, err = store.Commit(ctx, batch)
	if err != nil && strings.Contains(err.Error(), "batch key already names") {
		return nil // same tree already smoked this model
	}
	return err
}

func hostname() string {
	name, err := os.Hostname()
	if err != nil {
		return "unknown"
	}
	return name
}

func tail(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return "..." + s[len(s)-limit:]
}
