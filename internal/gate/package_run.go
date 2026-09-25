package gate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
	"overgo/internal/testevidence"
)

// packageRunStep names the step a standalone package run attributes its
// execution to.
const packageRunStep = "package-run"

// PackageRun runs one package's tests at the clean working tree in the named
// mode and records the gate's receipt for exactly these inputs, so a later
// read of the receipt finds the verdict a landing would have written. A
// landing runs a package complete only when its impact selection picks the
// package as a dependent; evidence that is judged by such a receipt -- the
// media and specialized model acceptances -- is otherwise current only by
// accident of which symbols a change touched. This is the deliberate act.
func PackageRun(ctx context.Context, repo, storePath, spec string, output io.Writer) error {
	packagePath, mode, given := strings.Cut(spec, receiptModeSeparator)
	if !given {
		mode = receiptModeComplete
	}
	if mode != receiptModeShort && mode != receiptModeComplete {
		return fmt.Errorf("gate: package run mode must be %s or %s", receiptModeShort, receiptModeComplete)
	}
	if strings.TrimSpace(packagePath) == "" {
		return errors.New("gate: -package-run needs a package path")
	}
	// The receipt binds the tree's inputs; a dirty tree has no one identity.
	if _, err := runrecord.VerifyingCommit(repo); err != nil {
		return fmt.Errorf("gate: package run: %w", err)
	}
	graph, err := loadPackageInputGraph(repo)
	if err != nil {
		return err
	}
	pkg, known := graph.canonicalPackage(packagePath)
	if !known {
		pkg, known = graph.canonicalPackage("./" + strings.TrimPrefix(packagePath, "./"))
	}
	if !known {
		return fmt.Errorf("gate: package run: %q is not a package of this module", packagePath)
	}
	inputs, err := packageInputIdentities(graph, []string{pkg})
	if err != nil {
		return err
	}
	environment, err := discoverEnvironment(repo)
	if err != nil {
		return err
	}
	g := &gateContext{repo: repo, storePath: storePath, environment: environment, start: time.Now(), clock: processmeasure.NewStopwatch(), packageGraph: &graph}
	defer func() { _ = g.closeStore() }()
	devices, _, err := g.splitDevice([]string{pkg})
	if err != nil {
		return err
	}
	run := func(ctx context.Context, packages []string, short bool, observe func(string, bool) error, leased bool) (testevidence.GoTestReport, error) {
		if len(devices) != 0 {
			return g.runContendedBatch(ctx, packages, short, observe, g.runGoTestsAdmitted)
		}
		return g.runGoTestsAdmitted(ctx, packages, short, observe, leased)
	}
	runErr := g.runPackageReceipt(ctx, pkg, mode, inputs[pkg], run)
	if err := g.closeStore(); err != nil {
		return errors.Join(runErr, err)
	}
	report, err := PackageReceipt(ctx, repo, storePath, pkg, mode)
	if err != nil {
		return errors.Join(runErr, err)
	}
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	return errors.Join(runErr, encoder.Encode(report))
}

// runPackageReceipt prepares the package's obligation through the package
// evidence ledger, runs the package through run, and requires the run to have
// recorded a terminal verdict: a run that ends without one leaves the
// obligation open and is an error, never a silent pass.
func (g *gateContext) runPackageReceipt(ctx context.Context, pkg, mode string, input artifact.ID, run testRunner) error {
	ledger, err := g.openPackageEvidence()
	if err != nil {
		return err
	}
	inputs := map[string]artifact.ID{pkg: input}
	if err := ledger.prepare(ctx, []string{pkg}, mode, inputs, g.retryCache); err != nil {
		return err
	}
	g.setTestStep(packageRunStep)
	before := ledger.previous[pkg]
	observe := g.packagePassObserver(ctx, ledger, mode, inputs)
	_, runErr := run(ctx, []string{pkg}, mode == receiptModeShort, observe, true)
	if ledger.previous[pkg] == before || !ledger.previous[pkg].Valid() {
		return errors.Join(runErr, fmt.Errorf("gate: package run of %s recorded no terminal verdict", pkg))
	}
	return runErr
}
