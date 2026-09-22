package gate

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/automationcheck"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

// PackageReceiptReport is the gate's standing verdict for one package at the
// working tree: whether a receipt exists for exactly these inputs in this
// environment, and what it says. The obligation is content-addressed, so a
// receipt from an earlier landing still answers for a tree whose inputs to the
// package have not changed since.
type PackageReceiptReport struct {
	Package     string            `json:"package"`
	Mode        string            `json:"mode"`
	Input       artifact.ID       `json:"input"`
	Environment artifact.ID       `json:"environment"`
	Obligation  artifact.ID       `json:"obligation"`
	Found       bool              `json:"found"`
	Passed      bool              `json:"passed"`
	Tests       map[string]string `json:"tests,omitempty"`
}

// Package test modes a receipt is keyed by.
const (
	receiptModeShort    = "short"
	receiptModeComplete = "complete"
)

// receiptModeSeparator splits a package from its mode in -package-receipt.
const receiptModeSeparator = "@"

// PackageReceipt reads, without running anything, the gate's receipt for a
// package at the current tree. It derives the package's input identity and the
// environment exactly as a landing does, so the obligation it looks up is the
// one a landing would have written. A tool that must know whether a suite's
// verdict is current -- the validation planner asking whether the media
// acceptances hold at this tree -- asks here instead of re-running tests.
func PackageReceipt(ctx context.Context, repo, storePath, packagePath, mode string) (PackageReceiptReport, error) {
	if mode != receiptModeShort && mode != receiptModeComplete {
		return PackageReceiptReport{}, fmt.Errorf("gate: package receipt mode must be %s or %s", receiptModeShort, receiptModeComplete)
	}
	graph, err := loadPackageInputGraph(repo)
	if err != nil {
		return PackageReceiptReport{}, err
	}
	// A repository path names the package as the planner spells it; the graph
	// knows import paths and ./ relative directories.
	pkg, known := graph.canonicalPackage(packagePath)
	if !known {
		pkg, known = graph.canonicalPackage("./" + strings.TrimPrefix(packagePath, "./"))
	}
	if !known {
		return PackageReceiptReport{}, fmt.Errorf("gate: package receipt: %q is not a package of this module", packagePath)
	}
	inputs, err := packageInputIdentities(graph, []string{pkg})
	if err != nil {
		return PackageReceiptReport{}, err
	}
	environment, err := discoverEnvironment(repo)
	if err != nil {
		return PackageReceiptReport{}, err
	}
	invocation, err := automationcheck.PackageInvocation(pkg, mode)
	if err != nil {
		return PackageReceiptReport{}, err
	}
	obligation, err := runrecord.NewAgentObligation(runrecord.AgentObligation{
		Task: invocation, Name: "go-test", Scope: mode + ":" + pkg,
		Sources: []artifact.ID{inputs[pkg], environment.ID},
	})
	if err != nil {
		return PackageReceiptReport{}, err
	}
	report := PackageReceiptReport{Package: pkg, Mode: mode, Input: inputs[pkg], Environment: environment.ID, Obligation: obligation.ID}
	store, err := overgodb.OpenReadOnly(filepath.Join(repo, storePath))
	if err != nil {
		return PackageReceiptReport{}, fmt.Errorf("gate: open store for package receipt: %w", err)
	}
	defer store.Close()
	receipt, found, err := packageReceiptCodec.Resolve(ctx, store, packageReceiptAlias+obligation.ID.String())
	if err != nil {
		return PackageReceiptReport{}, err
	}
	if found {
		report.Found, report.Passed, report.Tests = true, receipt.Passed, receipt.Tests
	}
	return report, nil
}

// printPackageReceipt answers -package-receipt PACKAGE[@MODE] as JSON.
func printPackageReceipt(repo, storePath, spec string) error {
	packagePath, mode, given := strings.Cut(spec, receiptModeSeparator)
	if !given {
		mode = receiptModeComplete
	}
	if strings.TrimSpace(packagePath) == "" {
		return errors.New("gate: -package-receipt needs a package path")
	}
	report, err := PackageReceipt(context.Background(), repo, storePath, packagePath, mode)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(report)
}
