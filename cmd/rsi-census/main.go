// rsi-census publishes the RSI ownership census: for each mechanic
// family the storage campaign consolidates, the package that currently
// owns it, every package re-implementing it, and the measured
// recurrence. The census is computed from repository syntax by
// repoanalysis.RSIOwnershipCensus; -update writes the published
// document at docs/rsi_ownership_census.json and -check verifies it is
// current, so consolidation rows cite measured duplication, never
// recollection.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/repoanalysis"
)

const censusPath = "docs/rsi_ownership_census.json"

func main() {
	clioptions.MainNamed("rsi-census", run)
}

func run() error {
	update := flag.Bool("update", false, "write "+censusPath)
	check := flag.Bool("check", false, "verify "+censusPath+" matches the source tree")
	flag.Parse()
	snapshot, err := repoanalysis.DiscoverGo(".", "internal", "cmd")
	if err != nil {
		return err
	}
	findings, err := repoanalysis.RSIOwnershipCensus(snapshot)
	if err != nil {
		return err
	}
	for _, finding := range findings {
		fmt.Printf("%-12s sites=%-4d owner=%-40s owner_sites=%-4d recurrence=%-4d consumers=%d\n",
			finding.Family, finding.Sites, finding.Owner, finding.OwnerSites,
			finding.Recurrence, len(finding.Consumers))
	}
	document := struct {
		Version  int                          `json:"version"`
		Doc      string                       `json:"doc"`
		Findings []repoanalysis.CensusFinding `json:"findings"`
	}{
		Version: 1,
		Doc: "Computed RSI ownership census; regenerate with go run ./cmd/rsi-census -update. " +
			"Owners and recurrence are derived from repository syntax, never hand-listed.",
		Findings: findings,
	}
	encoded, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	if *update {
		if err := os.WriteFile(filepath.FromSlash(censusPath), encoded, 0o644); err != nil {
			return err
		}
		fmt.Printf("rsi-census: wrote %s\n", censusPath)
	}
	if *check {
		published, err := os.ReadFile(filepath.FromSlash(censusPath))
		if err != nil {
			return fmt.Errorf("rsi-census: %s is absent; regenerate with -update", censusPath)
		}
		if !bytes.Equal(published, encoded) {
			return fmt.Errorf("rsi-census: %s is stale; regenerate with -update", censusPath)
		}
		fmt.Println("rsi-census: published census is current")
	}
	return nil
}
