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
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/clioptions"
	"overgo/internal/closurescan"
	"overgo/internal/jsonfile"
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
	// The agent harness surface against its reviewed baseline: read it here
	// before and after a change instead of from a throwaway program.
	var baseline closurescan.HarnessSurface
	if err := jsonfile.DecodeStrict("docs/harness_surface_baseline.json", &baseline); err != nil {
		return err
	}
	surface, err := closurescan.BuildAgentHarnessSurface(snapshot)
	if err != nil {
		return err
	}
	for _, metric := range closurescan.HarnessSurfaceReport(baseline, surface) {
		fmt.Printf("harness %-24s %d -> %d (%+d)\n", metric.Metric, metric.Base, metric.Value, metric.Value-metric.Base)
	}
	access, err := repoanalysis.StoreAccessCensus(snapshot)
	if err != nil {
		return err
	}
	for _, reach := range access {
		fmt.Printf("store access %-44s total=%-4d %v domains=%v\n", reach.Package, reach.Total, reach.Sites, reach.Domains)
	}
	// The tracked documents: printed, never published -- a census of the
	// documents that should leave the tree does not add one to it.
	tracked, err := repoanalysis.TrackedDocuments(context.Background(), ".")
	if err != nil {
		return err
	}
	documents, err := repoanalysis.TrackedDocumentCensus(snapshot, tracked, repoanalysis.TrackedDocumentFamilies)
	if err != nil {
		return err
	}
	for _, tracked := range documents {
		if kind := tracked.Family.Kind; kind == repoanalysis.DocumentUnread || kind == repoanalysis.DocumentGenerated {
			fmt.Printf("document %s %s writers=%v readers=%v\n", kind, tracked.Path, tracked.Writers, tracked.Readers)
		}
	}
	if err := repoanalysis.ValidateTrackedDocuments(documents, repoanalysis.TrackedDocumentFamilies); err != nil {
		return err
	}
	document := struct {
		Version     int                          `json:"version"`
		Doc         string                       `json:"doc"`
		Findings    []repoanalysis.CensusFinding `json:"findings"`
		StoreAccess []repoanalysis.StoreAccess   `json:"store_access"`
	}{
		Version: 1,
		Doc: "Computed RSI ownership census; regenerate with go run ./cmd/rsi-census -update. " +
			"Owners and recurrence are derived from repository syntax, never hand-listed.",
		Findings: findings, StoreAccess: access,
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
