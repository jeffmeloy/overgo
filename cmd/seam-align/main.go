// Command seam-align computes the seam alignment residual — the cheap
// causal proxy for whether a linear interface adapter can compose a seam —
// and bias-audits recorded residuals against measured bridge parity.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/composition"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
)

func main() { clioptions.MainNamed("seam-align", run) }

func run() error {
	flags := flag.NewFlagSet("seam-align", flag.ContinueOnError)
	activations := flags.String("activations", "", "paired seam activations JSON path ({source, target})")
	audit := flags.String("audit", "", "recorded measurements JSON path ({residuals, parities})")
	enumerate := flags.String("enumerate", "", "target component name: enumerate ranked donor seam candidates from the committed catalog")
	repoFlag := flags.String("repo", "", "OvergoDB store for -enumerate")
	measured := flags.String("measurements", "", "donor seam measurements JSON path for -enumerate ({measurements})")
	limit := flags.Int("limit", 16, "shortlist size for -enumerate")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if strings.TrimSpace(*enumerate) != "" {
		return enumerateCandidates(*repoFlag, *enumerate, *measured, *limit)
	}
	hasActivations, hasAudit := strings.TrimSpace(*activations) != "", strings.TrimSpace(*audit) != ""
	if hasActivations == hasAudit || flags.NArg() != 0 {
		return errors.New("usage: seam-align -activations <pairs.json> | -audit <measurements.json> | -enumerate <component> -repo <store> -measurements <m.json>")
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if hasActivations {
		var pairs struct {
			Source [][]float64 `json:"source"`
			Target [][]float64 `json:"target"`
		}
		if err := jsonfile.DecodeStrict(*activations, &pairs); err != nil {
			return err
		}
		residual, err := composition.SeamAlignmentResidual(pairs.Source, pairs.Target)
		if err != nil {
			return err
		}
		return encoder.Encode(struct {
			Residual float64 `json:"residual"`
		}{Residual: residual})
	}
	var measurements struct {
		Residuals []float64 `json:"residuals"`
		Parities  []float64 `json:"parities"`
	}
	if err := jsonfile.DecodeStrict(*audit, &measurements); err != nil {
		return err
	}
	verdict, err := composition.AlignmentResidualBiasAudit(measurements.Residuals, measurements.Parities)
	if err != nil {
		return err
	}
	return encoder.Encode(verdict)
}

// enumerateCandidates shortlists donors for one target component from the
// committed catalog and ranks the measured seams by residual then
// predicted fitness, reporting unmeasured shortlist entries by name.
func enumerateCandidates(repository, targetName, measurementsPath string, limit int) error {
	if strings.TrimSpace(repository) == "" || strings.TrimSpace(measurementsPath) == "" {
		return errors.New("seam-align -enumerate requires -repo and -measurements")
	}
	var measured struct {
		Measurements []composition.DonorSeamMeasurement `json:"measurements"`
	}
	if err := jsonfile.DecodeStrict(measurementsPath, &measured); err != nil {
		return err
	}
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	components, err := composition.LoadCatalog(ctx, store)
	if err != nil {
		return err
	}
	var target *composition.CatalogComponent
	for index := range components {
		if components[index].Name == targetName {
			target = &components[index]
			break
		}
	}
	if target == nil {
		return fmt.Errorf("component %q is not in the committed catalog", targetName)
	}
	index, err := composition.NewExactComponentIndex(components)
	if err != nil {
		return err
	}
	candidates, unmeasured, err := composition.EnumerateCompositionCandidates(
		index, *target, measured.Measurements, limit,
	)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(candidates); err != nil {
		return err
	}
	for _, descriptor := range unmeasured {
		fmt.Printf("unmeasured %s %s\n", descriptor.Model, descriptor.Name)
	}
	return nil
}
