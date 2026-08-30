// Command seam-align computes the seam alignment residual — the cheap
// causal proxy for whether a linear interface adapter can compose a seam —
// and bias-audits recorded residuals against measured bridge parity.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"os"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/composition"
	"overgo/internal/jsonfile"
)

func main() { clioptions.MainNamed("seam-align", run) }

func run() error {
	flags := flag.NewFlagSet("seam-align", flag.ContinueOnError)
	activations := flags.String("activations", "", "paired seam activations JSON path ({source, target})")
	audit := flags.String("audit", "", "recorded measurements JSON path ({residuals, parities})")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	hasActivations, hasAudit := strings.TrimSpace(*activations) != "", strings.TrimSpace(*audit) != ""
	if hasActivations == hasAudit || flags.NArg() != 0 {
		return errors.New("usage: seam-align -activations <pairs.json> | -audit <measurements.json>")
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
