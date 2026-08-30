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

	"overgo/internal/artifact"
	"overgo/internal/bridgetrain"
	"overgo/internal/clioptions"
	"overgo/internal/composition"
	"overgo/internal/jsonfile"
	"overgo/internal/optimizer"
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
	realize := flags.String("realize", "", "realization specification JSON path: align-init, train donors-frozen over recorded activations, assemble")
	selectFlag := flags.String("select", "", "composite scores JSON path: judge realized composites on the multidimensional fitness ({scores})")
	emit := flags.String("emit", "", "promotion emission JSON path: emit one fit composite through the ablation-armed gate ({verdict, policy, evidence})")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if strings.TrimSpace(*selectFlag) != "" {
		return selectComposites(*selectFlag)
	}
	if strings.TrimSpace(*emit) != "" {
		return emitPromotion(*emit)
	}
	if strings.TrimSpace(*realize) != "" {
		return realizeCandidate(*repoFlag, *realize)
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

// selectComposites judges realized composites on the multidimensional
// improvement fitness without scalarization and prints the ordered
// verdicts, fit composites first.
func selectComposites(scoresPath string) error {
	var scored struct {
		Scores []composition.CompositeScore `json:"scores"`
	}
	if err := jsonfile.DecodeStrict(scoresPath, &scored); err != nil {
		return err
	}
	verdicts, err := composition.ScoreCompositeSelection(scored.Scores)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(verdicts)
}

// emitPromotion emits one fit composite through the existing ablation-
// armed composition promotion gate and prints the validated evidence.
func emitPromotion(emissionPath string) error {
	var emission struct {
		Verdict  composition.CompositeFitnessVerdict             `json:"verdict"`
		Policy   composition.RepresentationBridgePromotionPolicy `json:"policy"`
		Evidence composition.RepresentationBridgePromotion       `json:"evidence"`
	}
	if err := jsonfile.DecodeStrict(emissionPath, &emission); err != nil {
		return err
	}
	policy, err := composition.NewRepresentationBridgePromotionPolicy(emission.Policy)
	if err != nil {
		return err
	}
	promotion, err := composition.EmitAblationGatedPromotion(emission.Verdict, policy, emission.Evidence)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	return encoder.Encode(promotion)
}

// realizeCandidate realizes one enumerated candidate from a strict
// specification: the adapter align-initializes from the recorded seam
// activations, trains donors-frozen through bridgetrain with the recorded
// activations replayed as the frozen forwards, and the assembled
// composite record commits with its checkpoint evidence.
func realizeCandidate(repository, specificationPath string) error {
	if strings.TrimSpace(repository) == "" {
		return errors.New("seam-align -realize requires -repo")
	}
	var specification struct {
		Candidate   composition.CompositionCandidate `json:"candidate"`
		Target      artifact.ID                      `json:"target"`
		Activations struct {
			Source [][]float64 `json:"source"`
			Target [][]float64 `json:"target"`
		} `json:"activations"`
		Training struct {
			Dataset        artifact.ID           `json:"dataset"`
			Examples       []bridgetrain.Example `json:"examples"`
			TrainingPolicy artifact.ID           `json:"training_policy"`
			Config         optimizer.Config      `json:"config"`
			Epochs         int                   `json:"epochs"`
		} `json:"training"`
		Assembly struct {
			Architecture string      `json:"architecture"`
			Recipe       artifact.ID `json:"recipe"`
		} `json:"assembly"`
	}
	if err := jsonfile.DecodeStrict(specificationPath, &specification); err != nil {
		return err
	}
	store, err := overgodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	realized, err := composition.RealizeCompositionCandidate(
		ctx, store, specification.Candidate, specification.Target,
		composition.RealizationSeamActivations{
			Source: specification.Activations.Source, Target: specification.Activations.Target,
		},
		composition.RealizationTraining{
			Dataset: specification.Training.Dataset, Examples: specification.Training.Examples,
			SourceForward:  composition.RecordedSeamForward{Model: specification.Candidate.Donor},
			TargetForward:  composition.RecordedSeamForward{Model: specification.Target},
			TrainingPolicy: specification.Training.TrainingPolicy,
			Config:         specification.Training.Config, Epochs: specification.Training.Epochs,
		},
		composition.RealizationAssembly{
			Architecture: specification.Assembly.Architecture, Recipe: specification.Assembly.Recipe,
		},
	)
	if err != nil {
		return err
	}
	if realized.Adapter.BridgeAfter.Valid() {
		if _, err := store.Commit(ctx, realized.Adapter.Batch); err != nil {
			return err
		}
	}
	compositeBatch, err := realized.Composite.Batch("composition/realized/" + realized.Composite.ID.String())
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, compositeBatch); err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(realized.Composite); err != nil {
		return err
	}
	fmt.Printf("residual %.6f adapter %s composite %s\n",
		realized.Residual, realized.Adapter.BridgeAfter, realized.Composite.ID)
	return nil
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
