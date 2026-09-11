//overgo:runtime-inputs caller

// Command mechanism-census publishes an exact mechanism ownership and gap census.
package main

import (
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/trainingprogram"
)

type censusSpecification struct {
	Assessments []trainingprogram.MechanismAssessment `json:"assessments"`
}

type provenanceSpecification struct {
	Assessments []trainingprogram.MechanismAssessment `json:"assessments"`
	Repository  string                                `json:"repository"`
	Commit      string                                `json:"commit"`
	License     artifact.ID                           `json:"license"`
	Sources     []runrecord.ExternalMechanismSource   `json:"sources"`
}

type mechanismEvidenceSource struct {
	Role   trainingprogram.MechanismEvidenceRole `json:"role"`
	Source artifact.ID                           `json:"source"`
}

type evidenceSpecification struct {
	Mechanism artifact.ID               `json:"mechanism"`
	Sources   []mechanismEvidenceSource `json:"sources"`
}

var mechanismEvidenceRoles = [...]trainingprogram.MechanismEvidenceRole{
	trainingprogram.MechanismEvidenceGap,
	trainingprogram.MechanismEvidenceFalsifier,
	trainingprogram.MechanismEvidenceCostBound,
	trainingprogram.MechanismEvidenceBenefit,
}

func main() {
	clioptions.MainNamed("mechanism-census", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("mechanism-census", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	input := flags.String("input", "", "path to the mechanism census JSON specification")
	provenanceInput := flags.String("provenance", "", "path to the external provenance JSON specification")
	evidenceInput := flags.String("evidence", "", "path to one complete typed mechanism-evidence bundle")
	storePath := flags.String("store", "", "OvergoDB root")
	if err := flags.Parse(args); err != nil {
		return err
	}
	modes := 0
	for _, selected := range []bool{*input != "", *provenanceInput != "", *evidenceInput != ""} {
		if selected {
			modes++
		}
	}
	if flags.NArg() != 0 || modes != 1 || *storePath == "" {
		return errors.New("usage: mechanism-census (-input <census.json> | -provenance <source.json> | -evidence <bundle.json>) -store <overgodb>")
	}
	path := cmp.Or(*input, *provenanceInput, *evidenceInput)
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	if *provenanceInput != "" {
		return publishProvenance(data, store, output)
	}
	if *evidenceInput != "" {
		return publishEvidence(data, store, output)
	}
	var specification censusSpecification
	if err := strictjson.DecodeBytes(data, &specification); err != nil {
		return err
	}
	census, err := trainingprogram.CompileMechanismCensus(specification.Assessments)
	if err != nil {
		return err
	}
	content, err := census.Content()
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, artifact.Batch{
		Key: "research/mechanism-census/" + census.ID().String(), Contents: []artifact.Content{content}, Lineage: census.Lineage(),
	}); err != nil {
		return err
	}
	open := 0
	for _, assessment := range census.Assessments() {
		if assessment.GapState == trainingprogram.MechanismGapOpen {
			open++
		}
	}
	fmt.Fprintf(output, "mechanism-census: committed %s assessments=%d open=%d closed=%d\n",
		census.ID(), len(specification.Assessments), open, len(specification.Assessments)-open)
	return nil
}

func publishEvidence(data []byte, store *overgodb.Store, output io.Writer) error {
	var specification evidenceSpecification
	if err := strictjson.DecodeBytes(data, &specification); err != nil {
		return err
	}
	if len(specification.Sources) != len(mechanismEvidenceRoles) {
		return errors.New("mechanism-census: evidence bundle must contain each role exactly once")
	}
	sources := make(map[trainingprogram.MechanismEvidenceRole]artifact.ID, len(specification.Sources))
	usedSources := make(map[artifact.ID]bool, len(specification.Sources))
	for _, claim := range specification.Sources {
		if _, duplicate := sources[claim.Role]; duplicate || usedSources[claim.Source] {
			return errors.New("mechanism-census: evidence bundle repeats a role or source")
		}
		sources[claim.Role], usedSources[claim.Source] = claim.Source, true
	}
	contents := make([]artifact.Content, 0, len(mechanismEvidenceRoles))
	lineage := make([]artifact.Lineage, 0, len(mechanismEvidenceRoles)*2)
	identities := make([]artifact.ID, 0, len(mechanismEvidenceRoles))
	for _, role := range mechanismEvidenceRoles {
		source, found := sources[role]
		if !found {
			return errors.New("mechanism-census: evidence bundle contains an unknown or missing role")
		}
		evidence, err := trainingprogram.NewMechanismEvidence(role, specification.Mechanism, source)
		if err != nil {
			return err
		}
		content, err := evidence.Content()
		if err != nil {
			return err
		}
		contents = append(contents, content)
		lineage = append(lineage, evidence.Lineage()...)
		identities = append(identities, evidence.ID)
	}
	bundle, err := artifact.JSONID(artifact.KindEvidence, identities)
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch(
		"research/mechanism-evidence/"+bundle.String(), contents, lineage, nil,
	)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(context.Background(), store, batch); err != nil {
		return err
	}
	fmt.Fprintf(output, "mechanism-census: committed evidence bundle %s mechanism=%s claims=%d\n",
		bundle, specification.Mechanism, len(identities))
	return nil
}

func publishProvenance(data []byte, store *overgodb.Store, output io.Writer) error {
	var specification provenanceSpecification
	if err := strictjson.DecodeBytes(data, &specification); err != nil {
		return err
	}
	census, err := trainingprogram.CompileMechanismCensus(specification.Assessments)
	if err != nil {
		return err
	}
	provenance, err := runrecord.NewExternalMechanismProvenance(
		census, specification.Repository, specification.Commit, specification.License, specification.Sources,
	)
	if err != nil {
		return err
	}
	censusContent, err := census.Content()
	if err != nil {
		return err
	}
	provenanceContent, err := provenance.Content()
	if err != nil {
		return err
	}
	lineage := append(census.Lineage(), provenance.Lineage()...)
	if _, err := artifact.CommitBatch(context.Background(), store, artifact.Batch{
		Key:      "research/external-provenance/" + provenance.ID.String(),
		Contents: []artifact.Content{censusContent, provenanceContent}, Lineage: lineage,
	}); err != nil {
		return err
	}
	fmt.Fprintf(output, "mechanism-census: committed provenance %s census=%s mechanisms=%d transfer=%s\n",
		provenance.ID, census.ID(), len(specification.Sources), provenance.Transfer)
	return nil
}
