// Command training-objective publishes one training objective as store
// authority: a modality pair, an objective kind, and a registered
// dataset become an approved objective document whose compiled program
// carries the shared forward/backward/Muon phases with derived
// hyperparameters -- there is no learning-rate or momentum flag here,
// because the optimizer policy owns those facts. Datasets are named by
// their registered catalog alias, so the corpus is store authority too.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/overgodb"
	"overgo/internal/recipecontract"
	"overgo/internal/trainingprogram"
)

func main() {
	clioptions.MainNamed("training-objective", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("training-objective", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("repo", "", "OvergoDB root")
	name := flags.String("name", "", "objective name recorded in the document")
	kindText := flags.String("kind", "", "objective kind (loss semantics)")
	inputText := flags.String("input", "", "input modality")
	outputText := flags.String("output", "", "output modality")
	metricText := flags.String("metric", "", "evaluation metric")
	datasetName := flags.String("dataset", "", "registered dataset catalog name the objective trains on")
	evidenceNote := flags.String("evidence", "", "evidence note identifying why this objective is approved")
	if err := flags.Parse(args); err != nil {
		return err
	}
	for flagName, value := range map[string]string{
		"name": *name, "kind": *kindText, "input": *inputText, "output": *outputText,
		"metric": *metricText, "dataset": *datasetName, "evidence": *evidenceNote,
	} {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("training-objective: -%s is required", flagName)
		}
	}
	if flags.NArg() != 0 {
		return errors.New("usage: training-objective [-repo <path>] -name <n> -kind <k> -input <m> -output <m> -metric <m> -dataset <catalog-name> -evidence <note>")
	}
	input := recipecontract.Modality(*inputText)
	outputModality := recipecontract.Modality(*outputText)
	if !recipecontract.ValidModality(input) || !recipecontract.ValidModality(outputModality) {
		return fmt.Errorf("training-objective: unknown modality in %q -> %q", *inputText, *outputText)
	}
	root := strings.TrimSpace(*repository)
	if root == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		root = roots.Store
	}
	store, err := overgodb.Open(root)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	objective, err := publishObjective(ctx, store, objectiveRequest{
		Name: *name, Kind: trainingprogram.ObjectiveKind(*kindText),
		Input: input, Output: outputModality,
		Metric:  trainingprogram.EvaluationMetric(*metricText),
		Dataset: *datasetName, Evidence: *evidenceNote,
	})
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "objective %s pair %s->%s dataset %s\n",
		objective.ID, input, outputModality, *datasetName)
	return err
}

type objectiveRequest struct {
	Name     string
	Kind     trainingprogram.ObjectiveKind
	Input    recipecontract.Modality
	Output   recipecontract.Modality
	Metric   trainingprogram.EvaluationMetric
	Dataset  string
	Evidence string
}

// publishObjective resolves the registered dataset, derives the stable
// contract profiles the same way the bootstrap recipe does, and
// commits the approved objective with every reference present.
func publishObjective(
	ctx context.Context,
	store *overgodb.Store,
	request objectiveRequest,
) (trainingprogram.ObjectiveDocument, error) {
	datasetID, found, err := store.ResolveAlias(ctx, "dataset.registered."+request.Dataset)
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	if !found {
		return trainingprogram.ObjectiveDocument{}, fmt.Errorf(
			"training-objective: dataset %q is not registered in the catalog", request.Dataset)
	}
	derived := func(kind artifact.Kind, role string) (artifact.ID, error) {
		return artifact.IdentifyBytes(kind, []byte("overgo/training-objective/"+string(request.Kind)+"/"+role))
	}
	splitID, err := artifact.IdentifyBytes(artifact.KindDatasetShard,
		[]byte("overgo/training-objective/split/"+datasetID.String()))
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	processorID, err := derived(artifact.KindProfile, "processor")
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	lossID, err := derived(artifact.KindProfile, "loss")
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	evaluationID, err := derived(artifact.KindProfile, "evaluation")
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	evidenceID, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(request.Evidence))
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	objective, err := trainingprogram.NewObjective(trainingprogram.ObjectiveSpec{
		Name: request.Name, Kind: request.Kind,
		Signature: recipecontract.ModalitySignature{
			Inputs:  []recipecontract.Modality{request.Input},
			Outputs: []recipecontract.Modality{request.Output},
		},
		Dataset: datasetID, Split: splitID, Processors: []artifact.ID{processorID},
		Loss: lossID, Evaluation: evaluationID,
		Metric: request.Metric, Evidence: []artifact.ID{evidenceID},
		Authority: trainingprogram.ObjectiveApproved,
	})
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	content, err := objective.Content()
	if err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	}
	batch := artifact.Batch{
		Key: "training-objective/" + objective.ID.String(),
		Artifacts: []artifact.Descriptor{
			{ID: splitID}, {ID: processorID}, {ID: lossID}, {ID: evaluationID}, {ID: evidenceID},
		},
		Contents: []artifact.Content{content},
		Aliases: []artifact.AliasBinding{{
			Name: "objective.registered." + string(request.Input) + "-" + string(request.Output), Target: objective.ID,
		}},
	}
	if current, bound, err := store.ResolveAlias(ctx, batch.Aliases[0].Name); err != nil {
		return trainingprogram.ObjectiveDocument{}, err
	} else if bound {
		if current == objective.ID {
			batch.Aliases = nil
		} else {
			batch.Aliases[0].Previous = artifact.IDPointer(current)
		}
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		return trainingprogram.ObjectiveDocument{}, err
	}
	return objective, nil
}
