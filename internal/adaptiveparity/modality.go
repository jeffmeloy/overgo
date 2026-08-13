package adaptiveparity

import (
	"fmt"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
)

type ModalityRow struct {
	Task     recipe.Task
	RecipeID string
	Inputs   []string
	Outputs  []string
	State    string
	Reason   string
}

var inferenceTasks = []recipe.Task{
	recipe.TaskInference, recipe.TaskGeneration, recipe.TaskEmbedding, recipe.TaskRerank,
	recipe.TaskProjection, recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
	recipe.TaskSpeech, recipe.TaskImageGen, recipe.TaskVideoGen, recipe.TaskVQA,
}

// InferenceModalityMatrix compiles the recipe-owned inference surface. Missing
// executable recipes become refusals; command or family names cannot imply them.
func InferenceModalityMatrix() ([]ModalityRow, error) {
	modelID := stableID(artifact.KindModel, "adaptive-parity-model")
	definitions := make(map[recipe.Task]recipe.Definition)
	base, err := modelrecipe.InferenceWithModelDefinition(
		modelID,
		stableID(artifact.KindProfile, "adaptive-parity-profile"),
		stableID(artifact.KindModelDefinition, "adaptive-parity-definition"),
		recipe.PlacementHost, modelrecipe.DecodeSessionRequest, recipe.ResidencyHostCache,
	)
	if err != nil {
		return nil, err
	}
	definitions[recipe.TaskInference] = base
	for _, task := range inferenceTasks {
		if task == recipe.TaskInference {
			continue
		}
		definition, err := modelrecipe.CapabilityDefinition(task, modelID)
		if err == nil {
			definitions[task] = definition
		}
	}

	rows := make([]ModalityRow, 0, len(inferenceTasks))
	for _, task := range inferenceTasks {
		definition, ok := definitions[task]
		if !ok {
			inputs, outputs := intendedSignature(task)
			rows = append(rows, ModalityRow{
				Task: task, Inputs: inputs, Outputs: outputs, State: "refused",
				Reason: "no typed executable modelrecipe capability",
			})
			continue
		}
		row, err := modalityRow(definition)
		if err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func modalityRow(definition recipe.Definition) (ModalityRow, error) {
	row := ModalityRow{Task: definition.Task, RecipeID: definition.ID.String(), State: "supported"}
	for _, input := range definition.Inputs {
		modality, err := endpointModality(definition.Task, input.Data)
		if err != nil {
			return ModalityRow{}, fmt.Errorf("adaptive parity: recipe %s input %s: %w", definition.ID.String(), input.Name, err)
		}
		row.Inputs = append(row.Inputs, modality)
	}
	for _, output := range definition.Outputs {
		modality, err := endpointModality(definition.Task, output.Data)
		if err != nil {
			return ModalityRow{}, fmt.Errorf("adaptive parity: recipe %s output %s: %w", definition.ID.String(), output.Name, err)
		}
		row.Outputs = append(row.Outputs, modality)
	}
	if len(row.Inputs) == 0 || len(row.Outputs) == 0 {
		return ModalityRow{}, fmt.Errorf("adaptive parity: recipe %s has empty modality boundary", definition.ID.String())
	}
	return row, nil
}

func endpointModality(task recipe.Task, kind recipe.DataKind) (string, error) {
	switch kind {
	case recipe.DataText, recipe.DataTokens, recipe.DataLogits:
		return "text", nil
	case recipe.DataImage:
		return "image", nil
	case recipe.DataAudio:
		return "audio", nil
	case recipe.DataVideo:
		return "video", nil
	case recipe.DataEmbeddings, recipe.DataScores, recipe.DataRanking:
		return "table", nil
	case recipe.DataTensor:
		switch task {
		case recipe.TaskForecast:
			return "time-series", nil
		case recipe.TaskTabular, recipe.TaskImageGen:
			return "table", nil
		}
	}
	return "", fmt.Errorf("task %q data kind %q has no modality mapping", task, kind)
}

func intendedSignature(task recipe.Task) ([]string, []string) {
	switch task {
	case recipe.TaskGeneration:
		return []string{"text"}, []string{"text"}
	case recipe.TaskEmbedding:
		return []string{"text"}, []string{"table"}
	case recipe.TaskRerank:
		return []string{"text", "text"}, []string{"table"}
	case recipe.TaskProjection:
		return []string{"image"}, []string{"text"}
	case recipe.TaskVideoGen:
		return []string{"text"}, []string{"video"}
	default:
		return nil, nil
	}
}

func stableID(kind artifact.Kind, label string) artifact.ID {
	id, err := artifact.IdentifyBytes(kind, []byte(label))
	if err != nil {
		panic(err)
	}
	return id
}

func (row ModalityRow) Signature() string {
	return strings.Join(row.Inputs, "+") + "->" + strings.Join(row.Outputs, "+")
}

func FindModalityRow(rows []ModalityRow, task recipe.Task) (ModalityRow, bool) {
	index := slices.IndexFunc(rows, func(row ModalityRow) bool { return row.Task == task })
	if index < 0 {
		return ModalityRow{}, false
	}
	return rows[index], true
}
