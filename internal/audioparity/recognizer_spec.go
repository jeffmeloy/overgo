package audioparity

import (
	_ "embed"
	"encoding/json"
	"fmt"

	"overgo/internal/audiodsp"
)

// A speech checkpoint carries its weights and its configuration, and not the
// two declarations a recognizer needs: how its encoder's tensors are bound,
// and how a waveform becomes the features that encoder reads. Those are
// reviewed here, beside the architecture recipe that names the model type
// each answers for, and are handed to the runtime rather than inferred from a
// family name at the point of use.
var (
	//go:embed recipes/granite5asr_execution.json
	graniteExecutionJSON []byte
	//go:embed recipes/granite5asr_frontend.json
	graniteFrontendJSON []byte
)

// declaredRecognizer is one reviewed pairing: the declarations that answer
// for a model type, and the architecture recipe the rest of its numbers come
// from.
type declaredRecognizer struct {
	execution, frontend []byte
	recipe              func() (audioArchitectureRecipe, error)
}

// reviewedRecognizers keys the reviewed declarations by the model type their
// own architecture recipe names, so a checkpoint resolves by what it declares
// itself to be.
var reviewedRecognizers = map[string]declaredRecognizer{}

func init() {
	recipe, _, err := graniteRecipe()
	if err != nil {
		return // the recipe's own validation reports this where it is read
	}
	reviewedRecognizers[recipe.ModelType] = declaredRecognizer{
		execution: graniteExecutionJSON, frontend: graniteFrontendJSON,
		recipe: func() (audioArchitectureRecipe, error) {
			value, _, recipeErr := graniteRecipe()
			return value, recipeErr
		},
	}
}

// ReviewedRecognizer is what a reviewed speech checkpoint declares to the
// runtime that will execute it: the encoder's tensor bindings as the reviewed
// bytes themselves, the frontend that turns a waveform into features, how
// those features are grouped, and which vocabulary entry the classifier reads
// as blank. The bindings stay bytes here so the review keeps its home on the
// host, and the runtime that decodes them keeps its own.
type ReviewedRecognizer struct {
	Execution []byte
	Frontend  audiodsp.FrontendConfig
	Grouping  audiodsp.GroupedFeatureConfig
	Blank     int
}

// RecognizerSpecFor returns the reviewed declarations for a model type,
// and reports whether any answer for it. A checkpoint whose type is not
// reviewed here is refused by its caller rather than loaded against another
// family's bindings.
func RecognizerSpecFor(modelType string) (ReviewedRecognizer, bool, error) {
	declared, known := reviewedRecognizers[modelType]
	if !known {
		return ReviewedRecognizer{}, false, nil
	}
	var frontend audiodsp.FrontendConfig
	if err := json.Unmarshal(declared.frontend, &frontend); err != nil {
		return ReviewedRecognizer{}, true, fmt.Errorf("audio parity: %s frontend declaration: %w", modelType, err)
	}
	recipe, err := declared.recipe()
	if err != nil {
		return ReviewedRecognizer{}, true, err
	}
	return ReviewedRecognizer{
		Execution: declared.execution, Frontend: frontend,
		Grouping: audiodsp.GroupedFeatureConfig{
			StackFrames: int(recipe.Preprocessor.StackFactor),
			DeltaRadius: int(recipe.Preprocessor.DeltaWinLength-1) / 2,
			// The processor selects every grouped frame; no declaration
			// states a stride of its own.
			FinalFrameSamples: 1,
		},
		Blank: int(recipe.Config.PadTokenID),
	}, true, nil
}
