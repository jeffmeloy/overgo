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
	//go:embed recipes/transducer_execution.json
	transducerExecutionJSON []byte
)

// recurrentDeclaration is a reviewed recurrent checkpoint's whole statement:
// the model type it answers for, the frontend its encoder reads, the
// encoder's own bindings and the decoder's. A checkpoint of this form names
// its blank entry in the decoder binding and needs no architecture recipe
// beside it.
type recurrentDeclaration struct {
	ModelType  string                  `json:"model_type"`
	Frontend   audiodsp.FrontendConfig `json:"frontend"`
	Encoder    json.RawMessage         `json:"encoder"`
	Transducer json.RawMessage         `json:"transducer"`
}

// declaredRecognizer is one reviewed pairing: the declarations that answer
// for a model type, and the architecture recipe the rest of its numbers come
// from.
type declaredRecognizer struct {
	execution, frontend []byte
	recipe              func() (audioArchitectureRecipe, error)
	// recurrent is set instead of the three above when the reviewed
	// checkpoint states everything in one declaration.
	recurrent *recurrentDeclaration
}

// reviewedRecognizers keys the reviewed declarations by the model type their
// own architecture recipe names, so a checkpoint resolves by what it declares
// itself to be.
var reviewedRecognizers = map[string]declaredRecognizer{}

func init() {
	var recurrent recurrentDeclaration
	if err := json.Unmarshal(transducerExecutionJSON, &recurrent); err == nil && recurrent.ModelType != "" {
		reviewedRecognizers[recurrent.ModelType] = declaredRecognizer{recurrent: &recurrent}
	}
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
	// Transducer is the recurrent decoder's reviewed binding, present only
	// for a checkpoint that declares one. Such a checkpoint states its own
	// blank entry there, and reads the frontend's frames as they come.
	Transducer []byte
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
	if declared.recurrent != nil {
		return ReviewedRecognizer{
			Execution: declared.recurrent.Encoder, Frontend: declared.recurrent.Frontend,
			Transducer: declared.recurrent.Transducer,
		}, true, nil
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
