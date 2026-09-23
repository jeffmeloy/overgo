package vitencoder

import (
	_ "embed"
	"fmt"

	"overgo/internal/strictjson"
)

// A checkpoint states its geometry and image processor but not how its
// tensors are bound. That is reviewed here per family, read from the
// family's own reference code, and handed to the encoder.

//go:embed recipes/dinov2.json
var dinov2Declaration []byte

// reviewed maps a model type to its reviewed declaration.
var reviewed = map[string][]byte{"dinov2": dinov2Declaration}

// declarationFor returns the reviewed declaration for a model type; known is
// false when no reviewed declaration answers for it.
func declarationFor(modelType string) (Declaration, bool, error) {
	data, known := reviewed[modelType]
	if !known {
		return Declaration{}, false, nil
	}
	var declaration Declaration
	if err := strictjson.DecodeBytes(data, &declaration); err != nil {
		return Declaration{}, false, fmt.Errorf("vitencoder: %s declaration: %w", modelType, err)
	}
	if declaration.ModelType != modelType {
		return Declaration{}, false, fmt.Errorf("vitencoder: the %s declaration names model type %q", modelType, declaration.ModelType)
	}
	return declaration, true, nil
}
