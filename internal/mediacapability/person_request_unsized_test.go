package mediacapability

import (
	"testing"

	"overgo/internal/modelrecipe"
)

// TestARequestTheModelDrawsAtNoSizeOfItsOwnIsNotLifted holds user-size-inapplicable: SimpleDiffusion states no size of
// its own, so its verified request, lifted of its size and steps with no page default to put in their place, is one
// the model refuses, and the request at the size a person asks for is the verified one, nothing lifted. The full
// measurement of 2026-11-24 ran it lifted and was refused, "diffusionimage: positive steps, height, and width required".
func TestARequestTheModelDrawsAtNoSizeOfItsOwnIsNotLifted(t *testing.T) {
	t.Parallel()
	verified := `{"seed":7,"steps":8,"height":256,"width":256}`
	if sized, lifted, err := PersonRequest(modelrecipe.ModuleDiffusionImagePrepare, "", verified); err != nil || lifted || sized != verified {
		t.Fatalf("SimpleDiffusion's request at the size a person asks for reads %s (lifted %t, %v), want its verified one, nothing lifted", sized, lifted, err)
	}
}
