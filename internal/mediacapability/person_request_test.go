package mediacapability

import (
	"encoding/json"
	"testing"

	"overgo/internal/recipe"
	"overgo/internal/testcheck"
)

// TestAPersonRequestTakesThePagesDefaults: a verified request loses the size it states and takes the defaults the
// model's page declares, so the size measured is the size a person gets; a size the page declares none for is left
// to the model's runtime; what the request asks for besides stays. One that states no size has none to lift.
func TestAPersonRequestTakesThePagesDefaults(t *testing.T) {
	const entry recipe.ModuleID = "fixture.person-request"
	requestTypes[entry] = struct {
		Prompt string `json:"prompt"`
		Width  int    `json:"width"`
		Height int    `json:"height"`
		Steps  int    `json:"steps"`
	}{}
	controlProviders[entry] = func(_ string, controls []Control) ([]Control, error) {
		applyBounds(controls, map[string]Bounds{"width": {Default: 2048}, "height": {Default: 2048}})
		return controls, nil
	}
	t.Cleanup(func() { delete(requestTypes, entry); delete(controlProviders, entry) })
	sized, lifted, err := PersonRequest(entry, "C:/models/fixture", `{"prompt":"a fox","width":32,"height":16,"steps":2,"seed":7}`)
	var fields map[string]any
	testcheck.NoError(t, json.Unmarshal([]byte(sized), &fields))
	if err != nil || !lifted || len(fields) != 4 || fields["width"] != 2048.0 || fields["height"] != 2048.0 || fields["prompt"] != "a fox" || fields["seed"] == nil {
		t.Fatalf("the request reads %s (lifted %t, %v), want the page's 2048 x 2048, its steps left to the model, its prompt and seed kept", sized, lifted, err)
	}
	if sized, lifted, err := PersonRequest("fixture.no-page", "", `{"prompt":"a fox","frames":21,"steps":2}`); err != nil || !lifted || sized != `{"prompt":"a fox"}` {
		t.Fatalf("a request of a module with no page reads %s (lifted %t, %v), want its size left to the model", sized, lifted, err)
	}
	if _, lifted, err := PersonRequest(entry, "", `{"prompt":"a fox"}`); err != nil || lifted {
		t.Fatalf("a request of no size: lifted %t (%v)", lifted, err)
	}
	if _, _, err := PersonRequest(entry, "", "not a request"); err == nil {
		t.Fatal("a request that is no JSON reads as one")
	}
}
