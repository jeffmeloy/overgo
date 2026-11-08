package mediacapability

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"

	"overgo/internal/recipe"
)

// personFields are what a verification states to keep its request small, and what a person asking for nothing else
// leaves to the model.
var personFields = []string{"frames", "width", "height", "steps"}

// PersonRequest is a verified request at the size a person asks for: each of personFields it states at the default the
// model's page offers, where the page declares one (Controls), and otherwise left to the model's own runtime to fill,
// as Wan's profile fills a clip's. The page and the measurement read one declaration, so the size timed is the size a
// person gets. lifted says the verification stated a size to lift; a request that states none is a person's already.
func PersonRequest(entry recipe.ModuleID, directory, request string) (sized string, lifted bool, err error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(request), &fields); err != nil {
		return "", false, fmt.Errorf("media capability: the verified request: %w", err)
	}
	controls, _ := Controls(entry, directory)
	for _, name := range personFields {
		if _, held := fields[name]; !held {
			continue
		}
		lifted = true
		delete(fields, name)
		if at := slices.IndexFunc(controls, func(control Control) bool { return control.Name == name && control.Bounds != nil }); at >= 0 && controls[at].Bounds.Default > 0 {
			fields[name] = json.RawMessage(strconv.Itoa(controls[at].Bounds.Default))
		}
	}
	data, err := json.Marshal(fields)
	return string(data), lifted, err
}
