package mediacapability

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"sync"

	"overgo/internal/recipe"
	"overgo/internal/strictjson"
)

// personFields are what a verification states to keep its request small, and what a person asking for nothing else
// leaves to the model.
var personFields = []string{"frames", "width", "height", "steps"}

// personSourcesJSON declares, by entry module, the source clip an edit at the size a person asks for takes, a clip of
// that size in the store: an edit's size is its source's, so a verified edit of a 32 x 16 clip stated no size to lift
// and LiveEdit's row had no time at a person's size (liveedit-size).
//
//go:embed person_sources.json
var personSourcesJSON []byte

// personSource is a declared person-size source: the clip by its artifact, its size, and why that clip.
type personSource struct {
	Source string `json:"source"`
	Width  int    `json:"width"`
	Height int    `json:"height"`
	Frames int    `json:"frames"`
	Why    string `json:"why"`
}

// personSources are the declared person-size sources, read once.
var personSources = sync.OnceValues(func() (map[recipe.ModuleID]personSource, error) {
	var sources map[recipe.ModuleID]personSource
	return sources, strictjson.DecodeBytes(personSourcesJSON, &sources)
})

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
	// An edit's size is its source's: at a person's size it edits the clip of that size its entry declares.
	if _, edits := fields["source_artifact"]; edits {
		sources, err := personSources()
		if err != nil {
			return "", false, fmt.Errorf("media capability: the declared person-size sources: %w", err)
		}
		if declared, held := sources[entry]; held {
			fields["source_artifact"], lifted = json.RawMessage(strconv.Quote(declared.Source)), true
		}
	}
	data, err := json.Marshal(fields)
	if err != nil || !lifted {
		return string(data), lifted, err
	}
	// A size left to a runtime that fills none is no request the model draws: it states no size of its own, and the
	// verified request is a person's already (SimpleDiffusion's refused "positive steps, height, and width required" in
	// the full measurement of 2026-11-24, user-size-inapplicable).
	if drawn, validates := requestTypes[entry].(interface{ DrawsRequest() error }); validates {
		decoded := reflect.New(reflect.TypeOf(drawn))
		if json.Unmarshal(data, decoded.Interface()) != nil || decoded.Elem().Interface().(interface{ DrawsRequest() error }).DrawsRequest() != nil {
			return request, false, nil
		}
	}
	return string(data), true, nil
}
