package mediacapability

import (
	"fmt"
	"reflect"
	"strings"

	"overgo/internal/diffusionimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
	"overgo/internal/speechsynth"
)

// Control is one typed field of a capability's request as a page can
// offer it: the request struct's own JSON field names and Go types are the
// declaration, so no capability lists its fields twice.
type Control struct {
	Name     string
	Type     string
	Required bool
}

const (
	// ControlText is a request field a page types as free text.
	ControlText = "text"
	// ControlInteger is a request field a page types as a whole number.
	ControlInteger = "integer"
	// ControlNumber is a request field a page types as a real number.
	ControlNumber = "number"
	// ControlBoolean is a request field a page sets true or false.
	ControlBoolean = "boolean"
)

// requestTypes binds each entry module to the zero value of the request its
// executor decodes; the build-tagged files add the device-bound ones.
var requestTypes = map[recipe.ModuleID]any{
	modelrecipe.ModuleOscillatorImagePrepare: oscillatorimage.Request{},
	modelrecipe.ModuleOscillatorVideoPrepare: oscillatorimage.VideoRequest{},
	modelrecipe.ModuleDiffusionImagePrepare:  diffusionimage.Request{},
	modelrecipe.ModuleLatentVideoPrepare:     latentvideo.WanRequest{},
	modelrecipe.ModuleReferenceVideoPrepare:  latentvideo.ReferenceEditRequest{},
	modelrecipe.ModuleSpeechTokenize:         speechsynth.SynthesisRequest{},
}

// Controls describes the request the entry module's executor decodes as
// page controls. A request carrying a field no page can type (a tensor, a
// nested document) is reported as the refusal reason instead, so the
// capability lists with why the page cannot run it rather than as a form
// that could never be submitted.
func Controls(entry recipe.ModuleID) ([]Control, string) {
	request, ok := requestTypes[entry]
	if !ok {
		return nil, fmt.Sprintf("entry module %s declares no page request", entry)
	}
	return describe(reflect.TypeOf(request))
}

func describe(request reflect.Type) ([]Control, string) {
	var controls []Control
	for index := range request.NumField() {
		field := request.Field(index)
		name, options, _ := strings.Cut(field.Tag.Get("json"), ",")
		if name == "" || name == "-" {
			continue
		}
		optional := strings.Contains(options, "omitzero") || strings.Contains(options, "omitempty")
		control := Control{Name: name}
		switch field.Type.Kind() {
		case reflect.String:
			control.Type, control.Required = ControlText, !optional
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			control.Type = ControlInteger
		case reflect.Float32, reflect.Float64:
			control.Type = ControlNumber
		case reflect.Bool:
			control.Type = ControlBoolean
		default:
			return nil, fmt.Sprintf("field %s needs %s the page cannot type", name, field.Type.Kind())
		}
		controls = append(controls, control)
	}
	return controls, ""
}
