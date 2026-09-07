package mediacapability

import (
	"fmt"
	"reflect"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/diffusionimage"
	"overgo/internal/latentvideo"
	"overgo/internal/modelrecipe"
	"overgo/internal/oscillatorimage"
	"overgo/internal/recipe"
	"overgo/internal/speechsynth"
	"overgo/internal/vqaserve"
)

// Control is one typed field of a capability's request as a page can
// offer it: the request struct's own JSON field names and Go types are the
// declaration, so no capability lists its fields twice.
type Control struct {
	Name     string
	Type     string
	Required bool
	// Choices are the values the model's own artifact exports for the
	// field (a speech model's voices); empty when any value may be typed.
	Choices []string
	// Label names an artifact slot the way a page shows it (the request
	// field's label tag); Media is the kind of artifact it takes (the
	// media tag: image, video or audio), so a page offers the right stored
	// files and routes an attachment to the slot of its kind.
	Label, Media string
	// Bounds are a numeric field's declared default, step and rate (the
	// model's profile and strides), from which a page derives its presets.
	Bounds *Bounds
}

// Bounds is a numeric control's declared default, the step a valid value
// moves by (zero when any value is valid) and, for a frame count, the
// frames one second holds.
type Bounds struct {
	Default, Step, Rate int
}

// Bounded reports a control's bounds as one tuple; declared is false for a
// control without them.
func (control Control) Bounded() (defaultValue, step, rate int, declared bool) {
	if control.Bounds == nil {
		var none Bounds
		return none.Default, none.Step, none.Rate, false
	}
	return control.Bounds.Default, control.Bounds.Step, control.Bounds.Rate, true
}

// Declared reports the control's fields as one tuple, so a consumer with
// its own control type binds the declaration without naming this one.
func (control Control) Declared() (name, kind string, required bool, choices []string) {
	return control.Name, control.Type, control.Required, control.Choices
}

// Slot reports an artifact control's label and media kind (empty for a typed field).
func (control Control) Slot() (label, media string) { return control.Label, control.Media }

// choiceProviders name, per entry module, the request fields whose values
// the model directory exports; the provider reads them from the artifact.
var choiceProviders = map[recipe.ModuleID]func(directory string) (map[string][]string, error){
	modelrecipe.ModuleSpeechTokenize: func(directory string) (map[string][]string, error) {
		voices, err := speechsynth.ListVoices(directory)
		if err != nil {
			return nil, err
		}
		return map[string][]string{"voice": voices}, nil
	},
}

const (
	// ControlArtifact is a request field naming an artifact the store holds,
	// which a page fills from a media card's stored id.
	ControlArtifact = "artifact"
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
	modelrecipe.ModuleVQAPrepare:             vqaserve.Request{},
}

// Controls describes the request the entry module's executor decodes as
// page controls, with the choices the model directory exports for a field.
// A request carrying a field no page can type (a tensor, a nested
// document) is reported as the refusal reason instead, so the capability
// lists with why the page cannot run it rather than as a form that could
// never be submitted; a directory whose exported choices cannot be read
// is reported the same way.
func Controls(entry recipe.ModuleID, directory string) ([]Control, string) {
	request, ok := requestTypes[entry]
	if !ok {
		return nil, fmt.Sprintf("entry module %s declares no page request", entry)
	}
	controls, refusal := describe(reflect.TypeOf(request))
	if refusal != "" || directory == "" {
		return controls, refusal
	}
	if provider, provides := choiceProviders[entry]; provides {
		choices, err := provider(directory)
		if err != nil {
			return nil, fmt.Sprintf("the model directory exports no choices: %v", err)
		}
		for index := range controls {
			controls[index].Choices = choices[controls[index].Name]
		}
	}
	if provider, provides := boundsProviders[entry]; provides {
		bounds, err := provider(directory)
		if err != nil {
			return nil, fmt.Sprintf("the model directory declares no bounds: %v", err)
		}
		applyBounds(controls, bounds)
	}
	return controls, ""
}

// boundsProviders name, per entry module, the request fields whose bounds
// the model's profile and strides declare; the provider reads them from
// the model directory.
var boundsProviders = map[recipe.ModuleID]func(directory string) (map[string]Bounds, error){
	modelrecipe.ModuleLatentVideoPrepare: func(directory string) (map[string]Bounds, error) {
		declared, err := latentvideo.ControlBounds(directory)
		if err != nil {
			return nil, err
		}
		bounds := make(map[string]Bounds, len(declared))
		for name, bound := range declared {
			bounds[name] = Bounds{Default: bound.Default, Step: bound.Step, Rate: bound.Rate}
		}
		return bounds, nil
	},
}

// applyBounds binds each declared bound to the control of its name.
func applyBounds(controls []Control, bounds map[string]Bounds) {
	for index := range controls {
		if bound, declared := bounds[controls[index].Name]; declared {
			controls[index].Bounds = &bound
		}
	}
}

// describe derives the page's controls from the request's exported JSON
// fields. A scalar field is a typed control; a field the page cannot type
// (a tensor, a plan) is omitted when the request marks it optional, since
// the runtime resolves it from the typed fields, and refuses the whole
// request when it is required.
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
		// An artifact id names something the store holds (a clip a page
		// attaches), which a page fills from a media card rather than types.
		if field.Type == reflect.TypeOf(artifact.ID{}) {
			control.Type, control.Required = ControlArtifact, !optional
			control.Label, control.Media = field.Tag.Get("label"), field.Tag.Get("media")
			controls = append(controls, control)
			continue
		}
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
			if optional {
				continue
			}
			return nil, fmt.Sprintf("field %s needs %s the page cannot type", name, field.Type.Kind())
		}
		controls = append(controls, control)
	}
	return controls, ""
}
