package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// inputKind is how ComfyUI's page holds an input of a node's definition.
type inputKind int

const (
	// socketInput is fed by a link alone; widgetInput holds a value the page saves, a link standing in for it where one
	// is drawn; dynamicInput is a choice whose chosen option brings widgets of its own, named after it ("style.photo");
	// growingInput is a socket that grows a name per link ("values.a").
	socketInput inputKind = iota
	widgetInput
	dynamicInput
	growingInput
)

// widgetTypes are the input types the page draws a widget for.
var widgetTypes = []string{"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

// inputOptions are the options of an input's definition the page reads.
type inputOptions struct {
	ForceInput           bool            `json:"forceInput"`
	Default              json.RawMessage `json:"default"`
	ControlAfterGenerate any             `json:"control_after_generate"`
	ImageUpload          bool            `json:"image_upload"`
	Options              json.RawMessage `json:"options"`
}

// dynamicOption is one choice of a dynamic combo, with the inputs it brings.
type dynamicOption struct {
	Key    string          `json:"key"`
	Inputs json.RawMessage `json:"inputs"`
}

// readSpec reads an input's definition: its kind, and the options it was given.
func readSpec(spec []json.RawMessage) (inputKind, inputOptions, error) {
	var options inputOptions
	if len(spec) == 0 {
		return socketInput, options, errors.New("an input with no definition")
	}
	if len(spec) > 1 {
		if err := json.Unmarshal(spec[1], &options); err != nil {
			return socketInput, options, err
		}
	}
	var name string
	if json.Unmarshal(spec[0], &name) != nil {
		// A list of choices is a combo.
		return widgetInput, options, nil
	}
	switch {
	case name == "COMFY_DYNAMICCOMBO_V3":
		return dynamicInput, options, nil
	case name == "COMFY_AUTOGROW_V3":
		return growingInput, options, nil
	case options.ForceInput:
		return socketInput, options, nil
	case slices.Contains(widgetTypes, name):
		return widgetInput, options, nil
	case len(options.Default) > 0:
		// A type of a node pack's own that the page draws a widget for, given a default.
		return widgetInput, options, nil
	}
	return socketInput, options, nil
}

// extraValues is how many values the page saves after a widget's own: a seed's control after generating, an image upload button.
func (o inputOptions) extraValues() int {
	extra := 0
	if control, held := o.ControlAfterGenerate.(bool); held && control || o.ControlAfterGenerate != nil && !held {
		extra++
	}
	if o.ImageUpload {
		extra++
	}
	return extra
}

// widgetValues hands out a node's saved widget values in order.
type widgetValues struct {
	values []json.RawMessage
	at     int
}

func (w *widgetValues) next(name string) (any, error) {
	if w.at >= len(w.values) {
		return nil, fmt.Errorf("the widget %q has no saved value: %d saved", name, len(w.values))
	}
	var value any
	err := json.Unmarshal(w.values[w.at], &value)
	w.at++
	return value, err
}

// orderedInputs are the names of an option's inputs as its definition lists them, required first, with each one's
// definition: a JSON object's order, which a Go map loses.
func orderedInputs(raw json.RawMessage) ([]string, map[string][]json.RawMessage, error) {
	var groups struct {
		Required json.RawMessage `json:"required"`
		Optional json.RawMessage `json:"optional"`
	}
	if err := json.Unmarshal(raw, &groups); err != nil {
		return nil, nil, err
	}
	var names []string
	specs := map[string][]json.RawMessage{}
	for _, group := range []json.RawMessage{groups.Required, groups.Optional} {
		if len(group) == 0 {
			continue
		}
		decoder := json.NewDecoder(bytes.NewReader(group))
		if _, err := decoder.Token(); err != nil {
			return nil, nil, err
		}
		for decoder.More() {
			key, err := decoder.Token()
			if err != nil {
				return nil, nil, err
			}
			var spec []json.RawMessage
			if err := decoder.Decode(&spec); err != nil {
				return nil, nil, err
			}
			name := fmt.Sprint(key)
			names, specs[name] = append(names, name), spec
		}
	}
	return names, specs, nil
}

// inputs are node's inputs as the request holds them: its widgets' saved values by the names its definition gives
// them, a link wherever one feeds an input, and the node packs' own forms (customInputs).
func (f *flattener) inputs(s *scope, node *uiNode, info nodeInfo) (map[string]any, error) {
	var saved []json.RawMessage
	if len(node.WidgetsValues) > 0 && string(node.WidgetsValues) != "null" {
		if err := json.Unmarshal(node.WidgetsValues, &saved); err != nil {
			return nil, fmt.Errorf("widget values: %w", err)
		}
	}
	inputs := map[string]any{}
	if custom, held := customInputs[node.Type]; held {
		if err := custom(saved, inputs); err != nil {
			return nil, err
		}
		saved = nil
	}
	values := &widgetValues{values: saved}
	slotOf := func(name string) (uiSlot, bool) {
		for _, slot := range node.Inputs {
			if slot.Name == name || slot.Widget != nil && slot.Widget.Name == name {
				return slot, true
			}
		}
		return uiSlot{}, false
	}
	linkTo := func(name string) (bool, error) {
		slot, held := slotOf(name)
		if !held || slot.Link == nil {
			return false, nil
		}
		source, feeds, err := f.link(s, slot)
		if feeds {
			inputs[name] = source
		}
		return true, err
	}
	widget := func(name string, options inputOptions) error {
		value, err := values.next(name)
		if err != nil {
			return err
		}
		values.at += options.extraValues()
		if linked, err := linkTo(name); linked || err != nil {
			return err
		}
		if list, isList := value.([]any); isList {
			// The page wraps a list, which the server would otherwise read as a link [node, slot].
			value = map[string]any{"__value__": list}
		}
		inputs[name] = value
		return nil
	}
	names := append(append([]string(nil), info.InputOrder.Required...), info.InputOrder.Optional...)
	for _, name := range names {
		spec, held := info.Input.Required[name]
		if !held {
			spec = info.Input.Optional[name]
		}
		kind, options, err := readSpec(spec)
		if err != nil {
			return nil, fmt.Errorf("input %q: %w", name, err)
		}
		switch kind {
		case socketInput:
			if _, err := linkTo(name); err != nil {
				return nil, err
			}
		case growingInput:
			for _, slot := range node.Inputs {
				if strings.HasPrefix(slot.Name, name+".") {
					if _, err := linkTo(slot.Name); err != nil {
						return nil, err
					}
				}
			}
		case widgetInput:
			if err := widget(name, options); err != nil {
				return nil, err
			}
		case dynamicInput:
			if err := widget(name, options); err != nil {
				return nil, err
			}
			chosen := fmt.Sprint(inputs[name])
			var choices []dynamicOption
			if err := json.Unmarshal(options.Options, &choices); err != nil {
				return nil, fmt.Errorf("the options of %q: %w", name, err)
			}
			for _, option := range choices {
				if option.Key != chosen {
					continue
				}
				subNames, subSpecs, err := orderedInputs(option.Inputs)
				if err != nil {
					return nil, fmt.Errorf("the inputs of %q's option %q: %w", name, chosen, err)
				}
				for _, sub := range subNames {
					_, subOptions, err := readSpec(subSpecs[sub])
					if err != nil {
						return nil, err
					}
					if err := widget(name+"."+sub, subOptions); err != nil {
						return nil, err
					}
				}
			}
		}
	}
	return inputs, nil
}

// customInputs are the forms node packs give their widgets in the request, which the page builds and the definition
// does not state, read from the values the page saves.
var customInputs = map[string]func(saved []json.RawMessage, inputs map[string]any) error{
	// rgthree's lora loader: a header widget by its type, each lora as lora_1, lora_2 and so on with its unset fields
	// left out, and the add button's empty value.
	"Power Lora Loader (rgthree)": func(saved []json.RawMessage, inputs map[string]any) error {
		loras := 0
		for _, raw := range saved {
			var value any
			if err := json.Unmarshal(raw, &value); err != nil {
				return err
			}
			switch held := value.(type) {
			case string:
				inputs["➕ Add Lora"] = held
			case map[string]any:
				if kind, typed := held["type"].(string); typed {
					inputs[kind] = held
					continue
				}
				if _, lora := held["lora"]; !lora {
					continue
				}
				for key, field := range held {
					if field == nil {
						delete(held, key)
					}
				}
				loras++
				inputs["lora_"+strconv.Itoa(loras)] = held
			}
		}
		return nil
	},
	// rgthree's image comparer: the images it last showed.
	"Image Comparer (rgthree)": func(saved []json.RawMessage, inputs map[string]any) error {
		if len(saved) == 0 {
			return nil
		}
		var images any
		if err := json.Unmarshal(saved[0], &images); err != nil {
			return err
		}
		inputs["rgthree_comparer"] = map[string]any{"images": images}
		return nil
	},
}
