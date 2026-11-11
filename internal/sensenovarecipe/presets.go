package sensenovarecipe

import (
	_ "embed"
	"fmt"
	"maps"
	"slices"

	"overgo/internal/sampling"
	"overgo/internal/strictjson"
)

//go:embed presets.json
var presetsJSON []byte

// preset is one of SenseNova's recipes beside the authors': how its guidance varies over the steps, written as Wan's
// and MOSS's presets are (sampling.Guidance).
type preset struct {
	Guidance sampling.Guidance `json:"guidance"`
}

// declaredPresets are the presets SenseNova offers, by name.
func declaredPresets() (map[string]preset, error) {
	var presets map[string]preset
	if err := strictjson.DecodeBytes(presetsJSON, &presets); err != nil {
		return nil, fmt.Errorf("sensenova recipe: presets: %w", err)
	}
	return presets, nil
}

// PresetNames are SenseNova's presets, in order, for a page to offer.
func PresetNames() ([]string, error) {
	presets, err := declaredPresets()
	return slices.Sorted(maps.Keys(presets)), err
}

// guidanceOf is the guidance over the steps the preset named runs; no name is the authors' recipe, which guides every
// step alike.
func guidanceOf(name string) (sampling.Guidance, error) {
	if name == "" {
		return sampling.Guidance{}, nil
	}
	presets, err := declaredPresets()
	if err != nil {
		return sampling.Guidance{}, err
	}
	declared, found := presets[name]
	if !found {
		return sampling.Guidance{}, fmt.Errorf("sensenova recipe: no preset %q", name)
	}
	return declared.Guidance, declared.Guidance.Validate()
}
