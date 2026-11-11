package mediacapability

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"slices"
	"sync"
)

// presetReadingsJSON names each preset under reading by its family, the package that declares it: pending until its
// verdict, then admitted or refused (guidance-ops/preset-readings writes each with its evidence in docs/verification).
//
//go:embed preset_readings.json
var presetReadingsJSON []byte

// The statuses a preset under reading takes.
const (
	presetPending  = "pending"
	presetAdmitted = "admitted"
	presetRefused  = "refused"
)

// Preset families: the packages whose declarations a page's presets come from.
const (
	latentImagePresets = "latentimage"
	latentVideoPresets = "latentvideo"
	senseNovaPresets   = "sensenova"
)

// presetReadings is the embedded record, read once, every status in it one it knows.
var presetReadings = sync.OnceValues(func() (map[string]map[string]string, error) {
	var readings map[string]map[string]string
	if err := json.Unmarshal(presetReadingsJSON, &readings); err != nil {
		return nil, fmt.Errorf("media capability: preset readings: %w", err)
	}
	for family, presets := range readings {
		for preset, status := range presets {
			if !slices.Contains([]string{presetPending, presetAdmitted, presetRefused}, status) {
				return nil, fmt.Errorf("media capability: the reading of %s's preset %s is %q, none of pending, admitted and refused", family, preset, status)
			}
		}
	}
	return readings, nil
})

// offeredPresets are the presets of a family a page offers: each its readings name only once admitted, and every
// other as declared.
func offeredPresets(readings map[string]map[string]string, family string, presets []string) []string {
	return slices.DeleteFunc(slices.Clone(presets), func(preset string) bool {
		status, read := readings[family][preset]
		return read && status != presetAdmitted
	})
}
