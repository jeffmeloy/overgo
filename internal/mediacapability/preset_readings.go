package mediacapability

import (
	_ "embed"
	"fmt"
	"slices"
	"sync"

	"overgo/internal/strictjson"
)

// presetReadingsJSON is the record of the guidance presets under reading (guidance-ops/preset-readings): each by its
// family, the package that declares it, and the model it is read on, with its verdict once read against that model's
// reference at a person's size, the two evaluation records it read, and the judge and suite that read them.
//
//go:embed preset_readings.json
var presetReadingsJSON []byte

// Preset families: the packages whose declarations a page's presets come from.
const (
	latentImagePresets = "latentimage"
	latentVideoPresets = "latentvideo"
	senseNovaPresets   = "sensenova"
)

// presetRecord is the record's form: the judge and each read model's suite, said once, and the readings.
type presetRecord struct {
	Judge    string                              `json:"judge"`
	Suites   map[string]string                   `json:"suites"`
	Readings map[string]map[string]presetReading `json:"readings"`
}

// presetReading is one preset's reading on its model: pending while it has no verdict.
type presetReading struct {
	Model     string         `json:"model"`
	Reference string         `json:"reference,omitempty"`
	Arm       string         `json:"arm,omitempty"`
	Verdict   *presetVerdict `json:"verdict,omitempty"`
}

// presetVerdict is the verdict evaluate -admit-preset prints for the reading's two records.
type presetVerdict struct {
	Wins     uint64  `json:"wins"`
	Losses   uint64  `json:"losses"`
	Ties     uint64  `json:"ties"`
	Media    int     `json:"media"`
	Tail     float64 `json:"tail"`
	Admitted bool    `json:"admitted"`
	Says     string  `json:"says"`
}

// presetReadings is the embedded record, read once.
var presetReadings = sync.OnceValues(func() (presetRecord, error) {
	var record presetRecord
	if err := strictjson.DecodeBytes(presetReadingsJSON, &record); err != nil {
		return presetRecord{}, fmt.Errorf("media capability: preset readings: %w", err)
	}
	return record, nil
})

// offeredPresets are the presets of a family a page offers: each the record reads only once its verdict admits it,
// and every other as declared.
func offeredPresets(record presetRecord, family string, presets []string) []string {
	return slices.DeleteFunc(slices.Clone(presets), func(preset string) bool {
		reading, read := record.Readings[family][preset]
		return read && (reading.Verdict == nil || !reading.Verdict.Admitted)
	})
}
