package mediacapability

import (
	_ "embed"
	"fmt"
	"slices"
	"sync"

	"overgo/internal/evaluation"
	"overgo/internal/strictjson"
)

// presetReadingsJSON is the record of the presets under reading (guidance-ops/preset-readings): each by the model it is
// read on, with its verdict once read against that model's reference at a person's size, the two evaluation records it
// read, and the judge and suite that read them.
//
//go:embed preset_readings.json
var presetReadingsJSON []byte

// presetRecord is the record's form: the judge and each read model's suite, said once, and each model's readings by
// preset, the model by its published name, so two models may each read a preset of one name (a speed choice that every
// model offers alike).
type presetRecord struct {
	Judge    string                              `json:"judge"`
	Suites   map[string]string                   `json:"suites"`
	Readings map[string]map[string]presetReading `json:"readings"`
}

// presetReading is one preset's reading on its model: pending while it has no verdict.
type presetReading struct {
	Reference string `json:"reference,omitempty"`
	Arm       string `json:"arm,omitempty"`
	// Negative is the negative prompt the arm asked each case with, beside a preset that guides away from one (NAG);
	// none where the arm named none, a model asking its own then (Wan's student, its reference default).
	Negative string                    `json:"negative,omitempty"`
	Verdict  *evaluation.PresetVerdict `json:"verdict,omitempty"`
	// Pairwise is the verdict of the arm's pictures chosen against the reference's in both orders, with the evaluation
	// record it read, where the model's pictures were read so (evaluate -pairwise-evaluation).
	Pairwise *pairwiseReading `json:"pairwise,omitempty"`
}

// pairwiseReading is a preset's pairwise verdict and the judged record of its pairwise suite.
type pairwiseReading struct {
	Record  string                   `json:"record"`
	Verdict evaluation.PresetVerdict `json:"verdict"`
}

// offered says a page offers the read preset (the owner's direction of 2026-11-16, common-controls/gain-only): its
// verdict, and its pairwise verdict where it was read so, each find no loss beyond chance, and one of them a gain.
func (reading presetReading) offered() bool {
	verdict, pairwise := reading.Verdict, reading.Pairwise
	return verdict != nil && verdict.Admitted && (pairwise == nil || pairwise.Verdict.Admitted) &&
		(verdict.Gained() || pairwise != nil && pairwise.Verdict.Gained())
}

// presetReadings is the embedded record, read once.
var presetReadings = sync.OnceValues(func() (presetRecord, error) {
	var record presetRecord
	if err := strictjson.DecodeBytes(presetReadingsJSON, &record); err != nil {
		return presetRecord{}, fmt.Errorf("media capability: preset readings: %w", err)
	}
	return record, nil
})

// offeredPresets are the presets of the named model a page offers: each the record reads only where its reading shows a
// gain, and every other as declared.
func offeredPresets(record presetRecord, model string, presets []string) []string {
	return slices.DeleteFunc(slices.Clone(presets), func(preset string) bool {
		reading, read := record.Readings[model][preset]
		return read && !reading.offered()
	})
}
