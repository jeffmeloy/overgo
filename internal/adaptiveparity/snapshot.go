// Package adaptiveparity owns imported reference evidence. Imports are inert:
// Overgo never executes or reads the source repository at runtime.
package adaptiveparity

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"overgo/internal/strictjson"
)

const Schema = 1

type Snapshot struct {
	Schema       int          `json:"schema"`
	Sources      []Source     `json:"sources"`
	Capabilities []Capability `json:"capabilities"`
}

type Source struct {
	Role       string `json:"role"`
	Repository string `json:"repository"`
	Commit     string `json:"commit"`
}

type Capability struct {
	ID          string       `json:"id"`
	Kind        string       `json:"kind"`
	EntryPoints []EntryPoint `json:"entry_points"`
	Signature   Signature    `json:"signature"`
	Artifacts   []Asset      `json:"artifacts"`
	Corpora     []Asset      `json:"corpora"`
	Goldens     []Asset      `json:"goldens"`
	Performance Performance  `json:"performance"`
}

type EntryPoint struct {
	Runtime string `json:"runtime"`
	Path    string `json:"path"`
	Symbol  string `json:"symbol"`
}

type Signature struct {
	Inputs  []string `json:"inputs"`
	Outputs []string `json:"outputs"`
}

type Asset struct {
	Identity string `json:"identity"`
	Locator  string `json:"locator"`
	SHA256   string `json:"sha256,omitempty"`
}

type Performance struct {
	ColdWall   Metric `json:"cold_wall"`
	WarmWall   Metric `json:"warm_wall"`
	PeakDevice Metric `json:"peak_device"`
	PeakHost   Metric `json:"peak_host"`
}

type Metric struct {
	State    string  `json:"state"`
	Value    float64 `json:"value,omitempty"`
	Unit     string  `json:"unit,omitempty"`
	Evidence string  `json:"evidence,omitempty"`
	Reason   string  `json:"reason,omitempty"`
}

// Import strictly decodes and validates a neutral snapshot.
func Import(reader io.Reader) (Snapshot, error) {
	var snapshot Snapshot
	if err := strictjson.Decode(reader, &snapshot); err != nil {
		return Snapshot{}, fmt.Errorf("adaptive parity snapshot: %w", err)
	}
	if err := snapshot.Validate(); err != nil {
		return Snapshot{}, err
	}
	return snapshot, nil
}

// Export writes deterministic JSON independent of source-repository layout.
func Export(writer io.Writer, snapshot Snapshot) error {
	if err := snapshot.Validate(); err != nil {
		return err
	}
	snapshot.Capabilities = slices.Clone(snapshot.Capabilities)
	slices.SortFunc(snapshot.Capabilities, func(a, b Capability) int { return strings.Compare(a.ID, b.ID) })
	encoder := json.NewEncoder(writer)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	return encoder.Encode(snapshot)
}

func (snapshot Snapshot) Validate() error {
	if snapshot.Schema != Schema {
		return fmt.Errorf("adaptive parity snapshot: schema %d, want %d", snapshot.Schema, Schema)
	}
	if len(snapshot.Sources) == 0 {
		return errors.New("adaptive parity snapshot: no pinned sources")
	}
	for _, source := range snapshot.Sources {
		if source.Role == "" || source.Repository == "" || !validCommit(source.Commit) {
			return errors.New("adaptive parity snapshot: source role, repository, and 40-hex commit required")
		}
	}
	if len(snapshot.Capabilities) == 0 {
		return errors.New("adaptive parity snapshot: no capabilities")
	}
	seen := map[string]bool{}
	for _, capability := range snapshot.Capabilities {
		if capability.ID == "" || capability.Kind == "" || seen[capability.ID] {
			return fmt.Errorf("adaptive parity snapshot: invalid or duplicate capability %q", capability.ID)
		}
		seen[capability.ID] = true
		if len(capability.EntryPoints) == 0 || len(capability.Signature.Inputs) == 0 || len(capability.Signature.Outputs) == 0 {
			return fmt.Errorf("adaptive parity snapshot: capability %q lacks entry point or modality signature", capability.ID)
		}
		for _, entry := range capability.EntryPoints {
			if entry.Runtime == "" || entry.Path == "" || entry.Symbol == "" || !neutralLocator(entry.Path) {
				return fmt.Errorf("adaptive parity snapshot: capability %q has invalid entry point", capability.ID)
			}
		}
		for _, modality := range append(slices.Clone(capability.Signature.Inputs), capability.Signature.Outputs...) {
			if !slices.Contains([]string{"text", "image", "audio", "video", "time-series", "table"}, modality) {
				return fmt.Errorf("adaptive parity snapshot: capability %q has invalid modality %q", capability.ID, modality)
			}
		}
		for kind, assets := range map[string][]Asset{"artifact": capability.Artifacts, "corpus": capability.Corpora, "golden": capability.Goldens} {
			if len(assets) == 0 {
				return fmt.Errorf("adaptive parity snapshot: capability %q lacks %s identity", capability.ID, kind)
			}
			for _, asset := range assets {
				if asset.Identity == "" || !neutralLocator(asset.Locator) || kind == "golden" && !validDigest(asset.SHA256) {
					return fmt.Errorf("adaptive parity snapshot: capability %q has invalid %s identity", capability.ID, kind)
				}
			}
		}
		for name, metric := range map[string]Metric{
			"cold_wall": capability.Performance.ColdWall, "warm_wall": capability.Performance.WarmWall,
			"peak_device": capability.Performance.PeakDevice, "peak_host": capability.Performance.PeakHost,
		} {
			if err := validateMetric(metric); err != nil {
				return fmt.Errorf("adaptive parity snapshot: capability %q %s: %w", capability.ID, name, err)
			}
		}
	}
	return nil
}

func validateMetric(metric Metric) error {
	switch metric.State {
	case "measured":
		if metric.Value < 0 || metric.Unit == "" || metric.Evidence == "" || metric.Reason != "" {
			return errors.New("measured metric needs non-negative value, unit, evidence, and no reason")
		}
	case "unmeasured":
		if metric.Reason == "" || metric.Value != 0 || metric.Unit != "" || metric.Evidence != "" {
			return errors.New("unmeasured metric needs only a reason")
		}
	default:
		return fmt.Errorf("invalid state %q", metric.State)
	}
	return nil
}

func validCommit(value string) bool { return len(value) == 40 && validHex(value) }
func validDigest(value string) bool { return len(value) == 64 && validHex(value) }

func validHex(value string) bool {
	_, err := hex.DecodeString(value)
	return err == nil
}

func neutralLocator(value string) bool {
	return value != "" && !strings.Contains(value, "\\") && !strings.HasPrefix(value, "/") &&
		!strings.Contains(value, "../") && !strings.Contains(value, ":/")
}
