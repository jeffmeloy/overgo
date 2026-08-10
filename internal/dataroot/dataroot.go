// Package dataroot is the single owner of where the data roots live (floor
// component 9): the RepoDB store and the three bulk roots — models (vendor
// inputs, irreplaceable), datasets (training data), checkpoints (trained
// outputs, lineage-bearing). Every command resolves through here so discovery
// cannot drift per-tool; explicit flags stay authoritative overrides.
//
// Resolution order:
//  1. OVERGO_DATA_ROOT env — one base directory holding repodb-store/,
//     models/, datasets/, checkpoints/.
//  2. local-models.json in the working directory — explicit per-root paths,
//     which is how overgo points at data that lives in another repository's
//     home without moving a byte (data never moves; code comes to the data).
//  3. Working-directory defaults (repodb-store, models, datasets,
//     checkpoints) — the pre-contract behavior, preserved.
package dataroot

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	// Env names the base directory for all roots (resolution step 1).
	Env = "OVERGO_DATA_ROOT"
	// ConfigFile is the per-checkout root override file (resolution step 2);
	// gitignored, machine-local by design.
	ConfigFile = "local-models.json"
)

// Roots: resolved data locations. Source names which resolution step won.
type Roots struct {
	Store       string `json:"store"`
	Models      string `json:"models"`
	Datasets    string `json:"datasets"`
	Checkpoints string `json:"checkpoints"`
	Source      string `json:"-"`
}

// Resolve returns the data roots for the given working directory.
func Resolve(workingDirectory string) (Roots, error) {
	if base := strings.TrimSpace(os.Getenv(Env)); base != "" {
		info, err := os.Stat(base)
		if err != nil || !info.IsDir() {
			return Roots{}, fmt.Errorf("dataroot: %s=%q is not a directory", Env, base)
		}
		return Roots{
			Store:       filepath.Join(base, "repodb-store"),
			Models:      filepath.Join(base, "models"),
			Datasets:    filepath.Join(base, "datasets"),
			Checkpoints: filepath.Join(base, "checkpoints"),
			Source:      Env,
		}, nil
	}
	configPath := filepath.Join(workingDirectory, ConfigFile)
	if raw, err := os.ReadFile(configPath); err == nil {
		var roots Roots
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&roots); err != nil {
			return Roots{}, fmt.Errorf("dataroot: parse %s: %w", ConfigFile, err)
		}
		defaults := fallback(workingDirectory)
		if roots.Store == "" {
			roots.Store = defaults.Store
		}
		if roots.Models == "" {
			roots.Models = defaults.Models
		}
		if roots.Datasets == "" {
			roots.Datasets = defaults.Datasets
		}
		if roots.Checkpoints == "" {
			roots.Checkpoints = defaults.Checkpoints
		}
		// Config values arrive in whatever separator style the user wrote;
		// normalize to host form so every downstream Join/Stat agrees.
		roots.Store = filepath.Clean(filepath.FromSlash(roots.Store))
		roots.Models = filepath.Clean(filepath.FromSlash(roots.Models))
		roots.Datasets = filepath.Clean(filepath.FromSlash(roots.Datasets))
		roots.Checkpoints = filepath.Clean(filepath.FromSlash(roots.Checkpoints))
		roots.Source = ConfigFile
		return roots, nil
	}
	return fallback(workingDirectory), nil
}

func fallback(workingDirectory string) Roots {
	return Roots{
		Store:       filepath.Join(workingDirectory, "repodb-store"),
		Models:      filepath.Join(workingDirectory, "models"),
		Datasets:    filepath.Join(workingDirectory, "datasets"),
		Checkpoints: filepath.Join(workingDirectory, "checkpoints"),
		Source:      "defaults",
	}
}

// ResolveModelPath maps a model reference to a filesystem path: an existing
// path (absolute or working-directory-relative) wins unchanged; otherwise a
// bare reference resolves under the models root, then the checkpoints root
// (trained outputs are servable artifacts too). The original reference
// returns unchanged when nothing resolves, so the caller's open error names
// what the user typed.
func (r Roots) ResolveModelPath(reference string) string {
	reference = strings.TrimSpace(reference)
	if reference == "" {
		return reference
	}
	if _, err := os.Stat(reference); err == nil {
		return reference
	}
	if filepath.IsAbs(reference) {
		return reference
	}
	for _, root := range []string{r.Models, r.Checkpoints} {
		candidate := filepath.Join(root, reference)
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return reference
}
