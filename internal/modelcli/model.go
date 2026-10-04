// Package modelcli binds command flags and recipe storage to model opening.
package modelcli

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/inference"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/tensor"
)

// ModelFlags holds common model-loading flags.
type ModelFlags struct {
	DeviceOrdinal *int
	Repository    *string
	loraPaths     stringList
}

type stringList []string

// String joins adapter paths for flag help.
func (values *stringList) String() string {
	return strings.Join(*values, ",")
}

// Set appends a nonempty adapter path.
func (values *stringList) Set(value string) error {
	if strings.TrimSpace(value) == "" {
		return errors.New("LoRA path is empty")
	}
	*values = append(*values, value)
	return nil
}

// AddModelFlags registers common model-loading flags.
func AddModelFlags(flags *flag.FlagSet, loraHelp string) *ModelFlags {
	result := &ModelFlags{}
	result.DeviceOrdinal = flags.Int("device", 0, "CUDA device ordinal")
	result.Repository = flags.String("repo", "", "OvergoDB containing the active model recipe; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./overgodb-store)")
	flags.Var(&result.loraPaths, "lora", loraHelp)
	return result
}

// OpenRunner resolves the active identity-bound recipe before inference.
func (flags *ModelFlags) OpenRunner(
	ctx context.Context,
	path string,
) (*inference.Runner, error) {
	return flags.OpenRunnerWithOptions(ctx, path, flags.OpenOptions(tensor.UnitScale))
}

// OpenRunnerWithOptions resolves the repository and model from flags.
func (flags *ModelFlags) OpenRunnerWithOptions(
	ctx context.Context,
	path string,
	options inference.OpenOptions,
) (*inference.Runner, error) {
	if flags == nil || flags.Repository == nil {
		return nil, errors.New("model recipe repository is required")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return nil, err
	}
	repository, err := flags.repositoryPath(roots)
	if err != nil {
		return nil, err
	}
	return OpenRunner(ctx, repository, roots.ResolveModelPath(path), options)
}

// RepositoryPath returns the exact store selected by model-loading flags.
func (flags *ModelFlags) RepositoryPath() (string, error) {
	if flags == nil || flags.Repository == nil {
		return "", errors.New("model recipe repository is required")
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return "", err
	}
	return flags.repositoryPath(roots)
}

func (flags *ModelFlags) repositoryPath(roots dataroot.Roots) (string, error) {
	repository := strings.TrimSpace(*flags.Repository)
	if repository != "" {
		return repository, nil
	}
	if _, err := os.Stat(roots.Store); err != nil {
		return "", fmt.Errorf("model recipe repository is required: no -repo flag and no store at %s (%s)", roots.Store, roots.Source)
	}
	return roots.Store, nil
}

// OpenRunner resolves storage inputs and opens the compiled inference program.
// A weights file the store's identity memo remembers is not hashed before the
// open: its remembered identity finds the recipe, and the open proves that
// identity against the bytes it uploads, in the same pass.
func OpenRunner(
	ctx context.Context,
	repository, path string,
	options inference.OpenOptions,
) (*inference.Runner, error) {
	if strings.TrimSpace(repository) == "" {
		return nil, errors.New("model recipe repository is required")
	}
	runner, err := openRunner(ctx, repository, path, options, true)
	if errors.Is(err, artifact.ErrContentMismatch) {
		// The memo was stale: the file changed without a new size or modification
		// time, and nothing was served. Resolve it again by hashing, under the
		// identity its bytes have now.
		fmt.Fprintf(os.Stderr, "model %s: its remembered identity does not match its bytes; hashing the file\n", path)
		return openRunner(ctx, repository, path, options, false)
	}
	return runner, err
}

func openRunner(
	ctx context.Context,
	repository, path string,
	options inference.OpenOptions,
	remembered bool,
) (*inference.Runner, error) {
	// Serving only READS the store (recipe resolution); a writer open here
	// starved concurrent serves and self-deadlocked the smoke lane against
	// its own child process.
	store, err := overgodb.OpenReadOnly(repository)
	if err != nil {
		return nil, fmt.Errorf("open model recipe repository: %w", err)
	}
	var claimer modelartifact.Claimer
	if remembered {
		claimer = discovery.LoadMemo(ctx, store).Claim
	}
	loaded, resolveErr := modelrecipe.ResolveClaimedGGUF(ctx, store, path, claimer)
	closeErr := store.Close()
	if resolveErr != nil || closeErr != nil {
		_ = loaded.Close()
		return nil, errors.Join(resolveErr, closeErr)
	}
	return inference.OpenWithProgram(ctx, &loaded, options)
}

// OpenOptions returns inference model-loading options.
func (flags *ModelFlags) OpenOptions(loraScale float32) inference.OpenOptions {
	return BuildOpenOptions(*flags.DeviceOrdinal, flags.loraPaths, loraScale)
}

// LoRAPaths returns a copy of the adapter paths.
func (flags *ModelFlags) LoRAPaths() []string {
	return slices.Clone(flags.loraPaths)
}

// BuildOpenOptions builds common inference open options.
func BuildOpenOptions(device int, loraPaths []string, loraScale float32) inference.OpenOptions {
	adapters := make([]inference.LoRAConfig, len(loraPaths))
	for index, path := range loraPaths {
		adapters[index] = inference.LoRAConfig{Path: path, Scale: loraScale}
	}
	return inference.OpenOptions{
		DeviceOrdinal: device, LoRAAdapters: adapters,
	}
}
