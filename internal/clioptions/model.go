package clioptions

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"slices"
	"strings"

	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/repodb"
)

// ModelFlags: common model-loading flags.
type ModelFlags struct {
	DeviceOrdinal *int
	Repository    *string
	loraPaths     stringList
}

type stringList []string

func (values *stringList) String() string {
	return strings.Join(*values, ",")
}

func (values *stringList) Set(value string) error {
	if strings.TrimSpace(value) == "" {
		return errors.New("LoRA path is empty")
	}
	*values = append(*values, value)
	return nil
}

// AddModelFlags: common model-loading flag registration.
func AddModelFlags(flags *flag.FlagSet, loraHelp string) *ModelFlags {
	result := &ModelFlags{}
	result.DeviceOrdinal = flags.Int("device", 0, "CUDA device ordinal")
	result.Repository = flags.String("repo", "", "RepoDB containing the active model recipe; empty resolves via the data-root contract (OVERGO_DATA_ROOT, local-models.json, or ./repodb-store)")
	flags.Var(&result.loraPaths, "lora", loraHelp)
	return result
}

// OpenRunner: resolves the active identity-bound recipe before inference.
func (flags *ModelFlags) OpenRunner(
	ctx context.Context,
	path string,
	loraScale float32,
) (*inference.Runner, error) {
	if flags == nil || flags.Repository == nil {
		return nil, errors.New("model recipe repository is required")
	}
	repository := strings.TrimSpace(*flags.Repository)
	// The data-root contract owns defaults; an explicit -repo flag stays
	// authoritative. Bare model references resolve under the models and
	// checkpoints roots so discovery cannot drift per-tool.
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return nil, err
	}
	if repository == "" {
		repository = roots.Store
		if _, err := os.Stat(repository); err != nil {
			return nil, fmt.Errorf("model recipe repository is required: no -repo flag and no store at %s (%s)", repository, roots.Source)
		}
	}
	return OpenRunner(ctx, repository, roots.ResolveModelPath(path), flags.OpenOptions(loraScale))
}

// OpenRunner: storage-bound assembly; inference receives only a compiled program.
func OpenRunner(
	ctx context.Context,
	repository, path string,
	options inference.OpenOptions,
) (*inference.Runner, error) {
	if strings.TrimSpace(repository) == "" {
		return nil, errors.New("model recipe repository is required")
	}
	// Serving only READS the store (recipe resolution); a writer open here
	// starved concurrent serves and self-deadlocked the smoke lane against
	// its own child process.
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return nil, fmt.Errorf("open model recipe repository: %w", err)
	}
	loaded, resolveErr := modelrecipe.ResolveActiveGGUF(ctx, store, path)
	closeErr := store.Close()
	if resolveErr != nil || closeErr != nil {
		_ = loaded.Close()
		return nil, errors.Join(resolveErr, closeErr)
	}
	return inference.OpenWithProgram(&loaded, options)
}

// OpenOptions: inference model-loading options.
func (flags *ModelFlags) OpenOptions(loraScale float32) inference.OpenOptions {
	return BuildOpenOptions(*flags.DeviceOrdinal, flags.loraPaths, loraScale)
}

// LoRAPaths: copied adapter paths
func (flags *ModelFlags) LoRAPaths() []string {
	return slices.Clone(flags.loraPaths)
}

// BuildOpenOptions: common inference open options
func BuildOpenOptions(device int, loraPaths []string, loraScale float32) inference.OpenOptions {
	adapters := make([]inference.LoRAConfig, len(loraPaths))
	for index, path := range loraPaths {
		adapters[index] = inference.LoRAConfig{Path: path, Scale: loraScale}
	}
	return inference.OpenOptions{
		DeviceOrdinal: device, LoRAAdapters: adapters,
	}
}
