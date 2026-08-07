package clioptions

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"slices"
	"strings"

	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/repodb"
)

// ModelFlags: common model-loading flags.
type ModelFlags struct {
	DeviceOrdinal *int
	Preload       *bool
	NativeQ8      *bool
	NativeQuant   *bool
	HostCache     *bool
	Repository    *string
	loraPaths     stringList
}

// ModelFlagConfig: flag names and defaults
type ModelFlagConfig struct {
	PreloadName        string
	PreloadDefault     bool
	NativeQuantName    string
	NativeQuantDefault bool
	NativeQ8Name       string
	NativeQ8Default    bool
	HostCacheName      string
	HostCacheDefault   bool
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
	return AddModelFlagsWithConfig(flags, loraHelp, ModelFlagConfig{
		PreloadName: "preload", NativeQuantName: "native-quant", NativeQ8Name: "native-q8",
		HostCacheName: "host-cache",
	})
}

// AddModelFlagsWithConfig: configured model flag registration
func AddModelFlagsWithConfig(flags *flag.FlagSet, loraHelp string, config ModelFlagConfig) *ModelFlags {
	result := &ModelFlags{}
	result.DeviceOrdinal = flags.Int("device", 0, "CUDA device ordinal")
	result.Repository = flags.String("repo", "", "required RepoDB containing the active model recipe")
	if config.PreloadName != "" {
		result.Preload = flags.Bool(config.PreloadName, config.PreloadDefault, "dequantize all model weights once into CUDA memory")
	}
	if config.NativeQ8Name != "" {
		result.NativeQ8 = flags.Bool(config.NativeQ8Name, config.NativeQ8Default, "preload Q8_0 weights without dequantizing them")
	}
	if config.NativeQuantName != "" {
		result.NativeQuant = flags.Bool(config.NativeQuantName, config.NativeQuantDefault, "preload supported quantized weights without dequantizing them")
	}
	if config.HostCacheName != "" {
		result.HostCache = flags.Bool(
			config.HostCacheName,
			config.HostCacheDefault,
			"retain lazily dequantized F32 weights in host memory",
		)
	}
	flags.Var(&result.loraPaths, "lora", loraHelp)
	return result
}

// OpenRunner: resolves the active identity-bound recipe before inference.
func (flags *ModelFlags) OpenRunner(
	ctx context.Context,
	path string,
	loraScale float32,
) (*inference.Runner, error) {
	if flags == nil || flags.Repository == nil || strings.TrimSpace(*flags.Repository) == "" {
		return nil, errors.New("model recipe repository is required")
	}
	return OpenRunner(ctx, *flags.Repository, path, flags.OpenOptions(loraScale))
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
	store, err := repodb.Open(repository)
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
	preload := flags.Preload != nil && *flags.Preload
	native := flags.NativeQ8 != nil && *flags.NativeQ8 || flags.NativeQuant != nil && *flags.NativeQuant
	result := BuildOpenOptions(*flags.DeviceOrdinal, preload, native, flags.loraPaths, loraScale)
	result.CacheHostWeights = flags.HostCache != nil && *flags.HostCache
	return result
}

// LoRAPaths: copied adapter paths
func (flags *ModelFlags) LoRAPaths() []string {
	return slices.Clone(flags.loraPaths)
}

// BuildOpenOptions: common inference open options
func BuildOpenOptions(device int, preload, nativeQuant bool, loraPaths []string, loraScale float32) inference.OpenOptions {
	adapters := make([]inference.LoRAConfig, len(loraPaths))
	for index, path := range loraPaths {
		adapters[index] = inference.LoRAConfig{Path: path, Scale: loraScale}
	}
	return inference.OpenOptions{
		DeviceOrdinal: device, PreloadDeviceWeights: preload,
		PreloadQuantizedWeights: nativeQuant, LoRAAdapters: adapters,
	}
}
