package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"overgo/internal/export"
	"overgo/internal/peft"
)

// mergeLoRACommand is gguf merge-lora: it merges a Hugging Face PEFT LoRA
// adapter into the safetensors checkpoint it was trained on, writing a
// checkpoint of the same tensors, shards and dtypes with the base's config and
// tokenizer files beside it.
func mergeLoRACommand(flags *flag.FlagSet, args []string) error {
	adapterDirectory := flags.String("adapter", "", "PEFT adapter directory (adapter_config.json and adapter_model.safetensors)")
	base := flags.String("base", "", "safetensors checkpoint directory the adapter was trained on")
	output := flags.String("output", "", "new checkpoint directory")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *adapterDirectory == "" || *base == "" || *output == "" || flags.NArg() != 0 {
		return fmt.Errorf("usage: %s -adapter adapter-directory -base checkpoint-directory -output merged-directory", flags.Name())
	}
	adapter, err := peft.Load(*adapterDirectory)
	if err != nil {
		return fmt.Errorf("%s: %w", flags.Name(), err)
	}
	report, err := peft.Merge(adapter, *base, *output)
	if err != nil {
		return fmt.Errorf("%s: %w", flags.Name(), err)
	}
	entries, err := os.ReadDir(*base)
	if err != nil {
		return err
	}
	var companions []string
	for _, entry := range entries {
		if name := entry.Name(); entry.Type().IsRegular() && !strings.HasSuffix(name, ".safetensors") && !strings.HasSuffix(name, ".safetensors.index.json") {
			companions = append(companions, name)
		}
	}
	if len(companions) > 0 {
		if err := export.CopyFiles(*base, *output, companions); err != nil {
			return fmt.Errorf("%s: %w", flags.Name(), err)
		}
	}
	fmt.Printf("merged %d of %d tensors from %d adapter modules; copied %d companion files\n", report.Merged, report.Tensors, len(adapter.Modules), len(companions))
	return nil
}
