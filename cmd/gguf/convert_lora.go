package main

import (
	"flag"
	"fmt"

	"overgo/internal/hfconvert"
	"overgo/internal/peft"
)

// convertLoRACommand is gguf convert-lora: it converts a Hugging Face PEFT LoRA
// adapter to the GGUF adapter the server loads with -lora, naming each pair by
// the GGUF weight it adapts through the checkpoint the adapter was trained on.
func convertLoRACommand(flags *flag.FlagSet, args []string) error {
	adapterDirectory := flags.String("adapter", "", "PEFT adapter directory (adapter_config.json and adapter_model.safetensors)")
	base := flags.String("base", "", "HF checkpoint directory the adapter was trained on; its config.json names the architecture")
	output := flags.String("output", "", "new GGUF adapter output path")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *adapterDirectory == "" || *base == "" || *output == "" || flags.NArg() != 0 {
		return fmt.Errorf("usage: %s -adapter adapter-directory -base checkpoint-directory -output adapter.gguf", flags.Name())
	}
	adapter, err := peft.Load(*adapterDirectory)
	if err != nil {
		return fmt.Errorf("%s: %w", flags.Name(), err)
	}
	report, err := hfconvert.ConvertLoRA(adapter, hfconvert.LoRAOptions{Directory: *base, OutputPath: *output})
	if err != nil {
		return fmt.Errorf("%s: %w", flags.Name(), err)
	}
	fmt.Printf("wrote %d adapter modules as %d tensors, %d bytes\n", report.Modules, report.Tensors, report.OutputBytes)
	return nil
}
