package main

import (
	"flag"
	"fmt"

	"overgo/internal/gemma4convert"
)

// convertGemma4Command is gguf convert-gemma4: it converts a Gemma 4
// checkpoint directory to a language-model GGUF and a multimodal-projector
// GGUF.
func convertGemma4Command(flags *flag.FlagSet, args []string) error {
	model := flags.String("model", "", "language-model GGUF output")
	mmproj := flags.String("mmproj", "", "multimodal-projector GGUF output")
	mmprojF32 := flags.Bool("mmproj-f32", false, "store projector tensors as F32")
	name := flags.String("name", "", "GGUF model name")
	fp8Native := flags.Bool("fp8-native", false, "preserve fp8 mlp gate/up/down as native F8E4M3 tensors instead of up-converting to BF16")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 || (*model == "" && *mmproj == "") {
		return fmt.Errorf("usage: %s [-model model.gguf] [-mmproj mmproj.gguf] [-mmproj-f32] [-fp8-native] [-name name] checkpoint-directory", flags.Name())
	}
	report, err := gemma4convert.Convert(gemma4convert.Options{
		Directory: flags.Arg(0), ModelPath: *model, MMProjPath: *mmproj,
		MMProjF32: *mmprojF32, Name: *name, FP8Native: *fp8Native,
	})
	if err != nil {
		return err
	}
	fmt.Printf("wrote %d model tensors and %d projector tensors\n", report.ModelTensors, report.MMProjTensors)
	return nil
}
