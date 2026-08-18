// Command rxbrain-probe is the delivery vehicle for the RxBrain inference
// port. Stage 1: load and print the declared configuration contract from a
// real checkpoint, refusing drifted declarations. Later stages extend this
// command through tensor inventory, vision encode, and the vision-QA text
// path, keeping one owner tool for the whole port.
package main

import (
	"flag"
	"fmt"
	"os"

	"overgo/internal/rxbrain"
)

func main() {
	model := flag.String("model", "", "RxBrain checkpoint directory (config.json + sharded safetensors)")
	flag.Parse()
	if *model == "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: rxbrain-probe -model <checkpoint-dir>")
		os.Exit(2)
	}
	config, err := rxbrain.Load(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "rxbrain-probe:", err)
		os.Exit(1)
	}
	fmt.Printf("architecture=%v hidden=%d layers=%d heads=%d kv_heads=%d cla_share=%d vocab=%d\n",
		config.Architectures, config.HiddenSize, config.NumHiddenLayers,
		config.NumAttentionHeads, config.NumKeyValueHeads, config.ClaShareFactor, config.VocabSize)
	fmt.Printf("tokens: bos=%d eos=%d image_start=%d flow_latent=%d act=%s\n",
		config.BOSTokenID, config.EOSTokenID, config.ImageStartTokenID,
		config.FlowLatentPlaceholde, config.HiddenAct)
	fmt.Println("honesty: declaration contract only; inference stages land behind this tool")
}
