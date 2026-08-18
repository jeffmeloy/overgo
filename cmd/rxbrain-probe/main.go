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
	inventory, err := rxbrain.LoadInventory(*model, config)
	if err != nil {
		fmt.Fprintln(os.Stderr, "rxbrain-probe:", err)
		os.Exit(1)
	}
	fmt.Printf("inventory: tensors=%d text=%d vision=%d generation=%d visual_tower=%d flow=%d shared=%d\n",
		len(inventory.TensorShard), inventory.TextBranch, inventory.VisionBranch,
		inventory.GenerationBranch, inventory.VisualTower, inventory.FlowAdapters, inventory.Shared)
	fmt.Println("honesty: declaration and inventory contracts only; decode stages land behind this tool")
}
