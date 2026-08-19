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
	bindText := flag.Bool("bind-text", false, "bind the base-text decode branch from the shards and report the promotion")
	forwardCheck := flag.Bool("forward-check", false, "run the host text forward over a fixed token sequence and report the greedy terminal")
	bindVision := flag.Bool("bind-vision", false, "bind the visual tower from the shards and report the promotion")
	ropeAlpha := flag.Float64("rope-alpha", 1000.0, "dynamic NTK-alpha rope scaling (checkpoint declares 1000)")
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
	if *bindText || *forwardCheck {
		weights, err := rxbrain.LoadTextWeights(*model, config)
		if err != nil {
			fmt.Fprintln(os.Stderr, "rxbrain-probe:", err)
			os.Exit(1)
		}
		parameters := len(weights.Embed) + len(weights.FinalNorm)
		for _, layer := range weights.Layers {
			parameters += len(layer.InputLN) + len(layer.PostLN) + len(layer.QProj) + len(layer.KProj) +
				len(layer.VProj) + len(layer.OProj) + len(layer.QueryLN) + len(layer.KeyLN) +
				len(layer.GateProj) + len(layer.UpProj) + len(layer.DownProj)
		}
		fmt.Printf("text branch bound: layers=%d parameters=%d (tied head)\n", len(weights.Layers), parameters)
		if *forwardCheck {
			tokens := []int{config.BOSTokenID, 3000, 4000, 5000}
			hidden, err := rxbrain.ForwardText(config, weights, tokens, *ropeAlpha)
			if err != nil {
				fmt.Fprintln(os.Stderr, "rxbrain-probe:", err)
				os.Exit(1)
			}
			next, score := rxbrain.GreedyNextToken(config, weights, hidden)
			fmt.Printf("forward check: %d tokens -> greedy next=%d score=%.3f (ntk-alpha rope inv[1]=%.6g)\n",
				len(tokens), next, score, rxbrain.RopeInvFreq(config, *ropeAlpha)[1])
		}
	}
	if *bindVision {
		vision, err := rxbrain.LoadVisionWeights(*model, config)
		if err != nil {
			fmt.Fprintln(os.Stderr, "rxbrain-probe:", err)
			os.Exit(1)
		}
		parameters := len(vision.PatchEmbedW) + len(vision.PatchEmbedB) + len(vision.PosEmbed) +
			len(vision.MergerProj1W) + len(vision.MergerProj1B) + len(vision.MergerProj2W) + len(vision.MergerProj2B) +
			len(vision.MergerPool0W) + len(vision.MergerPool0B) + len(vision.MergerPool2W) + len(vision.MergerPool2B)
		for _, block := range vision.Blocks {
			parameters += len(block.Norm1W) + len(block.Norm1B) + len(block.QKVW) + len(block.QKVB) +
				len(block.ProjW) + len(block.ProjB) + len(block.Norm2W) + len(block.Norm2B) +
				len(block.FC1W) + len(block.FC1B) + len(block.FC2W) + len(block.FC2B)
		}
		fmt.Printf("visual tower bound: blocks=%d parameters=%d (merger out=%d)\n",
			len(vision.Blocks), parameters, config.HiddenSize)
	}
	fmt.Println("honesty: text-branch contracts, host forward and tower binding only; vision encode and generation land behind this tool")
}
