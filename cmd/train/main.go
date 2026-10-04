// Command train executes native recipe-bound training; -route <name>, ahead
// of every other flag, runs a bounded family trainer (see trainRoutes).
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
	"overgo/internal/trainingprogram"
	"overgo/internal/trainingworkflow"
)

// defaultKeptCheckpoints is how many of the checkpoints -save-every writes the run
// keeps unless -keep-last names another number: the newest three.
const defaultKeptCheckpoints = 3

// denseDevice trains a dense-causal model on the device; the Windows build,
// which carries the CUDA worker, registers it.
var denseDevice trainingworkflow.DeviceTrainer

func main() {
	clioptions.MainNamed("train", run)
}

func run() error {
	// An interrupt ends every run cooperatively: a route's, the dense
	// workflow's and an objective run's training stops between updates.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if len(os.Args) > 1 && os.Args[1] == routeFlag {
		return runRoute(ctx, os.Args[2:])
	}
	model := flag.String("model", "", "model directory (safetensors + config.json + tokenizer.json)")
	dataset := flag.String("dataset", "", "UTF-8 training dataset")
	output := flag.String("out", "", "output directory for the trained checkpoint")
	resume := flag.String("resume", "", "resume checkpoint directory")
	reference := flag.String("reference", "", "frozen reference model directory for DPO")
	scale := clioptions.Float64Override(flag.CommandLine, "objective-scale", "RL objective scale")
	preferenceMode := flag.String("preference-mode", "", "DPO objective loss mode: empty for DPO's own, the reference-free simpo or orpo, or kto over unpaired completions (JSONL of id, group, prompt, completion and a boolean label, true for desirable)")
	preferenceMargin := clioptions.Float64Override(flag.CommandLine, "preference-margin", "SimPO target reward margin")
	steps := clioptions.IntOverride(flag.CommandLine, "steps", "Muon update steps; omitted derives from dataset units")
	maximumSequence := clioptions.IntOverride(flag.CommandLine, "seq", "maximum token sequence; omitted derives from model context")
	host := flag.Bool("host", false, "force host execution")
	freezeLexical := flag.Bool("freeze-lexical", false, "freeze tied embedding/head; requires CUDA resident training")
	maxWall := clioptions.DurationOverride(flag.CommandLine, "max-wall", "optional projected-wall bound")
	storePath := flag.String("store", "overgodb-store", "OvergoDB containing the active training recipe and policies")
	recipeID := flag.String("recipe", "", "active training recipe artifact ID")
	bootstrapRecipe := flag.String("bootstrap-recipe", "", "publish a token-training recipe for the model weights file at this path as a candidate, then exit; the first run under it that succeeds verifies and activates it")
	previewDataset := flag.String("preview-dataset", "", "validate one speech training example from a source/split/policy JSON manifest without loading a model or updating parameters")
	audioManifest := flag.String("audio-manifest", "", "CPU CTC adapter source-processing manifest; the active training recipe owns the registered dataset and split")
	objective := flag.String("objective", "", "registered text objective alias: train on its training split, publish the trained model at -out and the held-out verdict that binds it (table and forecast objectives train through -route tabular and -route seriesforecast)")
	proof := flag.Bool("proof", false, "with -objective: re-verify the objective's recipe by the training run alone, scoring no model and publishing no held-out verdict")
	loraRank := flag.Int("lora-rank", 0, "with -objective: train a LoRA adapter of this rank over the frozen model instead of the model, on the host; the adapter is written under -out/adapter as a Hugging Face PEFT directory")
	loraAlpha := flag.Float64("lora-alpha", 0, "with -lora-rank: the adapter's alpha, its scale being alpha over rank; omitted takes twice the rank")
	loraTargets := flag.String("lora-targets", "", "with -lora-rank: comma-separated projections to adapt (q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj); omitted adapts all of them")
	loraSeed := flag.Uint64("lora-seed", 0, "with -lora-rank: seed of the adapter's initialization")
	average := flag.Float64("average", 0, "with -objective: keep a moving average of the trained weights, or of an adapter's factors, at this decay in (0, 1) (0.99 is usual), warmed up so a short run is not mostly its start, and publish the average as the trained model; runs on the host")
	loraRSLoRA := flag.Bool("lora-rslora", false, "with -lora-rank: rank-stabilized LoRA, each module scaled by alpha over the root of its rank rather than over its rank")
	loraPlus := flag.Float64("lora-plus", 0, "with -lora-rank: LoRA+, the up factor learning this many times faster than the down factor (1 or more); omitted is plain LoRA")
	saveEvery := flag.Int("save-every", 0, "with -objective: save a checkpoint beside -out (out-step-000040) after every this many updates but the last, on the host")
	keepLast := flag.Int("keep-last", defaultKeptCheckpoints, "with -save-every: how many of the newest saved checkpoints to keep")
	resumeRun := flag.String("resume-run", "", "with -objective: continue from a checkpoint a run saved with -save-every, on the host; the run ends where an uninterrupted one would")
	loraFrom := flag.String("lora-from", "", "with -objective: continue training the PEFT adapter in this directory, which brings its own rank, alpha and targets")
	flag.Parse()
	if *proof && *objective == "" {
		return fmt.Errorf("train: -proof re-verifies an -objective recipe; it needs -objective")
	}
	lora := loraRequest(loraFlags{Rank: *loraRank, Alpha: *loraAlpha, Targets: *loraTargets, Seed: *loraSeed, From: *loraFrom, RSLoRA: *loraRSLoRA, Plus: *loraPlus})
	if lora == nil && (*loraAlpha != 0 || *loraTargets != "" || *loraSeed != 0 || *loraRSLoRA || *loraPlus != 0) {
		return fmt.Errorf("train: -lora-alpha, -lora-targets, -lora-seed, -lora-rslora and -lora-plus shape an adapter; they need -lora-rank or -lora-from")
	}
	if (*saveEvery != 0 || *resumeRun != "") && *objective == "" {
		return fmt.Errorf("train: saving checkpoints and resuming a run belong to an -objective run; -save-every and -resume-run need -objective")
	}
	if *average != 0 && *objective == "" {
		return fmt.Errorf("train: a moving average of the weights is kept under an -objective; -average needs -objective")
	}
	if lora != nil && *objective == "" {
		return fmt.Errorf("train: a LoRA adapter trains under an -objective; -lora-rank and -lora-from need -objective")
	}
	store, err := overgodb.Open(*storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	if *objective != "" {
		set := map[string]bool{}
		flag.Visit(func(given *flag.Flag) { set[given.Name] = true })
		return runObjectiveMode(ctx, store, objectiveRun{
			Alias: *objective, Recipe: *recipeID, Model: *model, Out: *output, Steps: *steps, Sequence: *maximumSequence, Host: *host,
			FreezeLexical: *freezeLexical, Proof: *proof, MaxWall: *maxWall, LoRA: lora, Average: *average, SaveEvery: *saveEvery, Keep: *keepLast, ResumeRun: *resumeRun,
		}, set, os.Stdout)
	}
	if *previewDataset != "" {
		if *bootstrapRecipe != "" || *recipeID != "" || *model != "" || *output != "" || *resume != "" || *audioManifest != "" {
			return fmt.Errorf("train: -preview-dataset cannot be combined with model execution or recipe bootstrap")
		}
		return previewTrainingData(ctx, store, *previewDataset, os.Stdout)
	}
	if *bootstrapRecipe != "" {
		if *audioManifest != "" {
			return fmt.Errorf("train: -audio-manifest cannot be combined with recipe bootstrap")
		}
		return bootstrapTrainingRecipe(store, *bootstrapRecipe, *dataset)
	}
	parsedRecipe, err := artifact.ParseID(*recipeID)
	if err != nil {
		return fmt.Errorf("training recipe: %w", err)
	}
	var audio *trainingworkflow.AudioTrainingSpec
	if *audioManifest != "" {
		data, err := artifact.ReadContentFile(*audioManifest)
		if err != nil {
			return err
		}
		audio = new(trainingworkflow.AudioTrainingSpec)
		if err := strictjson.DecodeBytes(data, audio); err != nil {
			return err
		}
	}

	request := trainingworkflow.Request{
		Audio:      audio,
		Repository: store, Recipe: parsedRecipe, Observations: store,
		ModelDirectory: *model, DatasetPath: *dataset, OutputDirectory: *output,
		ResumeDirectory: *resume, ReferenceDirectory: *reference,
		Steps: *steps, MaximumSequence: *maximumSequence,
		ObjectiveScale: *scale, Host: *host, Device: denseDevice, FreezeLexical: *freezeLexical,
		PreferenceMode: trainingprogram.PreferenceMode(*preferenceMode), PreferenceMargin: *preferenceMargin,
		Progress: os.Stderr, MaxProjectedWall: *maxWall,
	}
	result, err := trainingworkflow.Execute(ctx, request)
	if err != nil {
		return err
	}
	if result.Observation.Valid() {
		fmt.Printf("session observation: %s\n", result.Observation)
	}
	count := len(result.Losses) + len(result.DPO) + len(result.KTO) + len(result.GRPO)
	fmt.Printf("backend=%s objective=%s batches=%d stream_position=%d steps=%d recipe_lr=%g recipe_momentum=%g\n",
		result.Backend, result.Objective, count, result.StreamPosition, result.Plan.Updates(),
		result.Optimizer.BaseLearningRate, result.Optimizer.Momentum)
	for index, loss := range result.Losses {
		fmt.Printf("step %d: loss %.6f\n", index, loss)
	}
	for _, observation := range result.DPO {
		fmt.Printf("step %d: loss %.6f relative_margin %.6f\n", observation.Step, observation.Loss, observation.RelativeMargin)
	}
	for _, observation := range result.KTO {
		fmt.Printf("step %d: loss %.6f kl %.6f desirable_reward %.6f (%d) undesirable_reward %.6f (%d)\n", observation.Step, observation.Loss,
			observation.KL, observation.DesirableReward, observation.Desirable, observation.UndesirableReward, observation.Undesirable)
	}
	for _, observation := range result.GRPO {
		fmt.Printf("step %d: loss %.6f mean_reward %.6f\n", observation.Step, observation.Loss, observation.MeanReward)
	}
	fmt.Printf("checkpoint=%s written to %s\n", result.Checkpoint.ID(), *output)
	if result.Candidate.Valid() {
		fmt.Printf("candidate=%s (not activated)\n", result.Candidate)
	}
	return judgePreference(ctx, store, request, result, os.Stdout)
}

// judgePreference publishes the held-out verdict of a preference run from a
// base policy whose objective holds records out, and says when a run is not
// judged.
func judgePreference(ctx context.Context, store *overgodb.Store, request trainingworkflow.Request, result trainingworkflow.Result, output io.Writer) error {
	if result.Objective != trainingprogram.ObjectiveDPO {
		return nil
	}
	if request.ModelDirectory == "" {
		_, err := fmt.Fprintln(output, "held-out verdict: not judged; a resumed run names no base policy (-model)")
		return err
	}
	verdict, judged, err := trainingworkflow.JudgePreferenceHeldout(ctx, store, request, result)
	if err != nil || !judged {
		return err
	}
	return reportVerdict(output, verdict)
}
