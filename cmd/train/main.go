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
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
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
	scale := clioptions.Float64Override(flag.CommandLine, "objective-scale", "preference objective scale (DPO and its modes); the reward objective weighs a group by the order of its answers and takes none")
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
	loraQuantized := flag.Bool("lora-quantized", false, "with -lora-rank and a GGUF -model: keep each adapted projection as the file stores it, quantized, dequantizing it as its layer reads it, so the run needs far less memory; the model published is then the base as stored, with the adapter in its adapter directory")
	loraFrom := flag.String("lora-from", "", "with -objective: continue training the PEFT adapter in this directory, which brings its own rank, alpha and targets")
	orderSeed := flag.Uint64("order-seed", 0, "with -objective: read the training records in the order this seed draws rather than in membership order, so a run of fewer updates than the text holds windows trains on other text for each seed")
	runs := flag.Int("runs", 0, "with -objective: train this many runs from the same base, their order seeds counting up from -order-seed (from 1 without it), each published under -out in a directory named after its seed, and publish them as a change corpus, one line a run")
	propose := flag.String("propose", "", "change corpus artifact ID: propose a change from the corpus's runs under -out, fitted on the development part of the held-out pool -pool names, write it and its controls under -out/proposal, score each on the pool's selection part, each document read as far as the pool's acceptance fixed, and publish the judgment; needs -model, the corpus's base, and takes no -seq")
	pool := flag.String("pool", "", "with -propose: the acceptance of the objective's held-out pool, as training-objective -pool wrote it")
	confirm := flag.String("confirm", "", "change judgment artifact ID: have the promotion part of the judgment's pool say whether -candidate is a gain over the base, each document read as far as the pool's acceptance fixed; charges one confirmation to what the part's documents have left and writes the verdict and what is left; needs -model and -out as -propose had them, and takes no -seq")
	candidate := flag.String("candidate", "proposed", "with -confirm: the candidate to confirm, by the name its judgment gives it")
	proposal := flag.String("proposal", proposalDirectory, "with -propose and -confirm: the directory under -out the candidates are written in, and read from")
	evidence := flag.Bool("evidence", false, "with -propose: show the proposer the held-out verdicts the store holds of what was tried before it; it fits the gains they recorded for its corpus's runs beside the gains it reads on the pool, and scores the proposal it would have made without them beside its own")
	evidenceThrough := flag.Uint64("evidence-to-sequence", 0, "with -evidence: read the store as it stood at this journal sequence and not as it stands, as overgodb query -verdicts -to-sequence lists it")
	evidenceBytes := flag.Uint64("evidence-max-bytes", 0, "with -evidence: refuse a listing whose verdict documents exceed this many bytes (0 for no bound)")
	pilot := flag.String("pilot", "", "instruction acceptance artifact ID, as training-objective -pilot wrote it: train the pilot's reward arm and its control from -model under -out to each checkpoint its training specification names, freeze one checkpoint an arm on the selection groups, charge and read each comparison the acceptance named on the promotion groups, and say whether the reward arm is adopted")
	keep := flag.String(keepFlag, "", "with -pilot: keep the checkpoints the pilot wrote, for the reason given, which the record states; without it they are removed once the record is written under -out, unless the reward arm is adopted")
	flag.Parse()
	given := false
	flag.Visit(func(set *flag.Flag) { given = given || set.Name == keepFlag })
	if err := keepReason(given, *pilot, *keep); err != nil {
		return err
	}
	if (*evidence || *evidenceThrough != 0 || *evidenceBytes != 0) && (*propose == "" || !*evidence) {
		return fmt.Errorf("train: -evidence shows a proposer what was tried before it, and -evidence-to-sequence and -evidence-max-bytes bound what it is shown; they need -propose and -evidence")
	}
	if *proof && *objective == "" {
		return fmt.Errorf("train: -proof re-verifies an -objective recipe; it needs -objective")
	}
	lora := loraRequest(loraFlags{Rank: *loraRank, Alpha: *loraAlpha, Targets: *loraTargets, Seed: *loraSeed, From: *loraFrom, RSLoRA: *loraRSLoRA, Plus: *loraPlus, Quantized: *loraQuantized})
	if lora == nil && (*loraAlpha != 0 || *loraTargets != "" || *loraSeed != 0 || *loraRSLoRA || *loraPlus != 0 || *loraQuantized) {
		return fmt.Errorf("train: -lora-alpha, -lora-targets, -lora-seed, -lora-rslora, -lora-plus and -lora-quantized shape an adapter; they need -lora-rank or -lora-from")
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
	if (*orderSeed != 0 || *runs != 0) && *objective == "" {
		return fmt.Errorf("train: an order seed and a corpus of runs belong to an -objective run; -order-seed and -runs need -objective")
	}
	if *pilot != "" && *host {
		return fmt.Errorf("train: -pilot trains its arms through the weighted step, which the device takes; the host has no reward trainer, so -pilot takes no -host")
	}
	// A command a server starts opens the store the server writes to, so it
	// waits its turn for the writer lock where a plain open would fail at once.
	store, err := overgodb.OpenContext(ctx, *storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	if *output != "" && (*pilot != "" || *runs > 0 || *propose != "" || *confirm != "") {
		held, err := holdOutput(*output)
		if err != nil {
			return err
		}
		defer held.Close()
	}
	if *pilot != "" {
		return runPilot(ctx, store, *pilot, *model, *output, *keep, os.Stdout)
	}
	if *propose != "" || *confirm != "" {
		if *objective != "" || *model == "" || *output == "" || *propose != "" && *confirm != "" || *maximumSequence != 0 {
			return fmt.Errorf("train: -propose reads a corpus and -confirm a judgment, each of which names its own objective; one of them at a time, with -model and -out, no -objective, and no -seq: the pool's acceptance fixed how far each document is read")
		}
		if *confirm != "" {
			return confirmChange(ctx, store, *confirm, *candidate, *model, *output, *proposal, os.Stdout)
		}
		return proposeChange(ctx, store, *propose, *pool, *model, *output, *proposal, shownEvidence{shown: *evidence, through: *evidenceThrough, bytes: *evidenceBytes}, os.Stdout)
	}
	if *objective != "" {
		set := map[string]bool{}
		flag.Visit(func(given *flag.Flag) { set[given.Name] = true })
		return runObjectiveMode(ctx, store, objectiveRun{
			Alias: *objective, Recipe: *recipeID, Model: *model, Out: *output, Steps: *steps, Sequence: *maximumSequence, Host: *host,
			FreezeLexical: *freezeLexical, Proof: *proof, MaxWall: *maxWall, LoRA: lora, Average: *average, SaveEvery: *saveEvery, Keep: *keepLast, ResumeRun: *resumeRun,
			Order: *orderSeed, Runs: *runs,
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

// keepFlag names the flag that gives the reason a pilot's checkpoints are kept.
const keepFlag = "keep-checkpoints"

// keepReason refuses the keep flag where it would be misread: without
// -pilot, the one run that removes its checkpoints, and with a blank reason,
// because a flag given with nothing after it must never read as no flag and
// let the checkpoints go.
func keepReason(given bool, pilot, reason string) error {
	if given && (pilot == "" || strings.TrimSpace(reason) == "") {
		return fmt.Errorf("train: -%s gives the reason the pilot's checkpoints are kept, which its record states; it needs -pilot and a reason that is not blank", keepFlag)
	}
	return nil
}

// holdOutput makes the output directory of a run that writes many things
// under it, a pilot, a corpus, a proposal or a confirmation, and locks
// processlock.OutputLock there until the lock it returns is closed. A second
// run into the directory is refused while the first goes. The lock is also
// for a command removing what a finished run left to tell a live run by.
func holdOutput(out string) (*processlock.Lock, error) {
	if err := os.MkdirAll(out, clioptions.OutputDirectoryMode); err != nil {
		return nil, err
	}
	lock, err := processlock.Acquire(filepath.Join(out, processlock.OutputLock), clioptions.PrivateFileMode)
	if err != nil {
		return nil, fmt.Errorf("train: %s is held by a run that is still going, or its lock cannot be taken: %w", out, err)
	}
	return lock, nil
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
