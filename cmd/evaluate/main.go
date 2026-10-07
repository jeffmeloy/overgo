// Command evaluate runs benchmark suites against local models: a
// hand-written manifest names models and suite files, or -all derives
// both sides from the store -- the servable model listing and the
// suites compiled from the active benchmark catalog -- and campaigns
// each model in its own worker process.
package main

import (
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime/pprof"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
)

type manifest struct {
	Repository string         `json:"repository"`
	Catalog    string         `json:"catalog"`
	CodeCommit string         `json:"code_commit"`
	Device     int            `json:"device"`
	Models     []modelRequest `json:"models"`
	// ChatProtocol scores multiple-choice prompts through each model's
	// declared chat template instead of the raw base-style completion.
	// The raw protocol stays the recorded anchor: template presence does
	// not distinguish base from instruct artifacts, so the shaped
	// protocol is an explicit measurement choice.
	ChatProtocol bool `json:"chat_protocol,omitzero"`
}

type modelRequest struct {
	Path   string   `json:"path"`
	Suites []string `json:"suites"`
	// Judge names the speech recognizer directory that listens to a speech model's
	// recordings and writes down what it heard.
	Judge string `json:"judge,omitempty"`
	// Task names the capability a model evaluates under when its path alone
	// does not: seq2seq opens an encoder-decoder model under its active
	// seq2seq recipe.
	Task string `json:"task,omitempty"`
	// Projector evaluates the model under its active projection recipe with
	// this multimodal projector, its cases asking about videos of MediaRoot.
	Projector string `json:"projector,omitempty"`
	MediaRoot string `json:"media_root,omitempty"`
}

type workerLauncher func(context.Context, int) error

func main() {
	clioptions.Main(run)
}

func run() error {
	manifestPath := flag.String("manifest", "", "evaluation manifest")
	worker := flag.Bool("worker", false, "run one model worker")
	modelIndex := flag.Int("model-index", -1, "worker model index")
	modelPath := flag.String("model-path", "", "worker model path for -all (the parent lists the catalog once and hands each worker its model), or the image model -render-tifa draws with")
	budget := flag.Duration("budget", evaluationBudget, "wall-clock ceiling on one -all pass, shared as equal slices across the models still to run; a model cut at its slice is recorded budget-exceeded")
	importCache := flag.String("import-hf-cache", "", "scan a HuggingFace dataset cache root, import every recognized benchmark, and publish the active catalog")
	listSuites := flag.Bool("list-derived-suites", false, "compile the store's benchmark catalog into suites and list their descriptors")
	repository := flag.String("repo", "overgodb-store", "OvergoDB root for -import-hf-cache, -list-derived-suites, and -all")
	allModels := flag.Bool("all", false, "evaluate every servable local model against the store's derived suites")
	chatProtocol := flag.Bool("chat-protocol", false, "score multiple-choice suites through each model's declared chat template; the raw completion protocol stays the recorded anchor")
	device := flag.Int("device", 0, "CUDA device ordinal for -all")
	family := flag.String("family", "", "restrict -all and -list-derived-suites to one derived suite family (e.g. mmlu); only that family's records are read and compiled")
	replay := flag.String("replay", "", "for an -all worker: an evaluation record of the family's suite to carry forward when a replay of a sample of its cases reproduces their outputs; the suite runs in full otherwise")
	catalogLimit := flag.Int("catalog-limit", 256, "servable model listing bound for -all")
	declareDomains := flag.String("declare-domain", "", "comma-separated eval domains to declare for the positional model path (e.g. dna)")
	declareReferences := flag.String("declare-references", "", "JSON spec of published or externally measured reference scores per model location ({declarations:[{model, references:[{suite, metric, value, protocol, source}]}]})")
	transcriptionManifest := flag.String("transcription-manifest", "", "score stored transcription runs from separate suite and prediction files")
	alignmentManifest := flag.String("alignment-manifest", "", "execute and score explicitly conditioned CPU CTC word alignment")
	diarizationManifest := flag.String("diarization-manifest", "", "execute CPU speaker attribution and score every declared source interval")
	transcriptionResourceManifest := flag.String("transcription-resource-manifest", "", "execute a pinned native speech or projected transcription corpus from clean source")
	importDNA := flag.String("import-dna-corpus", "", "import a bounded slice of every parquet subset under this corpus root and merge the entries into the active benchmark catalog")
	dnaLimit := flag.Int("dna-limit", 16, "sequences imported per corpus subset for -import-dna-corpus")
	captionClips := flag.String("build-caption-matching", "", "write the video caption-matching suite of the clips in this directory to -suite-out: each case asks which of four captions describes a clip")
	captionSource := flag.String("captions", "", "with -build-caption-matching: the caption file; empty takes the one the clip directory holds")
	captionCases := flag.Int("cases", defaultCaptionCases, "with -build-caption-matching: clips the suite asks about")
	captionSeed := flag.Uint64("seed", defaultCaptionSeed, "with -build-caption-matching: the seed that picks the clips and the distractors")
	framesOut := flag.String("frames-out", "", "with -build-caption-matching: cut each case's clip to its middle frame in this directory and ask for a description of the frame, for a model that reads one image and describes it")
	speechParquet := flag.String("build-speech-round-trip", "", "write the speech round-trip suite of this LibriSpeech parquet file's transcripts to -suite-out: -cases of them, read aloud and judged by the recognizer -judge names")
	speechJudge := flag.String("judge", "", "with -build-speech-round-trip, -render-tifa or -render-vidgen: the model that judges, a speech recognizer directory that listens or a vision-language model file that looks")
	tifaQuestions := flag.String("render-tifa", "", "draw a picture of each of -cases TIFA captions with the image model -model-path names, and write the suite that asks the judge about them to -suite-out, the pictures to -media-out: the argument is TIFA's question-answer file")
	vidgenCaptions := flag.String("render-vidgen", "", "draw a clip of each of -cases VidGen captions with the video model -model-path names, and write the suite that asks the judge which of four captions each shows to -suite-out, the clips to -media-out: the argument is VidGen's caption file")
	editClips := flag.String("render-edits", "", "change each of -cases captioned clips of this directory with the video editor -model-path names, by an instruction from a fixed list, and write the suite that asks the judge which change each edited clip shows and which caption still describes it to -suite-out, the edited clips to -media-out; -captions names the caption file")
	renderSet := flag.String("render-set", "", "with -render-tifa or -render-vidgen: comma-separated name=number settings that replace fields of the request the model's verification ran, such as width=416,height=240")
	sceneClips := flag.String("build-scene-retrieval", "", "write the scene-retrieval suite of the clips in this directory to -suite-out, the frames to -media-out: a frame of every clip with a sibling scene of its source video, and of -distractors clips with none")
	sceneDistractors := flag.Int("distractors", defaultSceneDistractors, "with -build-scene-retrieval: clips of a source video with no other scene to add to the gallery")
	reconstructionClips := flag.String("build-reconstruction", "", "write the reconstruction suite of the clips in this directory to -suite-out: -cases clips, picked by -seed and bound to the digest of their files, each cut to -frames frames for an autoencoder to take into its latent and give back")
	reconstructionFrames := flag.Int("frames", defaultReconstructionFrames, "with -build-reconstruction: frames of each clip the autoencoder is given; 1 is a picture")
	reconstructionWidth := flag.Int("width", defaultReconstructionWidth, "with -build-reconstruction: the width each frame is cut to")
	reconstructionHeight := flag.Int("height", defaultReconstructionHeight, "with -build-reconstruction: the height each frame is cut to")
	mediaOut := flag.String("media-out", "", "with -render-tifa, -render-vidgen or -build-scene-retrieval: the directory the pictures, clips or frames are written to, with the render manifest of a generator")
	speechVoice := flag.String("voice", defaultSpeechVoice, "with -build-speech-round-trip: the voice that reads")
	functionCalls := flag.String("build-function-calls", "", "write the function-call suite of the Berkeley Function-Calling Leaderboard's simple category held in this directory to -suite-out")
	seriesCSV := flag.String("build-series-forecast", "", "write the series-forecast suite of this ETT file's oil temperature to -suite-out: consecutive windows that show -context readings and ask for the -horizon after them")
	seriesContext := flag.Int("context", defaultSeriesContext, "with -build-series-forecast: readings each window shows")
	seriesHorizon := flag.Int("horizon", defaultSeriesHorizon, "with -build-series-forecast: readings each window asks for")
	passagesParquet := flag.String("build-passages", "", "write the held-out likelihood suite of this WikiText-2 test parquet file to -suite-out: passages evenly spaced through the split, each starting on a line and ending on a word")
	tabularParquet := flag.String("build-tabular-requests", "", "write the held-out classification requests of this credit-g parquet table to the directory -requests-out names: disjoint requests of context rows with their classes and query rows to answer")
	requestsOut := flag.String("requests-out", "", "with -build-tabular-requests: the directory the request files are written to")
	suiteOut := flag.String("suite-out", "", "with a -build- flag: the suite file to write")
	cpuProfile := flag.String("cpuprofile", "", "write a Go CPU profile of this process to the file (the host side of a pass; a worker's file is its own)")
	flag.Parse()
	analysisPath := cmp.Or(strings.TrimSpace(*alignmentManifest), strings.TrimSpace(*diarizationManifest))
	if path := analysisPath; path != "" {
		if flag.NArg() != 0 || *worker || *allModels || *manifestPath != "" || *transcriptionManifest != "" || *transcriptionResourceManifest != "" || *cpuProfile != "" || *importCache != "" || *listSuites || *importDNA != "" || *declareDomains != "" || *declareReferences != "" {
			return errors.New("usage: evaluate -alignment-manifest or -diarization-manifest <manifest.json> [-repo <store>] [-budget <duration>]")
		}
		if *alignmentManifest != "" && *diarizationManifest != "" {
			return errors.New("evaluate: select one speech analysis manifest")
		}
		if *budget <= 0 {
			return errors.New("evaluate: speech analysis requires a positive budget")
		}
		ctx, cancel := context.WithTimeoutCause(context.Background(), *budget, errors.New("speech analysis evaluation budget exhausted"))
		defer cancel()
		if *diarizationManifest != "" {
			return evaluateDiarizationManifest(ctx, *repository, path)
		}
		return evaluateAlignmentManifest(ctx, *repository, path)
	}
	if strings.TrimSpace(*transcriptionManifest) != "" && strings.TrimSpace(*transcriptionResourceManifest) != "" {
		return errors.New("evaluate: select one transcription manifest")
	}
	if *cpuProfile != "" && strings.TrimSpace(*transcriptionResourceManifest) != "" {
		return errors.New("evaluate: CPU profiling is not allowed during timed transcription evaluation")
	}
	if *cpuProfile != "" {
		profile, err := os.Create(*cpuProfile)
		if err != nil {
			return err
		}
		defer profile.Close()
		if err := pprof.StartCPUProfile(profile); err != nil {
			return err
		}
		defer pprof.StopCPUProfile()
	}
	if path := strings.TrimSpace(*transcriptionManifest); path != "" {
		if flag.NArg() != 0 || *worker || strings.TrimSpace(*manifestPath) != "" || *allModels {
			return errors.New("usage: evaluate -transcription-manifest <manifest.json> [-repo <store>]")
		}
		return evaluateTranscriptionManifest(context.Background(), *repository, path)
	}
	if path := strings.TrimSpace(*transcriptionResourceManifest); path != "" {
		if flag.NArg() != 0 || *worker || strings.TrimSpace(*manifestPath) != "" || *allModels {
			return errors.New("usage: evaluate -transcription-resource-manifest <manifest.json> [-repo <store>]")
		}
		if *budget <= 0 {
			return errors.New("evaluate: transcription execution requires a positive budget")
		}
		ctx, cancel := context.WithTimeoutCause(context.Background(), *budget, errors.New("transcription evaluation budget exhausted"))
		defer cancel()
		return evaluateTranscriptionResourceManifest(ctx, *repository, path)
	}
	if clips := strings.TrimSpace(*sceneClips); clips != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" || strings.TrimSpace(*mediaOut) == "" {
			return errors.New("usage: evaluate -build-scene-retrieval <clip directory> -media-out <frames directory> -suite-out <suite.json> [-distractors N] [-seed N]")
		}
		return writeSceneRetrieval(context.Background(), clips, strings.TrimSpace(*mediaOut), strings.TrimSpace(*suiteOut), *sceneDistractors, *captionSeed)
	}
	if clips := strings.TrimSpace(*reconstructionClips); clips != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" {
			return errors.New("usage: evaluate -build-reconstruction <clip directory> -suite-out <suite.json> [-frames N] [-width N] [-height N] [-cases N] [-seed N]")
		}
		return writeClipReconstruction(clips, strings.TrimSpace(*suiteOut), *captionCases, *reconstructionFrames, *reconstructionWidth, *reconstructionHeight, *captionSeed)
	}
	if clips := strings.TrimSpace(*editClips); clips != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" || strings.TrimSpace(*mediaOut) == "" || strings.TrimSpace(*speechJudge) == "" || strings.TrimSpace(*modelPath) == "" {
			return errors.New("usage: evaluate -render-edits <clip directory> -model-path <editor directory> -judge <vision-language model> -media-out <directory> -suite-out <suite.json> [-captions <file>] [-repo <store>] [-cases N] [-seed N]")
		}
		return writeVideoEdits(context.Background(), clips, strings.TrimSpace(*captionSource), strings.TrimSpace(*modelPath), *repository, strings.TrimSpace(*speechJudge), strings.TrimSpace(*mediaOut), strings.TrimSpace(*suiteOut), *captionCases, *captionSeed)
	}
	if source := cmp.Or(strings.TrimSpace(*tifaQuestions), strings.TrimSpace(*vidgenCaptions)); source != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" || strings.TrimSpace(*mediaOut) == "" || strings.TrimSpace(*speechJudge) == "" || strings.TrimSpace(*modelPath) == "" ||
			strings.TrimSpace(*tifaQuestions) != "" && strings.TrimSpace(*vidgenCaptions) != "" {
			return errors.New("usage: evaluate -render-tifa <tifa_v1.0_question_answers.json> | -render-vidgen <VidGen_1M_video_caption.json> -model-path <generator directory> -judge <vision-language model> -media-out <directory> -suite-out <suite.json> [-repo <store>] [-render-set name=number,...] [-cases N] [-seed N]")
		}
		draw := writeTIFAImages
		if strings.TrimSpace(*vidgenCaptions) != "" {
			draw = writeVidGenClips
		}
		return draw(context.Background(), source, strings.TrimSpace(*modelPath), *repository, strings.TrimSpace(*speechJudge), strings.TrimSpace(*mediaOut), strings.TrimSpace(*suiteOut), strings.TrimSpace(*renderSet), *captionCases, *captionSeed)
	}
	if parquet := strings.TrimSpace(*speechParquet); parquet != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" || strings.TrimSpace(*speechJudge) == "" {
			return errors.New("usage: evaluate -build-speech-round-trip <test.parquet> -judge <recognizer directory> -suite-out <suite.json> [-voice name] [-cases N] [-seed N]")
		}
		return writeSpeechRoundTrip(context.Background(), parquet, strings.TrimSpace(*speechJudge), strings.TrimSpace(*speechVoice), strings.TrimSpace(*suiteOut), *captionCases, *captionSeed)
	}
	if csv := strings.TrimSpace(*seriesCSV); csv != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" {
			return errors.New("usage: evaluate -build-series-forecast <ETTh1.csv> -suite-out <suite.json> [-context N] [-horizon N]")
		}
		return writeSeriesForecast(csv, strings.TrimSpace(*suiteOut), *seriesContext, *seriesHorizon)
	}
	if directory := strings.TrimSpace(*functionCalls); directory != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" {
			return errors.New("usage: evaluate -build-function-calls <leaderboard directory> -suite-out <suite.json>")
		}
		return writeFunctionCalls(directory, strings.TrimSpace(*suiteOut))
	}
	if parquet := strings.TrimSpace(*tabularParquet); parquet != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*requestsOut) == "" {
			return errors.New("usage: evaluate -build-tabular-requests <credit-g dataset.parquet> -requests-out <directory>")
		}
		return writeTabularRequests(context.Background(), parquet, strings.TrimSpace(*requestsOut))
	}
	if parquet := strings.TrimSpace(*passagesParquet); parquet != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" {
			return errors.New("usage: evaluate -build-passages <wikitext test parquet> -suite-out <suite.json>")
		}
		return writePassages(context.Background(), parquet, strings.TrimSpace(*suiteOut))
	}
	if clips := strings.TrimSpace(*captionClips); clips != "" {
		if flag.NArg() != 0 || strings.TrimSpace(*suiteOut) == "" {
			return errors.New("usage: evaluate -build-caption-matching <clip directory> -suite-out <suite.json> [-captions <file>] [-cases N] [-seed N]")
		}
		return writeCaptionMatching(context.Background(), clips, strings.TrimSpace(*captionSource), strings.TrimSpace(*suiteOut), strings.TrimSpace(*framesOut), *captionCases, *captionSeed)
	}
	if root := strings.TrimSpace(*importDNA); root != "" {
		if flag.NArg() != 0 {
			return errors.New("usage: evaluate -import-dna-corpus <root> [-dna-limit N] [-repo <store>]")
		}
		store, err := overgodb.Open(*repository)
		if err != nil {
			return err
		}
		defer store.Close()
		catalog, count, err := evaluation.CatalogDNACorpus(context.Background(), store, root, *dnaLimit)
		if err != nil {
			return err
		}
		fmt.Printf("benchmark catalog %s merged %d DNA corpus slice(s)\n", catalog, count)
		return nil
	}
	if spec := strings.TrimSpace(*declareReferences); spec != "" {
		if flag.NArg() != 0 {
			return errors.New("usage: evaluate -declare-references <spec.json> [-repo <store>]")
		}
		return declareReferenceScores(context.Background(), *repository, spec, *catalogLimit)
	}
	if csv := strings.TrimSpace(*declareDomains); csv != "" {
		if flag.NArg() != 1 {
			return errors.New("usage: evaluate -declare-domain <domains-csv> [-repo <store>] <model-path>")
		}
		modelPath, _ := checked.First(flag.Args())
		return declareEvalDomain(context.Background(), *repository, modelPath, csv, *catalogLimit)
	}
	if *allModels {
		if strings.TrimSpace(*manifestPath) != "" {
			return errors.New("usage: evaluate -all [-repo <store>] [-device N] [-family name]")
		}
		if *worker {
			previous, err := artifact.ParseID(*replay)
			if *replay != "" && err != nil {
				return err
			}
			return runAllWorker(context.Background(), *repository, *device, *family, *modelPath, *chatProtocol, *budget, previous)
		}
		return runAllParent(context.Background(), *repository, *device, *family, *catalogLimit, *chatProtocol, *budget)
	}
	if *listSuites {
		if flag.NArg() != 0 || *worker || strings.TrimSpace(*manifestPath) != "" {
			return errors.New("usage: evaluate -list-derived-suites [-repo <store>]")
		}
		store, err := overgodb.OpenReadOnly(*repository)
		if err != nil {
			return err
		}
		defer store.Close()
		// Placeholder authorities admit compilation for listing; running a
		// suite still binds the real model, recipe, and environment.
		suites, skipped, err := evaluation.DeriveStoreSuiteFamily(context.Background(), store, evaluation.ListingAuthorities(), strings.TrimSpace(*family))
		if err != nil {
			return err
		}
		for _, suite := range suites {
			descriptor := suite.Descriptor()
			fmt.Printf("%s cases=%d source=%s\n", descriptor.Kind, descriptor.Cases, descriptor.Source)
		}
		for family, dropped := range skipped {
			fmt.Printf("skipped %d %s case(s) outside the exact vocabulary\n", dropped, family)
		}
		return nil
	}
	if cache := strings.TrimSpace(*importCache); cache != "" {
		if flag.NArg() != 0 || *worker || strings.TrimSpace(*manifestPath) != "" {
			return errors.New("usage: evaluate -import-hf-cache <root> [-repo <store>]")
		}
		store, err := overgodb.Open(*repository)
		if err != nil {
			return err
		}
		defer store.Close()
		catalog, count, err := evaluation.CatalogHFCacheBenchmarks(context.Background(), store, cache)
		if err != nil {
			return err
		}
		fmt.Printf("benchmark catalog %s published from %d imported dataset(s)\n", catalog, count)
		return nil
	}
	if strings.TrimSpace(*manifestPath) == "" || flag.NArg() != 0 {
		return errors.New("usage: evaluate -manifest manifest.json")
	}
	compiled, err := readManifest(*manifestPath)
	if err != nil {
		return err
	}
	executableCommit, err := runrecord.ExecutableCodeCommit(".")
	if err != nil {
		return err
	}
	if err := requireProducer(compiled.CodeCommit, executableCommit); err != nil {
		return err
	}
	if *worker {
		if *modelIndex < 0 || *modelIndex >= len(compiled.Models) {
			return errors.New("evaluate: worker model index is invalid")
		}
		return executeModel(context.Background(), compiled, compiled.Models[*modelIndex], openEvaluationSession)
	}
	if err := catalogBenchmarks(context.Background(), compiled); err != nil {
		return err
	}
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	absoluteManifest, err := filepath.Abs(*manifestPath)
	if err != nil {
		return err
	}
	return runParent(context.Background(), compiled, func(ctx context.Context, index int) error {
		receipt, err := processcontrol.Run(ctx, processcontrol.Command{
			Path:   executable,
			Args:   []string{"-worker", "-manifest", absoluteManifest, "-model-index", strconv.Itoa(index)},
			Stdout: os.Stdout, Stderr: os.Stderr,
		})
		if err != nil {
			return err
		}
		if receipt.ExitCode != 0 {
			return fmt.Errorf("evaluate: isolated worker exited with status %d", receipt.ExitCode)
		}
		return nil
	})
}

// requireProducer refuses a manifest that names a commit other than the one
// this executable is built from: the commit becomes the producer of every plan
// and run the evaluation records, so a hand-written one would attribute the
// evidence to source that did not produce it.
func requireProducer(claimed, executable string) error {
	if claimed != executable {
		return fmt.Errorf("evaluate: the manifest names commit %.12s as the producer, but this executable is built from %.12s", claimed, executable)
	}
	return nil
}

func readManifest(path string) (manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return manifest{}, err
	}
	var value manifest
	if err := strictjson.DecodeBytes(data, &value); err != nil {
		return manifest{}, fmt.Errorf("evaluate: decode manifest: %w", err)
	}
	base, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return manifest{}, err
	}
	if !filepath.IsAbs(value.Repository) {
		value.Repository = filepath.Join(base, value.Repository)
	}
	if !filepath.IsAbs(value.Catalog) {
		value.Catalog = filepath.Join(base, value.Catalog)
	}
	for modelIndex := range value.Models {
		model := &value.Models[modelIndex]
		if !filepath.IsAbs(model.Path) {
			model.Path = filepath.Join(base, model.Path)
		}
		for suiteIndex := range model.Suites {
			if !filepath.IsAbs(model.Suites[suiteIndex]) {
				model.Suites[suiteIndex] = filepath.Join(base, model.Suites[suiteIndex])
			}
		}
	}
	return compileManifest(value)
}

func compileManifest(value manifest) (manifest, error) {
	value.Repository = strings.TrimSpace(value.Repository)
	value.Catalog = filepath.Clean(strings.TrimSpace(value.Catalog))
	value.CodeCommit = strings.TrimSpace(value.CodeCommit)
	if value.Repository == "" || value.Catalog == "." || value.CodeCommit == "" || value.Device < 0 || len(value.Models) == 0 {
		return manifest{}, errors.New("evaluate: incomplete manifest")
	}
	models := make([]modelRequest, 0, len(value.Models))
	indexByPath := make(map[string]int, len(value.Models))
	for _, request := range value.Models {
		request.Path = filepath.Clean(strings.TrimSpace(request.Path))
		if request.Path == "." || len(request.Suites) == 0 {
			return manifest{}, errors.New("evaluate: model path or suites are absent")
		}
		request.Suites = slices.Clone(request.Suites)
		for index := range request.Suites {
			request.Suites[index] = filepath.Clean(strings.TrimSpace(request.Suites[index]))
			if request.Suites[index] == "." {
				return manifest{}, errors.New("evaluate: suite path is absent")
			}
		}
		if index, exists := indexByPath[request.Path]; exists {
			models[index].Suites = append(models[index].Suites, request.Suites...)
			continue
		}
		indexByPath[request.Path] = len(models)
		models = append(models, request)
	}
	value.Models = models
	return value, nil
}

func catalogBenchmarks(ctx context.Context, value manifest) error {
	store, err := overgodb.Open(value.Repository)
	if err != nil {
		return err
	}
	if _, err := evaluation.CatalogLocalBenchmarks(ctx, store, value.Catalog); err != nil {
		return errors.Join(err, store.Close())
	}
	return store.Close()
}

// evaluationBudget is the owner ceiling on one full evaluation pass
// (2026-09-01): a pass that cannot finish inside it must fail loudly with
// the models it never reached named, so the operator scales suites or
// hardware -- exceeding the ceiling silently is not an option, and neither
// is reporting a truncated pass as complete.
const evaluationBudget = 6 * time.Hour

func runParent(ctx context.Context, value manifest, launch workerLauncher) error {
	if ctx == nil || launch == nil || len(value.Models) == 0 {
		return errors.New("evaluate: incomplete parent execution")
	}
	deadline := time.Now().Add(evaluationBudget)
	var failures []error
	for index, model := range value.Models {
		if time.Now().After(deadline) {
			failures = append(failures, fmt.Errorf(
				"model %q: unevaluated; the %s evaluation budget elapsed", model.Path, evaluationBudget))
			continue
		}
		if err := launch(ctx, index); err != nil {
			failures = append(failures, fmt.Errorf("model %q: %w", model.Path, err))
		}
	}
	return errors.Join(failures...)
}
