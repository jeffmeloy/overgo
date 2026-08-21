package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	llamaserver "overgo/internal/server"
)

const (
	serverReadHeaderTimeout = 10 * time.Second
	serverReadTimeout       = 30 * time.Second
	serverIdleTimeout       = 2 * time.Minute
	serverShutdownTimeout   = 30 * time.Second
	serverMaxHeaderBytes    = 1 << 20

	defaultAnalysisTensorSamples = 4096
	defaultAnalysisTensorBytes   = 64 << 20
	defaultAnalysisPositions     = 64
	defaultAnalysisMDSIterations = 1000
	defaultAnalysisMDSTolerance  = 1e-6
)

func main() {
	clioptions.Main(run)
}

func run() error {
	address := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	modelID := flag.String("model-id", llamaserver.DefaultModelID, "API model identifier")
	modelFlags := clioptions.AddModelFlags(flag.CommandLine, "load GGUF LoRA adapter at global scale 1; repeatable")
	loraDisabled := flag.Bool("lora-init-without-apply", false, "load adapters with global scale 0")
	maxTokens := flag.Int("max-tokens", llamaserver.DefaultMaxTokens, "maximum max_tokens accepted per request")
	contextShift := flag.Bool(
		"context-shift",
		false,
		"discard oldest attention KV entries when generation reaches model context",
	)
	maxConcurrent := flag.Int("max-concurrent", llamaserver.DefaultMaxConcurrent, "maximum admitted generation requests")
	spmInfill := flag.Bool(
		"spm-infill",
		false,
		"use suffix/prefix/middle instead of prefix/suffix/middle for FIM",
	)
	promptCacheEntries := flag.Int(
		"prompt-cache-entries",
		1,
		"maximum independently reusable prompt states retained by the Runner",
	)
	maxEmbeddingInputs := flag.Int(
		"max-embedding-inputs", llamaserver.DefaultMaxEmbeddingInputs, "maximum strings accepted by one embedding request",
	)
	requestTimeout := flag.Duration(
		"request-timeout",
		0,
		"maximum end-to-end request duration; zero disables",
	)
	responseStoreEntries := flag.Int(
		"response-store-entries",
		llamaserver.DefaultStoredResponses,
		"maximum Responses continuation histories retained in memory",
	)
	responseStoreBytes := flag.Int(
		"response-store-bytes",
		llamaserver.DefaultResponseStoreBytes,
		"maximum aggregate bytes retained for Responses continuation",
	)
	apiKeyFile := flag.String(
		"api-key-file",
		"",
		"read the /v1 bearer token from this file (or OVERGO_API_KEY)",
	)
	projectorPath := flag.String("mmproj", "", "multimodal projector GGUF")
	projectorCUDA := flag.Bool("mmproj-cuda", false, "offload supported multimodal projector operations to CUDA")
	mediaPolicyPath := flag.String("media-policy", "media_policy.json", "remote-media JSON policy; empty disables URLs")
	resourcePolicyPath := flag.String("resource-policy", "resource_policy.json", "Responses file-ID JSON policy; empty disables file IDs")
	ffmpegPath := flag.String("ffmpeg", os.Getenv("OVERGO_FFMPEG"), "FFmpeg executable for encoded video")
	videoFPS := flag.Float64("video-fps", llamaserver.DefaultVideoFPS, "video frame sampling rate")
	videoMaxFrames := flag.Int("video-max-frames", llamaserver.DefaultVideoFrameLimit, "maximum decoded video frames")
	trainingEnabled := flag.Bool("training", false, "enable active recipe-bound training workspace")
	modelBuilderEnabled := flag.Bool("model-builder", false, "enable corpus-derived model builder workspace")
	var evaluationSuites []string
	flag.Func("evaluation-suite", "compiled evaluation suite JSON; repeatable", func(value string) error {
		value = strings.TrimSpace(value)
		if value == "" {
			return errors.New("evaluation suite path is empty")
		}
		evaluationSuites = append(evaluationSuites, value)
		return nil
	})
	evaluationCommit := flag.String("evaluation-commit", "", "source commit bound to evaluation evidence")
	analysisTensorSamples := flag.Uint64("analysis-tensor-samples", defaultAnalysisTensorSamples, "samples retained per analyzed tensor")
	analysisTensorBytes := flag.Uint64("analysis-tensor-bytes", defaultAnalysisTensorBytes, "aggregate tensor bytes read per analysis")
	analysisPositions := flag.Int("analysis-state-positions", defaultAnalysisPositions, "maximum positions retained by state analysis")
	analysisMDSIterations := flag.Int("analysis-mds-iterations", defaultAnalysisMDSIterations, "state-layout convergence iteration bound")
	analysisMDSTolerance := flag.Float64("analysis-mds-tolerance", defaultAnalysisMDSTolerance, "state-layout relative convergence tolerance")
	flag.Parse()
	if flag.NArg() != 1 {
		return errors.New("usage: server [options] <model.gguf>")
	}
	loraScale := float32(1)
	if *loraDisabled {
		loraScale = 0
	}
	openOptions := modelFlags.OpenOptions(loraScale)
	openOptions.PromptCacheEntries = *promptCacheEntries
	shutdownContext, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	runner, err := modelFlags.OpenRunnerWithOptions(shutdownContext, flag.Arg(0), openOptions)
	if err != nil {
		return err
	}
	defer runner.Close()
	// Browse roots remain optional unless training is enabled.
	roots, rootsErr := dataroot.ResolveCurrent()
	datasetsRoot, repoPath := "", strings.TrimSpace(*modelFlags.Repository)
	if rootsErr == nil {
		datasetsRoot = roots.Datasets
		if repoPath == "" {
			repoPath = roots.Store
		}
	}
	var generator llamaserver.Generator = runner
	var workspaceStore *repodb.Store
	if repoPath != "" {
		workspaceStore, err = repodb.Open(repoPath)
		if err != nil {
			return fmt.Errorf("open workspace repository: %w", err)
		}
		defer workspaceStore.Close()
	}
	var workflowWorkspaces llamaserver.WorkflowWorkspaceSet
	if *trainingEnabled {
		if rootsErr != nil {
			return rootsErr
		}
		workspace, err := llamaserver.NewTrainingWorkspace(shutdownContext, workspaceStore, roots, runner.ModelID())
		if err != nil {
			return fmt.Errorf("open training workspace: %w", err)
		}
		workflowWorkspaces = append(workflowWorkspaces, workspace)
	}
	if *modelBuilderEnabled {
		workspace, err := llamaserver.NewModelBuilderWorkspace(workspaceStore)
		if err != nil {
			return fmt.Errorf("open model builder workspace: %w", err)
		}
		workflowWorkspaces = append(workflowWorkspaces, workspace)
	}
	if len(workflowWorkspaces) > 0 {
		generator = &serverRuntime{Runner: runner, WorkflowWorkspaceAPI: workflowWorkspaces}
	}
	var evaluationWorkspace llamaserver.EvaluationWorkspaceAPI
	if len(evaluationSuites) > 0 {
		description, err := runner.RecipeRuntimeDescription(recipe.TaskInference)
		if err != nil {
			return err
		}
		environment, err := runrecord.CurrentEnvironment(fmt.Sprintf("cuda:%d", *modelFlags.DeviceOrdinal), "cuda")
		if err != nil {
			return err
		}
		evaluationWorkspace, err = llamaserver.NewEvaluationWorkspace(
			workspaceStore, runner, description.Identity, environment,
			strings.TrimSpace(*evaluationCommit), evaluationSuites,
		)
		if err != nil {
			return fmt.Errorf("open evaluation workspace: %w", err)
		}
	}
	var vision, audio projector.Session
	if *projectorPath != "" {
		repository, repositoryErr := modelFlags.RepositoryPath()
		if repositoryErr != nil {
			return repositoryErr
		}
		store, openErr := repodb.OpenReadOnly(repository)
		if openErr != nil {
			return fmt.Errorf("open model recipe repository: %w", openErr)
		}
		vision, err = projector.OpenActiveSession(shutdownContext, store, runner.ModelID(), *projectorPath, projector.OpenOptions{
			CUDA: *projectorCUDA, DeviceOrdinal: *modelFlags.DeviceOrdinal,
		})
		err = errors.Join(err, store.Close())
		if err != nil {
			return fmt.Errorf("open multimodal projector: %w", err)
		}
		defer vision.Close()
		if vision.Capabilities().Audio {
			audio = vision
		}
	}
	apiKey := strings.TrimSpace(os.Getenv("OVERGO_API_KEY"))
	if *apiKeyFile != "" {
		data, readErr := os.ReadFile(*apiKeyFile)
		if readErr != nil {
			return fmt.Errorf("read API key: %w", readErr)
		}
		apiKey = strings.TrimSpace(string(data))
		if apiKey == "" {
			return errors.New("API key file is empty")
		}
	}
	var mediaPolicy *llamaserver.RemoteMediaPolicy
	if *mediaPolicyPath != "" {
		mediaPolicy, err = llamaserver.LoadRemoteMediaPolicy(*mediaPolicyPath)
		if err != nil {
			return err
		}
	}
	var resourcePolicy *llamaserver.ResponseFilePolicy
	var responseToolPolicy llamaserver.ResponseToolPolicy
	if *resourcePolicyPath != "" {
		resourcePolicy, err = llamaserver.LoadResponseFilePolicy(*resourcePolicyPath)
		if err != nil {
			return err
		}
		responseToolPolicy = resourcePolicy.ResponseTools
	}
	handler, err := llamaserver.New(llamaserver.Config{
		ModelID:            *modelID,
		MaxTokens:          *maxTokens,
		MaxConcurrent:      *maxConcurrent,
		MaxEmbeddingInputs: *maxEmbeddingInputs,
		DefaultTemperature: llamaserver.DefaultSamplingTemperature,
		DefaultTopP:        llamaserver.DefaultSamplingTopP,
		DefaultTopK:        llamaserver.DefaultSamplingTopK,
		APIKey:             apiKey,
		ContextShift:       *contextShift,
		RequestTimeout:     *requestTimeout,
		SPMInfill:          *spmInfill,
		ImageProjector:     vision,
		AudioProjector:     audio,
		RemoteMediaPolicy:  mediaPolicy,
		ResponseFiles:      resourcePolicy,
		ResponseToolPolicy: responseToolPolicy,
		MaxStoredResponses: *responseStoreEntries,
		ResponseStoreBytes: *responseStoreBytes,
		FFmpegPath:         *ffmpegPath,
		VideoFPS:           *videoFPS,
		VideoMaxFrames:     *videoMaxFrames,
		DatasetsRoot:       datasetsRoot,
		RepoDBPath:         repoPath,
		Repository:         workspaceStore,
		Evaluation:         evaluationWorkspace,
		Analysis: llamaserver.AnalysisPolicy{
			TensorSamples: *analysisTensorSamples, TensorReadBytes: *analysisTensorBytes,
			StatePositions: *analysisPositions, MDSIterations: *analysisMDSIterations,
			MDSTolerance: *analysisMDSTolerance,
		},
	}, generator)
	if err != nil {
		return err
	}
	defer handler.Close()
	httpServer := &http.Server{
		Addr:              *address,
		Handler:           handler,
		ReadHeaderTimeout: serverReadHeaderTimeout,
		ReadTimeout:       serverReadTimeout,
		WriteTimeout:      0,
		IdleTimeout:       serverIdleTimeout,
		MaxHeaderBytes:    serverMaxHeaderBytes,
	}
	log.Printf("serving model %q on http://%s", *modelID, *address)
	serverError := make(chan error, 1)
	go func() {
		serverError <- httpServer.ListenAndServe()
	}()
	select {
	case err = <-serverError:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-shutdownContext.Done():
	}
	log.Print("shutting down")
	deadline, cancel := context.WithTimeout(context.Background(), serverShutdownTimeout)
	defer cancel()
	if err := httpServer.Shutdown(deadline); err != nil {
		return fmt.Errorf("server shutdown: %w", err)
	}
	err = <-serverError
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
