package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"time"

	"overgo/internal/checked"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/driver"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/mediacapability"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	llamaserver "overgo/internal/server"
	"overgo/internal/tensor"
)

const (
	serverReadHeaderTimeout = 10 * time.Second
	serverReadTimeout       = 30 * time.Second
	serverIdleTimeout       = 2 * time.Minute
	serverMaxHeaderBytes    = 1 << 20

	// generationCatalogLimit bounds the activated models the generation workspace lists.
	generationCatalogLimit = 256
	// transcriptionPolicyDocument: the transcription policy declared beside
	// the store (in its data root); found, it serves without the flag.
	transcriptionPolicyDocument = "transcription_policy.json"

	defaultAnalysisTensorSamples = 4096
	defaultAnalysisTensorBytes   = 64 << 20
	defaultAnalysisPositions     = 64
	defaultAnalysisMDSIterations = 1000
	defaultAnalysisMDSTolerance  = 1e-6
)

func main() {
	clioptions.Main(func() error { return deviceStartupError(run()) })
}

// deviceStartupError types a device allocation failure so the supervising
// launcher classifies the exit without reading error text.
func deviceStartupError(err error) error {
	if driver.IsOutOfMemory(err) {
		return fmt.Errorf("%w: %v", processcontrol.ErrDeviceMemory, err)
	}
	return err
}

func run() error {
	analysisDefaults := llamaserver.AnalysisPolicy{
		TensorSamples: defaultAnalysisTensorSamples, TensorReadBytes: defaultAnalysisTensorBytes,
		StatePositions: defaultAnalysisPositions, MDSIterations: defaultAnalysisMDSIterations,
		MDSTolerance: defaultAnalysisMDSTolerance,
	}
	address := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	modelID := flag.String("model-id", "", "API model identifier; unset uses recipe policy")
	modelFlags := modelcli.AddModelFlags(flag.CommandLine, "load GGUF LoRA adapter at global scale 1; repeatable")
	loraDisabled := flag.Bool("lora-init-without-apply", false, "load adapters with global scale 0")
	maxTokens := clioptions.IntOverride(flag.CommandLine, "max-tokens", "maximum max_tokens accepted per request; unset uses recipe policy")
	contextShift := flag.Bool(
		"context-shift",
		false,
		"discard oldest attention KV entries when generation reaches model context",
	)
	maxConcurrent := clioptions.IntOverride(flag.CommandLine, "max-concurrent", "maximum admitted generation requests; unset uses recipe policy")
	spmInfill := flag.Bool(
		"spm-infill",
		false,
		"use suffix/prefix/middle instead of prefix/suffix/middle for FIM",
	)
	promptCacheEntries := clioptions.IntOverride(
		flag.CommandLine,
		"prompt-cache-entries",
		"maximum independently reusable prompt states retained by the Runner; unset uses recipe policy",
	)
	maxEmbeddingInputs := clioptions.IntOverride(
		flag.CommandLine, "max-embedding-inputs", "maximum strings accepted by one embedding request; unset uses recipe policy",
	)
	requestTimeout := clioptions.DurationOverride(
		flag.CommandLine,
		"request-timeout",
		"maximum end-to-end request duration; zero disables",
	)
	responseStoreEntries := clioptions.IntOverride(
		flag.CommandLine,
		"response-store-entries",
		"maximum Responses continuation histories retained in memory; unset uses recipe policy",
	)
	responseStoreBytes := clioptions.IntOverride(
		flag.CommandLine,
		"response-store-bytes",
		"maximum aggregate bytes retained for Responses continuation; unset uses recipe policy",
	)
	apiKeyFile := flag.String(
		"api-key-file",
		"",
		"read the /v1 bearer token from this file (or OVERGO_API_KEY)",
	)
	projectorPath := flag.String("mmproj", "", "multimodal projector GGUF")
	projectorCUDA := clioptions.BoolOverride(flag.CommandLine, "mmproj-cuda", "offload projector operations to CUDA; unset follows active inference placement")
	mediaPolicyPath := flag.String("media-policy", "media_policy.json", "remote-media JSON policy; empty disables URLs")
	resourcePolicyPath := flag.String("resource-policy", "resource_policy.json", "Responses file-ID JSON policy; empty disables file IDs")
	ffmpegPath := flag.String("ffmpeg", os.Getenv("OVERGO_FFMPEG"), "FFmpeg executable for encoded video")
	videoFPS := clioptions.Float64Override(flag.CommandLine, "video-fps", "video frame sampling rate; unset uses recipe policy")
	videoMaxFrames := clioptions.IntOverride(flag.CommandLine, "video-max-frames", "maximum decoded video frames; unset uses recipe policy")
	trainingEnabled := flag.Bool("training", false, "enable active recipe-bound training workspace")
	transcriptionPolicyPath := flag.String("transcription-policy", "", "strict CPU transcription resource policy; unset discovers transcription_policy.json beside the store")
	modelBuilderEnabled := flag.Bool("model-builder", false, "enable corpus-derived model builder workspace")
	webuiDir := flag.String("webui-dir", "", "serve the workbench client from this directory with caching disabled (development); empty serves the embedded client")
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
	analysisTensorSamples := flag.Uint64("analysis-tensor-samples", analysisDefaults.TensorSamples, "samples retained per analyzed tensor")
	analysisTensorBytes := flag.Uint64("analysis-tensor-bytes", analysisDefaults.TensorReadBytes, "aggregate tensor bytes read per analysis")
	analysisPositions := flag.Int("analysis-state-positions", analysisDefaults.StatePositions, "maximum positions retained by state analysis")
	analysisMDSIterations := flag.Int("analysis-mds-iterations", analysisDefaults.MDSIterations, "state-layout convergence iteration bound")
	analysisMDSTolerance := flag.Float64("analysis-mds-tolerance", analysisDefaults.MDSTolerance, "state-layout relative convergence tolerance")
	flag.Parse()
	explicit := clioptions.ExplicitOverrides(flag.CommandLine)
	if flag.NArg() != 1 {
		return errors.New("usage: server [options] <model-reference>")
	}
	var transcriptionPolicy *llamaserver.TranscriptionPolicy
	if *transcriptionPolicyPath != "" {
		policy, err := llamaserver.LoadTranscriptionPolicy(*transcriptionPolicyPath)
		if err != nil {
			return fmt.Errorf("load transcription policy: %w", err)
		}
		transcriptionPolicy = &policy
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
	if err := clioptions.RequireLoopbackWithoutCredential(*address, apiKey); err != nil {
		return fmt.Errorf("%w (set -api-key-file or OVERGO_API_KEY)", err)
	}
	var loraScale float32
	if !*loraDisabled {
		loraScale = tensor.UnitScale
	}
	openOptions := modelFlags.OpenOptions(loraScale)
	if _, set := explicit["prompt-cache-entries"]; set {
		openOptions.PromptCacheEntries = *promptCacheEntries
	}
	shutdownContext, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	modelReference, _ := checked.First(flag.Args())
	// A reference the store declares as a remote model serves through the
	// relay; every other reference is a local model path the runner opens.
	repositoryPath, err := modelFlags.RepositoryPath()
	if err != nil {
		return err
	}
	remote, err := resolveRemoteServing(shutdownContext, repositoryPath, modelReference)
	if err != nil {
		return err
	}
	if remote != nil {
		serving := remote.policy.Serving
		clioptions.ApplyDefault(explicit, "model-id", modelID, remote.provider.Model)
		clioptions.ApplyDefault(explicit, "max-tokens", maxTokens, serving.MaxTokens)
		clioptions.ApplyDefault(explicit, "max-concurrent", maxConcurrent, serving.MaxConcurrent)
		clioptions.ApplyDefault(explicit, "response-store-entries", responseStoreEntries, serving.StoredResponses)
		clioptions.ApplyDefault(explicit, "response-store-bytes", responseStoreBytes, serving.ResponseStoreBytes)
		hubRoot := ""
		if roots, rootsErr := dataroot.ResolveCurrent(); rootsErr == nil {
			hubRoot = roots.Models
		}
		return serveRemote(shutdownContext, remote, serveOptions{
			address: *address, apiKey: apiKey, modelID: *modelID,
			maxTokens: *maxTokens, maxConcurrent: *maxConcurrent,
			storedResponses: *responseStoreEntries, responseStoreBytes: *responseStoreBytes,
			requestTimeout: *requestTimeout, repository: repositoryPath, hubRoot: hubRoot, webuiDir: *webuiDir,
		})
	}
	native, err := resolveTranscriptionServing(shutdownContext, repositoryPath, modelReference)
	if err != nil {
		return err
	}
	if native != nil {
		return serveTranscription(shutdownContext, *native, transcriptionPolicy, serveOptions{
			address: *address, apiKey: apiKey, modelID: *modelID, maxConcurrent: *maxConcurrent,
			requestTimeout: *requestTimeout, repository: repositoryPath, webuiDir: *webuiDir,
		})
	}
	runner, err := modelFlags.OpenRunnerWithOptions(shutdownContext, modelReference, openOptions)
	if err != nil {
		return err
	}
	defer runner.Close()
	description, err := runner.RecipeRuntimeDescription(recipe.TaskInference)
	if err != nil {
		return err
	}
	clioptions.ApplyDefault(explicit, "mmproj-cuda", projectorCUDA, slices.ContainsFunc(description.Stages, func(stage recipe.Stage) bool {
		return stage.Node.Placement == recipe.PlacementDevice || stage.Node.Placement == recipe.PlacementHybrid
	}))
	policy := runner.RuntimePolicy()
	agentRetrieval, err := newAgentRetrievalProvider(runner, runner.ModelID(), policy.ID)
	if err != nil {
		return err
	}
	serving := policy.Serving
	clioptions.ApplyDefault(explicit, "model-id", modelID, serving.ModelID)
	clioptions.ApplyDefault(explicit, "max-tokens", maxTokens, serving.MaxTokens)
	clioptions.ApplyDefault(explicit, "max-concurrent", maxConcurrent, serving.MaxConcurrent)
	clioptions.ApplyDefault(explicit, "max-embedding-inputs", maxEmbeddingInputs, serving.MaxEmbeddingInputs)
	clioptions.ApplyDefault(explicit, "response-store-entries", responseStoreEntries, serving.StoredResponses)
	clioptions.ApplyDefault(explicit, "response-store-bytes", responseStoreBytes, serving.ResponseStoreBytes)
	clioptions.ApplyDefault(explicit, "video-fps", videoFPS, serving.Video.FPS)
	clioptions.ApplyDefault(explicit, "video-max-frames", videoMaxFrames, serving.Video.MaxFrames)
	// Browse roots remain optional unless training is enabled.
	roots, rootsErr := dataroot.ResolveCurrent()
	repoPath, hubRoot := strings.TrimSpace(*modelFlags.Repository), ""
	if rootsErr == nil {
		hubRoot = roots.Models
		if repoPath == "" {
			repoPath = roots.Store
		}
	}
	var generator llamaserver.Generator = runner
	var workspaceStore *overgodb.Store
	if repoPath != "" {
		workspaceStore, err = overgodb.Open(repoPath)
		if err != nil {
			return fmt.Errorf("open workspace repository: %w", err)
		}
		defer workspaceStore.Close()
	}
	var workflowWorkspaces llamaserver.WorkflowWorkspaceSet
	// A transcription policy declared beside the store (the data root's
	// transcription_policy.json) serves without the flag, so a server the
	// swap proxy launches offers the store's active transcription recipe;
	// its refusal (the recipe retired) is logged, not fatal.
	discovered := false
	if transcriptionPolicy == nil && repoPath != "" {
		candidate := filepath.Join(filepath.Dir(repoPath), transcriptionPolicyDocument)
		if _, statErr := os.Stat(candidate); statErr == nil {
			policy, err := llamaserver.LoadTranscriptionPolicy(candidate)
			if err != nil {
				return fmt.Errorf("load transcription policy: %w", err)
			}
			transcriptionPolicy, discovered = &policy, true
		}
	}
	if transcriptionPolicy != nil {
		if workspaceStore == nil {
			return errors.New("transcription workspace requires a repository")
		}
		commit, err := runrecord.ExecutableCodeCommit(".")
		if err != nil {
			return fmt.Errorf("bind transcription source: %w", err)
		}
		workspace, err := llamaserver.NewTranscriptionWorkspace(shutdownContext, workspaceStore, *transcriptionPolicy, commit)
		switch {
		case err != nil && discovered:
			log.Printf("transcription workspace unavailable under the declared policy: %v", err)
		case err != nil:
			return fmt.Errorf("open transcription workspace: %w", err)
		default:
			defer workspace.Close(context.WithoutCancel(shutdownContext))
			workflowWorkspaces = append(workflowWorkspaces, workspace)
		}
	}
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
		workspace, err := llamaserver.NewModelBuilderWorkspace(shutdownContext, workspaceStore)
		if err != nil {
			return fmt.Errorf("open model builder workspace: %w", err)
		}
		workflowWorkspaces = append(workflowWorkspaces, workspace)
	}
	// Generation rides the store alone: every active media recipe with
	// bytes on disk is a capability of any server opened over the store.
	if workspaceStore != nil {
		workflowWorkspaces = append(workflowWorkspaces, llamaserver.NewStoreGenerationWorkspace(workspaceStore,
			llamaserver.BindGenerationCatalog(mediacapability.Catalog, mediacapability.Controls, mediacapability.OutputContent), generationCatalogLimit))
	}
	if len(workflowWorkspaces) > 0 {
		generator = &serverRuntime{Runner: runner, WorkflowWorkspaceAPI: workflowWorkspaces}
	}
	var evaluationWorkspace llamaserver.EvaluationWorkspaceAPI
	// The workspace opens whenever a trustworthy commit identity exists:
	// suite files bind explicitly, and with none named the suites derive
	// from the store's benchmark catalog, so the workbench evaluates out
	// of the box. Without -evaluation-commit the clean worktree HEAD
	// stands in; a dirty tree leaves evaluation off rather than binding
	// evidence to a commit that differs from executed source.
	workspaceCommit := strings.TrimSpace(*evaluationCommit)
	if workspaceCommit == "" {
		if head, headErr := runrecord.VerifyingCommit("."); headErr == nil {
			workspaceCommit = head
		} else if len(evaluationSuites) > 0 {
			return fmt.Errorf("open evaluation workspace: %w", headErr)
		}
	}
	if workspaceCommit != "" {
		environment, err := runrecord.CurrentEnvironment(fmt.Sprintf("cuda:%d", *modelFlags.DeviceOrdinal), "cuda")
		if err != nil {
			return err
		}
		evaluationWorkspace, err = llamaserver.NewEvaluationWorkspace(
			workspaceStore, runner, description.Identity, environment,
			workspaceCommit, *responseStoreEntries, evaluationSuites,
		)
		if err != nil && len(evaluationSuites) > 0 {
			return fmt.Errorf("open evaluation workspace: %w", err)
		}
		if err != nil {
			log.Printf("evaluation workspace unavailable: %v", err)
			evaluationWorkspace = nil
		}
	}
	// The projector comes from the command line or from the store: with no
	// -mmproj, the model's active projection recipe names the projector
	// bytes, so every launcher serves the modalities the store declares.
	var vision, audio projector.Session
	{
		repository, repositoryErr := modelFlags.RepositoryPath()
		if repositoryErr != nil {
			return repositoryErr
		}
		store, openErr := overgodb.OpenReadOnly(repository)
		if openErr != nil {
			return fmt.Errorf("open model recipe repository: %w", openErr)
		}
		resolved := *projectorPath
		if resolved == "" {
			declared, ok, resolveErr := discovery.ActiveProjector(shutdownContext, store, runner.ModelID(), discovery.LoadMemo(shutdownContext, store))
			if resolveErr != nil {
				return errors.Join(fmt.Errorf("resolve declared projector: %w", resolveErr), store.Close())
			}
			if ok {
				resolved = declared
				log.Printf("serving the declared projector %s", resolved)
			}
		}
		if resolved != "" {
			vision, err = projector.OpenActiveSession(shutdownContext, store, runner.ModelID(), resolved, projector.OpenOptions{
				CUDA: *projectorCUDA, DeviceOrdinal: *modelFlags.DeviceOrdinal,
			})
		}
		err = errors.Join(err, store.Close())
		if err != nil {
			return fmt.Errorf("open multimodal projector: %w", err)
		}
		if vision != nil {
			defer vision.Close()
			if vision.Capabilities().Audio {
				audio = vision
			}
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
	if *resourcePolicyPath != "" {
		resourcePolicy, err = llamaserver.LoadResponseFilePolicy(*resourcePolicyPath)
		if err != nil {
			return err
		}
	}
	handler, err := llamaserver.New(llamaserver.Config{
		RuntimePolicy:      policy,
		ModelID:            *modelID,
		MaxTokens:          *maxTokens,
		MaxConcurrent:      *maxConcurrent,
		MaxEmbeddingInputs: *maxEmbeddingInputs,
		APIKey:             apiKey,
		ContextShift:       *contextShift,
		RequestTimeout:     *requestTimeout,
		SPMInfill:          *spmInfill,
		ImageProjector:     vision,
		AudioProjector:     audio,
		RemoteMediaPolicy:  mediaPolicy,
		ResponseFiles:      resourcePolicy,
		MaxStoredResponses: *responseStoreEntries,
		ResponseStoreBytes: *responseStoreBytes,
		FFmpegPath:         *ffmpegPath,
		VideoFPS:           *videoFPS,
		VideoMaxFrames:     *videoMaxFrames,
		OvergoDBPath:       repoPath,
		Repository:         workspaceStore,
		HubToken:           os.Getenv("OVERGO_HF_TOKEN"),
		HubDownloadRoot:    hubRoot,
		WebUIDir:           *webuiDir,
		Evaluation:         evaluationWorkspace,
		AgentEmbedder:      agentRetrieval,
		AgentReranker:      agentRetrieval,
		LibraryIntake:      serverLibraryIntake(),
		ProviderKeys:       providerIntake.Keys,
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
	log.Printf("serving model %q on http://%s", *modelID, *address)
	return serve(shutdownContext, *address, handler, *requestTimeout)
}

// serve binds the address, announces the bound listener, and runs the
// handler until the context ends; the drain then waits for the requests in
// flight, each already bounded by the declared request timeout when one is
// set, and a further interrupt abandons it.
func serve(ctx context.Context, address string, handler http.Handler, requestTimeout time.Duration) error {
	httpServer := &http.Server{
		Handler:           handler,
		ReadHeaderTimeout: serverReadHeaderTimeout,
		ReadTimeout:       serverReadTimeout,
		IdleTimeout:       serverIdleTimeout,
		MaxHeaderBytes:    serverMaxHeaderBytes,
	}
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}
	// The announcement follows the bind: a launcher reading it may connect at once.
	log.Printf("%s%s", processcontrol.ListeningAnnouncement, listener.Addr())
	serverError := make(chan error, 1)
	go func() {
		serverError <- httpServer.Serve(listener)
	}()
	select {
	case err := <-serverError:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case <-ctx.Done():
	}
	log.Print("shutting down; a second interrupt abandons the drain")
	drain, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if requestTimeout > 0 {
		var cancel context.CancelFunc
		drain, cancel = context.WithTimeoutCause(drain, requestTimeout, errors.New("server: the drain outlived the request timeout"))
		defer cancel()
	}
	if err := httpServer.Shutdown(drain); err != nil {
		return fmt.Errorf("server shutdown: %w", err)
	}
	err = <-serverError
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}
