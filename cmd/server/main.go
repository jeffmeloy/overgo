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

	"llamacpp2go/internal/inference"
	"llamacpp2go/internal/projector"
	llamaserver "llamacpp2go/internal/server"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	address := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	modelID := flag.String("model-id", "llamacpp2go", "API model identifier")
	deviceOrdinal := flag.Int("device", 0, "CUDA device ordinal")
	preload := flag.Bool("preload", false, "dequantize all model weights once into CUDA memory")
	nativeQ8 := flag.Bool("native-q8", false, "preload Q8_0 weights without dequantizing them")
	nativeQuant := flag.Bool("native-quant", false, "preload supported quantized weights without dequantizing them")
	var loraPaths []string
	flag.Func("lora", "load GGUF LoRA adapter at global scale 1; repeatable", func(value string) error {
		if strings.TrimSpace(value) == "" {
			return errors.New("LoRA path is empty")
		}
		loraPaths = append(loraPaths, value)
		return nil
	})
	loraDisabled := flag.Bool("lora-init-without-apply", false, "load adapters with global scale 0")
	maxTokens := flag.Int("max-tokens", 4096, "maximum max_tokens accepted per request")
	contextShift := flag.Bool(
		"context-shift",
		false,
		"discard oldest attention KV entries when generation reaches model context",
	)
	maxConcurrent := flag.Int("max-concurrent", 1, "maximum admitted generation requests")
	infillBatchSize := flag.Int(
		"batch-size",
		inference.DefaultInfillBatchSize,
		"logical prompt batch size used by native FIM formatting",
	)
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
	maxEmbeddingInputs := flag.Int("max-embedding-inputs", 16, "maximum strings accepted by one embedding request")
	requestTimeout := flag.Duration(
		"request-timeout",
		0,
		"maximum end-to-end request duration; zero disables",
	)
	responseStoreEntries := flag.Int(
		"response-store-entries",
		128,
		"maximum Responses continuation histories retained in memory",
	)
	responseStoreBytes := flag.Int(
		"response-store-bytes",
		64<<20,
		"maximum aggregate bytes retained for Responses continuation",
	)
	apiKeyFile := flag.String(
		"api-key-file",
		"",
		"read the /v1 bearer token from this file (or LLAMACPP2GO_API_KEY)",
	)
	projectorPath := flag.String("mmproj", "", "multimodal projector GGUF")
	projectorCUDA := flag.Bool("mmproj-cuda", false, "offload supported multimodal projector operations to CUDA")
	mediaPolicyPath := flag.String("media-policy", "media_policy.yaml", "remote-media YAML policy; empty disables URLs")
	resourcePolicyPath := flag.String("resource-policy", "resource_policy.yaml", "Responses file-ID YAML policy; empty disables file IDs")
	ffmpegPath := flag.String("ffmpeg", os.Getenv("LLAMACPP2GO_FFMPEG"), "FFmpeg executable for encoded video")
	videoFPS := flag.Float64("video-fps", 2, "video frame sampling rate")
	videoMaxFrames := flag.Int("video-max-frames", 32, "maximum decoded video frames")
	flag.Parse()
	if flag.NArg() != 1 {
		return errors.New("usage: server [options] <model.gguf>")
	}
	loraAdapters := make([]inference.LoRAConfig, len(loraPaths))
	for index, path := range loraPaths {
		scale := float32(1)
		if *loraDisabled {
			scale = 0
		}
		loraAdapters[index] = inference.LoRAConfig{Path: path, Scale: scale}
	}
	runner, err := inference.OpenWithOptions(flag.Arg(0), inference.OpenOptions{
		DeviceOrdinal:           *deviceOrdinal,
		PreloadDeviceWeights:    *preload,
		PreloadQuantizedWeights: *nativeQ8 || *nativeQuant,
		PromptCacheEntries:      *promptCacheEntries,
		LoRAAdapters:            loraAdapters,
	})
	if err != nil {
		return err
	}
	defer runner.Close()
	var vision projector.ImageProjector
	var audio projector.AudioProjector
	if *projectorPath != "" {
		vision, err = projector.OpenImageProjectorWithOptions(*projectorPath, projector.OpenOptions{
			CUDA: *projectorCUDA, DeviceOrdinal: *deviceOrdinal,
		})
		if err != nil {
			return fmt.Errorf("open multimodal projector: %w", err)
		}
		defer vision.Close()
		audio, _ = vision.(projector.AudioProjector)
	}
	apiKey := strings.TrimSpace(os.Getenv("LLAMACPP2GO_API_KEY"))
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
		DefaultTemperature: 1,
		DefaultTopP:        1,
		DefaultTopK:        40,
		APIKey:             apiKey,
		ContextShift:       *contextShift,
		RequestTimeout:     *requestTimeout,
		InfillBatchSize:    *infillBatchSize,
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
	}, runner)
	if err != nil {
		return err
	}
	defer handler.Close()
	httpServer := &http.Server{
		Addr:              *address,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      0,
		IdleTimeout:       2 * time.Minute,
		MaxHeaderBytes:    1 << 20,
	}
	log.Printf("serving model %q on http://%s", *modelID, *address)
	shutdownContext, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
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
	deadline, cancel := context.WithTimeout(context.Background(), 30*time.Second)
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
