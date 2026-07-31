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
	apiKeyFile := flag.String(
		"api-key-file",
		"",
		"read the /v1 bearer token from this file (or LLAMACPP2GO_API_KEY)",
	)
	flag.Parse()
	if flag.NArg() != 1 {
		return errors.New("usage: server [options] <model.gguf>")
	}
	runner, err := inference.OpenWithOptions(flag.Arg(0), inference.OpenOptions{
		DeviceOrdinal:           *deviceOrdinal,
		PreloadDeviceWeights:    *preload,
		PreloadQuantizedWeights: *nativeQ8 || *nativeQuant,
		PromptCacheEntries:      *promptCacheEntries,
	})
	if err != nil {
		return err
	}
	defer runner.Close()
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
	}, runner)
	if err != nil {
		return err
	}
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
