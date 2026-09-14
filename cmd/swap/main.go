//overgo:runtime-inputs caller

// Command swap serves every servable model in the store from one
// endpoint: requests route by their model field, the supervisor swaps
// which child overgo server runs, and the workbench rides through
// unchanged. llama-swap is the reference for the lifecycle; the
// store's servable catalog is the configuration.
package main

import (
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/libraryintake"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelswap"
	"overgo/internal/overgodb"
	"overgo/internal/providerintake"
	"overgo/internal/recipe"
	"overgo/internal/server"
)

func main() {
	clioptions.MainNamed("swap", run)
}

func run() error {
	listen := flag.String("listen", "127.0.0.1:8080", "proxy listen address")
	store := flag.String("repo", "overgodb-store", "OvergoDB root the children serve over")
	binary := flag.String("server", "bin/overgo-server.exe", "built overgo server binary the supervisor launches")
	idle := flag.Duration("idle", 0, "stop an unreferenced child after this idle time; 0 keeps it resident")
	catalogLimit := flag.Int("catalog-limit", 256, "servable catalog listing bound")
	defaultModel := flag.String("default", "", "model served for model-less requests before any child runs")
	training := flag.Bool("training", false, "enable the recipe-bound training workspace in every served child")
	modelBuilder := flag.Bool("model-builder", false, "enable the corpus-derived model builder workspace in every served child")
	flag.Parse()
	// The proxy forwards requests to its children without authenticating
	// them itself, so it must never listen beyond this host.
	if err := clioptions.RequireLoopbackWithoutCredential(*listen, ""); err != nil {
		return err
	}
	repository, err := overgodb.Open(*store)
	if err != nil {
		return err
	}
	defer repository.Close()
	resolver := &modelswap.CatalogResolver{Store: repository, Limit: *catalogLimit}
	launcher := modelswap.ServerLauncher{Binary: *binary, Store: *store, Workspaces: modelswap.Workspaces{Training: *training, ModelBuilder: *modelBuilder}}
	supervisor, err := modelswap.New(launcher, *idle)
	if err != nil {
		return err
	}
	defer supervisor.Close()
	// The cold start: with no default and no child the proxy serves the shell itself; the picker launches the first child,
	// and the Library shares the retained repository with the catalog resolver. The workbench behind the shell answers
	// downloads, validation and operations over the store, so an empty store reaches a served model without a child.
	intake := providerintake.Intake{CatalogLimit: *catalogLimit}.Library(libraryintake.ModelFiles, libraryintake.Register)
	intake.Validate = libraryintake.Validate
	workbench, err := idleWorkbench(repository, intake, *catalogLimit)
	if err != nil {
		return err
	}
	defer workbench.Close()
	shell := &server.IdleShell{
		Catalog:    resolver.Catalog,
		Repository: repository,
		Intake:     intake,
		Workbench:  workbench,
	}
	proxy := &modelswap.Proxy{Supervisor: supervisor, Resolver: resolver, Keys: resolver, Idle: shell}
	if *defaultModel != "" {
		if fileExists(*defaultModel) {
			// An on-disk model file is launchable directly -- no store
			// replay needed to start serving it.
			proxy.Default = modelswap.Servable{Name: filepath.Base(*defaultModel), Location: *defaultModel}
		} else {
			servable, found, err := resolver.Resolve(context.Background(), *defaultModel)
			if err != nil || !found {
				return fmt.Errorf("swap: default model %q is neither an on-disk file nor servable: %v", *defaultModel, err)
			}
			proxy.Default = servable
		}
	}
	server := &http.Server{Addr: *listen, Handler: proxy, ReadHeaderTimeout: 30 * time.Second}
	fmt.Printf("swap proxy on http://%s over %s\n", *listen, *store)
	return server.ListenAndServe()
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// idleWorkbench opens the store's workbench for the cold proxy: hub
// downloads into the data root's models directory, library validation
// through the model intake, and the operations those produce.
func idleWorkbench(repository *overgodb.Store, intake server.LibraryIntake, catalogLimit int) (*server.Handler, error) {
	policy, found, err := modelrecipe.CatalogRuntimePolicy(recipe.TaskInference)
	if err != nil || !found {
		return nil, fmt.Errorf("swap: inference runtime policy: %w", cmp.Or(err, errors.New("absent from the catalog")))
	}
	hubRoot := ""
	if roots, err := dataroot.Resolve("."); err == nil {
		hubRoot = roots.Models
	}
	// Media recipes execute through the existing store workspace without loading a chat model.
	generation := server.NewStoreGenerationWorkspace(repository, server.BindGenerationCatalog(mediacapability.Catalog, mediacapability.Controls, mediacapability.OutputContent), catalogLimit)
	return server.New(server.Config{
		RuntimePolicy: policy, Repository: repository, LibraryIntake: intake,
		HubToken: os.Getenv("OVERGO_HF_TOKEN"), HubDownloadRoot: hubRoot,
	}, &idleRuntime{GenerationRefused: server.GenerationRefused{Reason: "Choose a chat model to send a conversation."}, WorkflowWorkspaceAPI: generation})
}

// idleRuntime retains explicit text refusal alongside admitted media workflows.
type idleRuntime struct {
	server.GenerationRefused
	server.WorkflowWorkspaceAPI
}
