// Command swap serves every servable model in the store from one
// endpoint: requests route by their model field, the supervisor swaps
// which child overgo server runs (one at a time -- the store's writer
// lock admits one serving process), and the workbench rides through
// unchanged. llama-swap is the reference for the lifecycle; the
// store's servable catalog is the configuration.
package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/modelswap"
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
	flag.Parse()
	resolver := &modelswap.CatalogResolver{Store: *store, Limit: *catalogLimit}
	supervisor, err := modelswap.New(modelswap.ServerLauncher{Binary: *binary, Store: *store}, *idle)
	if err != nil {
		return err
	}
	defer supervisor.Close()
	proxy := &modelswap.Proxy{Supervisor: supervisor, Resolver: resolver}
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
