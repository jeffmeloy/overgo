// Command remote-provider declares hosted models in the store and lists
// them. A declaration file names the provider (its service name, its
// OpenAI-compatible API endpoint and the environment variable holding
// the key) and the model ids to serve through it; each model becomes a
// store manifest at a remote location with an active remote inference
// recipe, so the servable catalog lists it beside local models and the
// server relays to it. The listing shows every declared model with the
// refusal its provider carries when the key is absent.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/remoteprovider"
	"overgo/internal/runrecord"
)

// catalogListingLimit bounds the model manifests one listing reads, the
// bound the server's generation catalog reads under.
const catalogListingLimit = 256

// declaration is the file form of one provider and the models it serves.
type declaration struct {
	Name           string   `json:"name"`
	Endpoint       string   `json:"endpoint"`
	KeyEnvironment string   `json:"key_environment"`
	Models         []string `json:"models"`
	ContextLength  uint32   `json:"context_length,omitzero"`
}

func main() {
	clioptions.Main(func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("remote-provider", flag.ContinueOnError)
	flags.SetOutput(output)
	repository := flags.String("repo", "", "OvergoDB store; empty resolves via the data-root contract")
	declare := flags.String("declare", "", "strict JSON declaration: name, endpoint, key_environment, models, context_length")
	list := flags.Bool("list", false, "list the store's remote models with their refusals")
	limit := flags.Int("limit", catalogListingLimit, "maximum models the listing reads")
	if err := flags.Parse(args); err != nil {
		return err
	}
	store, err := openStore(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	ctx := context.Background()
	if *declare != "" {
		var document declaration
		if err := jsonfile.DecodeStrict(*declare, &document); err != nil {
			return err
		}
		commit, err := runrecord.ExecutableCodeCommit(".")
		if err != nil {
			return err
		}
		for _, model := range document.Models {
			declared, err := remoteprovider.Declare(ctx, store, remoteprovider.Provider{
				Name: document.Name, Endpoint: document.Endpoint, KeyEnvironment: document.KeyEnvironment,
				Model: model, ContextLength: document.ContextLength,
			}, commit)
			if err != nil {
				return err
			}
			fmt.Fprintf(output, "declared %s model=%s recipe=%s %s\n", declared.Location, declared.Model, declared.Recipe.ID, servability(declared.Provider))
		}
		return nil
	}
	if !*list {
		return fmt.Errorf("remote-provider: usage: remote-provider [-repo <store>] -declare <file> | -list")
	}
	declared, err := remoteprovider.List(ctx, store, *limit)
	if err != nil {
		return err
	}
	for _, model := range declared {
		fmt.Fprintf(output, "%s model=%s %s\n", model.Location, model.Model, servability(model.Provider))
	}
	return nil
}

func servability(provider remoteprovider.Provider) string {
	if refusal := remoteprovider.Refusal(provider); refusal != "" {
		return "refused: " + refusal
	}
	return "servable"
}

func openStore(repository string) (*overgodb.Store, error) {
	if repository == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return nil, err
		}
		repository = roots.Store
	}
	return overgodb.Open(repository)
}
