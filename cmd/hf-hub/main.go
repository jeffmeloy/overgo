// hf-hub is the command-line face of Hugging Face intake: search the hub,
// resolve a repository revision's file inventory, and download with size and
// digest verification. The workbench's discovery surface drives the same
// client through the server; this command is the scriptable path and the
// production proof that the client's contract holds outside tests.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"overgo/internal/clioptions"
	"overgo/internal/hfhub"
)

func main() {
	clioptions.MainNamed("hf-hub", run)
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	return runArguments(ctx, os.Args[1:])
}

func runArguments(ctx context.Context, args []string) error {
	if len(args) < 1 {
		return errors.New("usage: hf-hub <search|resolve|download> [options] <query|repository>")
	}
	verb := args[0]
	flags := flag.NewFlagSet("hf-hub "+verb, flag.ContinueOnError)
	endpoint := flags.String("endpoint", "", "hub endpoint; empty means the public hub")
	token := flags.String("token", os.Getenv("OVERGO_HF_TOKEN"), "bearer token (default OVERGO_HF_TOKEN)")
	datasets := flags.Bool("datasets", false, "address the dataset namespace instead of models")
	revision := flags.String("revision", "", "repository revision; empty resolves the default branch")
	destination := flags.String("dest", "", "download destination directory")
	timeout := flags.Duration("timeout", 0, "total caller budget; zero runs until completion or interrupt")
	limit := flags.Int("limit", 20, "search result bound")
	filter := flags.String("filter", "", "hub tag filter, e.g. gguf")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	if *timeout < 0 {
		return errors.New("hf-hub: timeout must be non-negative")
	}
	if *timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeoutCause(ctx, *timeout, fmt.Errorf("hf-hub caller budget: %w", context.DeadlineExceeded))
		defer cancel()
	}
	client, err := hfhub.New(*endpoint, *token)
	if err != nil {
		return err
	}
	kind := hfhub.KindModel
	if *datasets {
		kind = hfhub.KindDataset
	}
	subject := strings.Join(flags.Args(), " ")
	switch verb {
	case "search":
		listings, err := client.Search(ctx, hfhub.SearchQuery{Kind: kind, Search: subject, Filter: *filter, Limit: *limit})
		if err != nil {
			return err
		}
		for _, listing := range listings {
			gated := ""
			if listing.Gated {
				gated = " gated"
			}
			fmt.Printf("%-48s downloads=%d likes=%d%s\n", listing.ID, listing.Downloads, listing.Likes, gated)
		}
		return nil
	case "resolve":
		resolved, err := client.Resolve(ctx, kind, subject, *revision)
		if err != nil {
			return err
		}
		fmt.Printf("%s @ %s\n", resolved.ID, resolved.Revision)
		for _, file := range resolved.Files {
			digest := ""
			if file.SHA256 != "" {
				digest = " sha256=" + file.SHA256
			}
			fmt.Printf("  %-56s %12d bytes%s\n", file.Path, file.Size, digest)
		}
		return nil
	case "download":
		if strings.TrimSpace(*destination) == "" {
			return errors.New("hf-hub download requires -dest")
		}
		resolved, err := client.Download(ctx, hfhub.DownloadRequest{
			Kind: kind, Repository: subject, Revision: *revision, Destination: *destination,
			Observe: reportProgress(),
		})
		if err != nil {
			return err
		}
		fmt.Printf("\ndownloaded %s @ %s: %d files into %s\n", resolved.ID, resolved.Revision, len(resolved.Files), *destination)
		return nil
	default:
		return fmt.Errorf("hf-hub: unknown verb %q", verb)
	}
}

// reportProgress renders one advancing line per file without flooding the
// terminal: a report lands only when the percentage moves.
func reportProgress() func(hfhub.Progress) {
	lastPath, lastPercent := "", int64(-1)
	return func(progress hfhub.Progress) {
		if progress.Total <= 0 {
			return
		}
		percent := progress.Received * 100 / progress.Total
		if progress.Path == lastPath && percent == lastPercent {
			return
		}
		if progress.Path != lastPath {
			if lastPath != "" {
				fmt.Println()
			}
			lastPath = progress.Path
		}
		lastPercent = percent
		fmt.Printf("\r%-56s %3d%%", progress.Path, percent)
	}
}
