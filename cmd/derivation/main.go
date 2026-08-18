// Command derivation is the owner tool for derivation-profile artifacts: it
// registers corpus-to-topology policies as content-addressed documents and
// judges candidate-versus-incumbent promotions by the descendant metric
// contract. Profiles argue only through the models they produce.
//
//	go run ./cmd/derivation -register <profile.json> -record <repodb>
//	go run ./cmd/derivation -promote <candidate-id> -incumbent <incumbent-id> -contract <contract.json> -record <repodb>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/jsonfile"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "derivation:", err)
		os.Exit(1)
	}
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("derivation", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	register := flags.String("register", "", "path to a derivation-profile JSON to identify and commit")
	promote := flags.String("promote", "", "candidate derivation-profile artifact ID to judge")
	incumbent := flags.String("incumbent", "", "incumbent derivation-profile artifact ID")
	contractPath := flags.String("contract", "", "path to the descendant metric contract JSON (per-seed incumbent/candidate model quality)")
	recordStore := flags.String("record", "", "RepoDB root")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *recordStore == "" || (*register == "") == (*promote == "") {
		return errors.New("usage: derivation -register <profile.json> -record <repodb> | derivation -promote <id> -incumbent <id> -contract <contract.json> -record <repodb>")
	}
	store, err := repodb.Open(*recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	if *register != "" {
		var profile scratchmodel.DerivationProfile
		if err := jsonfile.Decode(*register, &profile); err != nil {
			return err
		}
		document, err := scratchmodel.NewDerivationProfileDocument(profile)
		if err != nil {
			return err
		}
		content, err := document.Content()
		if err != nil {
			return err
		}
		if _, err := store.Commit(ctx, artifact.Batch{
			Key: "derivation/profile/" + document.ID.String(), Contents: []artifact.Content{content},
		}); err != nil {
			return err
		}
		fmt.Fprintf(output, "derivation profile committed: %s version=%s\n", document.ID, document.Profile.Version)
		return nil
	}
	if *incumbent == "" || *contractPath == "" {
		return errors.New("promotion requires -incumbent and -contract")
	}
	read := func(text string) (scratchmodel.DerivationProfileDocument, error) {
		id, err := artifact.ParseID(text)
		if err != nil {
			return scratchmodel.DerivationProfileDocument{}, err
		}
		content, ok, err := store.Content(ctx, id)
		if err != nil {
			return scratchmodel.DerivationProfileDocument{}, err
		}
		if !ok {
			return scratchmodel.DerivationProfileDocument{}, fmt.Errorf("profile %s is not committed", text)
		}
		return scratchmodel.ParseDerivationProfileDocument(content.Data)
	}
	candidateDocument, err := read(*promote)
	if err != nil {
		return err
	}
	incumbentDocument, err := read(*incumbent)
	if err != nil {
		return err
	}
	var contract runrecord.MetricContract
	if err := jsonfile.Decode(*contractPath, &contract); err != nil {
		return err
	}
	promotion, err := scratchmodel.PromoteDerivationProfile(candidateDocument, incumbentDocument, contract)
	if err != nil {
		return err
	}
	batch, err := promotion.Batch("derivation/promotion/" + promotion.ID.String())
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	verdict := "REFUSED"
	if promotion.Promoted {
		verdict = "PROMOTED"
	}
	fmt.Fprintf(output, "derivation promotion committed: %s (%s)\n", promotion.ID, verdict)
	fmt.Fprintf(output, "reason: %s\n", promotion.Reason)
	fmt.Fprintln(output, "honesty: profiles are judged solely by their descendants' measured quality")
	return nil
}
