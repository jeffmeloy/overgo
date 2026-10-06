package main

import (
	"context"
	"fmt"
	"io"
	"os"

	"overgo/internal/landing"
)

// land runs the small gate over the working tree for one plan row. A message
// file of "-" is read from standard input.
func land(reference, messageFile string, preflight bool) error {
	var message []byte
	var err error
	if messageFile == "-" {
		message, err = io.ReadAll(os.Stdin)
	} else {
		message, err = os.ReadFile(messageFile)
	}
	if err != nil {
		return err
	}
	commit, err := landing.Run(context.Background(), landing.Options{Repo: ".", Ref: reference, Message: message, Log: os.Stderr, Preflight: preflight, Quiet: true})
	if err != nil {
		return err
	}
	if preflight {
		fmt.Printf("land: %s passes the cheap stages; nothing was committed\n", reference)
		return nil
	}
	fmt.Printf("land: %s landed as %s\n", reference, commit)
	return nil
}
