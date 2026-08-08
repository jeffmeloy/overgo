// guard: PreToolUse hook binary. Reads the harness tool-call JSON on stdin;
// exit 2 with a stderr reason denies the call, exit 0 allows. Fail-open on
// unparseable input so a hiccup never bricks a session — the corpus, not this
// binary, is where strictness lives.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"overgo/internal/guard"
)

func main() {
	var call guard.ToolCall
	if err := json.NewDecoder(os.Stdin).Decode(&call); err != nil {
		os.Exit(0)
	}
	root, err := os.Getwd()
	if err != nil {
		os.Exit(0)
	}
	message, wrapped := guard.Verdict(call, root)
	if message == "" {
		os.Exit(0)
	}
	if wrapped {
		message += " (the guard unwrapped a shell -c payload and judged it as the command it is)"
	}
	fmt.Fprintf(os.Stderr, "BLOCKED by guard: %s -- do NOT retry with force; if a delete is truly needed, ask the user.\n", message)
	os.Exit(2)
}
