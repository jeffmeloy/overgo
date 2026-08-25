// Command webui-lane runs Overgo's required real-browser GUI acceptance lane.
package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"

	"overgo/internal/clioptions"
	"overgo/internal/webuilane"
)

func main() {
	clioptions.MainNamed("webui lane", run)
}

func run() error {
	browser, err := webuilane.FindBrowser(os.Getenv("OVERGO_BROWSER"))
	if err != nil {
		return err
	}
	probe, err := webuilane.Open(context.Background(), browser, "data:text/html,<title>overgo-webui-lane</title>")
	if err != nil {
		return err
	}
	var title string
	if err := probe.SetViewport(context.Background(), len(browser), len(browser)); err == nil {
		err = probe.Evaluate(context.Background(), "document.title", &title)
	}
	_ = probe.Close()
	if err != nil {
		return fmt.Errorf("browser transport self-check failed: %w", err)
	}
	if title != "overgo-webui-lane" {
		return fmt.Errorf("browser transport self-check returned title %q", title)
	}
	command := exec.CommandContext(
		context.Background(), "go", "test", "./internal/server",
		"-run", "^TestWebUIBrowserAcceptance$", "-count=1", "-timeout=2m", "-v",
	)
	command.Env = append(os.Environ(), "OVERGO_WEBUI_LANE=1", "OVERGO_BROWSER="+browser)
	command.Stdout, command.Stderr = os.Stdout, os.Stderr
	if err := command.Run(); err != nil {
		return err
	}
	fmt.Printf("webui lane: PASS browser=%s\n", browser)
	return nil
}
