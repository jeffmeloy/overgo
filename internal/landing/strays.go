package landing

import (
	"bytes"
	"context"
	"path"
	"slices"
	"strings"

	"overgo/internal/gitauthority"
	"overgo/internal/processlock"
	"overgo/internal/protection"
)

// documentHomes are where an ignored JSON or Markdown file belongs in a checkout: a row's probes under tmp, the
// system's own state, the toolchain, the build's outputs, and the harness's own directory, whose runtime files the
// harness writes (SQA-334: its scheduled tasks file would have refused every landing). documentFiles are the ignored ones the checkout keeps by
// name: the machine's model roots, the preview's launch list, and the plan the GUI lane left on 2026-10-09, which its
// owner has yet to take.
var (
	documentHomes = []string{"tmp/", processlock.StateDirectory + "/", ".tools/", "build/", protection.HarnessConfigDirectory}
	documentFiles = []string{"local-models.json", protection.HarnessConfigDirectory + "launch.json", "docs/gui_update_plan.json"}
	documentKinds = []string{".json", ".md"}
)

// strayDocuments are the JSON and Markdown files git ignores in the checkout at repo outside documentHomes and
// documentFiles: a draft or a hand-off kept out of the commits, which a later session reads as if it held rules or
// plans. A file that is not ignored is the change's own and lands with it.
func strayDocuments(ctx context.Context, repo string) ([]string, error) {
	listed, err := gitauthority.Query(ctx, repo, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
	if err != nil {
		return nil, err
	}
	var strays []string
	for entry := range bytes.SplitSeq(listed, []byte{0}) {
		name := string(entry)
		if !slices.Contains(documentKinds, path.Ext(name)) || slices.Contains(documentFiles, name) ||
			slices.ContainsFunc(documentHomes, func(home string) bool { return strings.HasPrefix(name, home) }) {
			continue
		}
		strays = append(strays, name)
	}
	return strays, nil
}

// strayRefusal is what the rules stage refuses a change with while a stray document stands in the checkout.
func strayRefusal(strays []string, item string) string {
	return "an ignored document outside its homes stands in the checkout: " + strings.Join(strays, ", ") +
		"; a draft or hand-off belongs in the plan (plan -add -from -, plan -note) or under tmp/" + item + "/; edit: mkdir -p tmp/" + item +
		" && mv " + strings.Join(strays, " ") + " tmp/" + item + "/"
}
