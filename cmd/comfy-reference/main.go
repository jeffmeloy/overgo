// Command comfy-reference turns a ComfyUI workflow, as ComfyUI's page saves it, into the API request ComfyUI runs for
// one of its sections, and keeps the workflow and ComfyUI's own requests for it as a fixture.
//
// ComfyUI runs as a separate process; this command reads only the files its page saves and the answers of its HTTP API.
//
//	comfy-reference -import-workflow MOSS_IMAGE_ULTRA_WORKFLOW.json -import-object-info object_info.json -import-saved <user>/default
//	comfy-reference -section "TEXT TO IMAGE"
//	comfy-reference -record "TEXT TO IMAGE" -install <portable>/ComfyUI -out <folder> [-set node.input=<JSON>] [-save-latent node]
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"path/filepath"
	"time"

	"overgo/internal/clioptions"
)

func main() {
	clioptions.Main(run)
}

func run() error {
	fixture := flag.String("fixture", ultraFixture, "the fixture folder the workflow, its hash, the node definitions, the sections and ComfyUI's requests are kept in")
	workflow := flag.String("import-workflow", "", "a workflow as ComfyUI's page saves it, copied into the fixture with its hash")
	objectInfo := flag.String("import-object-info", "", "ComfyUI's /object_info answer, kept for the node types the workflow uses")
	saved := flag.String("import-saved", "", "the folder ComfyUI's page saved sections.json and api_prompt_<section>.json to")
	section := flag.String("section", "", "print the API request of this section of the fixture's workflow")
	record := flag.String("record", "", "run this section of the fixture's workflow in the ComfyUI at -server, and keep its recording and files in -out")
	server := flag.String("server", "http://127.0.0.1:8189", "with -record: the ComfyUI the section runs in, listening on the machine only")
	install := flag.String("install", "", "with -record: that ComfyUI's folder, whose commit and custom-node packs' commits the recording names")
	out := flag.String("out", "", "with -record: the folder the recording and its files are kept in")
	var set, latents []string
	flag.Func("set", "with -record: an input given over the section's own, node.input=<JSON value>; repeatable", func(value string) error {
		set = append(set, value)
		return nil
	})
	flag.Func("save-latent", "with -record: a node whose first output, a latent, is saved beside the images; repeatable", func(node string) error {
		latents = append(latents, node)
		return nil
	})
	flag.Parse()
	if *record == "" {
		return act(*fixture, *workflow, *objectInfo, *saved, *section)
	}
	if *install == "" || *out == "" {
		return errors.New("comfy-reference: -record needs -install and -out")
	}
	ultra, err := readFixture(*fixture)
	if err != nil {
		return err
	}
	taken, err := recorder{ultra: ultra, server: *server, install: *install, out: *out, set: set, latents: latents, poll: time.Second}.record(context.Background(), *record)
	if err != nil {
		return err
	}
	_, err = fmt.Printf("comfy-reference: recorded %q in %d ms, %d files, at most %d bytes of the card in use; %s\n",
		taken.Section, taken.Milliseconds, len(taken.Files), taken.PeakCardBytes, filepath.Join(*out, slug(taken.Section)+".json"))
	return err
}

func act(fixture, workflow, objectInfo, saved, section string) error {
	if workflow != "" {
		if err := importFixture(fixture, workflow, objectInfo, saved); err != nil {
			return err
		}
		fmt.Println("comfy-reference: kept", workflow, "and ComfyUI's requests for its sections in", fixture)
	}
	if section == "" {
		return nil
	}
	ultra, err := readFixture(fixture)
	if err != nil {
		return err
	}
	request, err := ultra.request(section)
	if err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(request, "", " ")
	if err != nil {
		return err
	}
	_, err = fmt.Println(string(encoded))
	return err
}
