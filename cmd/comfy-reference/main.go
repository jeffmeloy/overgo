// Command comfy-reference turns a ComfyUI workflow, as ComfyUI's page saves it, into the API request ComfyUI runs for
// one of its sections, and keeps the workflow and ComfyUI's own requests for it as a fixture.
//
// ComfyUI runs as a separate process; this command reads only the files its page saves and the answers of its HTTP API.
//
//	comfy-reference -import-workflow MOSS_IMAGE_ULTRA_WORKFLOW.json -import-object-info object_info.json -import-saved <user>/default
//	comfy-reference -section "TEXT TO IMAGE"
package main

import (
	"encoding/json"
	"flag"
	"fmt"

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
	flag.Parse()
	return act(*fixture, *workflow, *objectInfo, *saved, *section)
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
