// Package commanddoc is the single source for a shipped command's help identity
// and its API-manifest discovery classification, so a command's help output and
// its manifest projection share one owner-declared declaration rather than
// drifting between a help string and a parallel manifest registry.
package commanddoc

import "overgo/internal/clioptions"

// Descriptor pairs the command's help identity with the caller classification the
// manifest records. Purpose, audience, and constraints render in help; the
// classification is manifest-only, decided by the command's actual callers.
type Descriptor struct {
	Command        clioptions.Command
	Classification string
}

// Plan describes the plan command: the master-lead session and its unattended
// driver dispatch through it, and operators inspect and steer it, so it is
// called both ways.
var Plan = Descriptor{
	Command: clioptions.Command{
		Name:     "plan",
		Purpose:  "edit and dispatch the campaign in docs/plan.json; a landing (cmd/loop -land) verifies and advances it",
		Audience: "the master-lead session and its unattended driver working one dispatched row at a time",
		Constraints: []string{
			"-add takes the step's verify command through -vcmd; boolean -verify runs the top open step's verify",
			"one operation per invocation; committing and advancing a row belong to cmd/loop -land, not plan",
			"help opens no store and mutates no plan",
			"scratch probes live in tmp (module overgo/tmp, replace overgo => ..): run one with go run ./tmp/<file>.go from the repository root or go run ./<file>.go from tmp; they stay outside ./...",
		},
	},
	Classification: "both",
}

// OvergodbQuery describes the overgodb-query command: an operator or an agent
// reads the committed catalog through it without mutating anything.
var OvergodbQuery = Descriptor{
	Command: clioptions.Command{
		Name:     "overgodb-query",
		Purpose:  "read the catalog: filtered artifact listings, lineage traversals, and the derived operational ledgers",
		Audience: "an operator or agent inspecting committed OvergoDB state without mutating it",
		Constraints: []string{
			"-repo names the store (or empty for the data-root contract) and -limit is a positive result bound",
			"-content prints the raw committed bytes of the artifact named by -id",
			"help opens no store and reads nothing",
		},
	},
	Classification: "both",
}

// operator declares a command an operator runs by hand, whose help identity
// is its purpose and audience alone.
func operator(name, purpose, audience string) Descriptor {
	return Descriptor{Command: clioptions.Command{Name: name, Purpose: purpose, Audience: audience}, Classification: "operator-facing"}
}

// Declared is every command whose caller the tree does not otherwise show:
// the manifest projects each identity, and a command nothing names and
// nothing declares is refused by TestEveryCommandHasACaller.
var Declared = map[string]Descriptor{
	"plan":           Plan,
	"overgodb-query": OvergodbQuery,
	"seam-align": operator("seam-align", "compute a seam's alignment residual, the causal proxy for whether a linear adapter composes it, and bias-audit recorded residuals against measured bridge parity",
		"an operator studying model composition before training a bridge"),
	"audio-inspect": operator("audio-inspect", "measure local WAV or FLAC audio and publish its admission record",
		"an operator admitting audio sources for speech datasets"),
	"latentvideo-run": operator("latentvideo-run", "run resident Wan denoising and causal decode and record the clip evidence",
		"an operator producing and checking Wan video clips"),
	"mechanism-census": operator("mechanism-census", "publish the exact mechanism ownership and gap census",
		"an operator auditing which mechanisms the tree owns"),
	"buildscrub": operator("buildscrub", "report, and with -apply remove, build/ artifacts that build/RETAINED.json does not declare",
		"an operator keeping build/ to declared artifacts"),
	"hf-hub": operator("hf-hub", "search the Hugging Face hub, resolve a revision's files and download them with size and digest verification",
		"an operator bringing models in from the hub"),
	"activation-cases": operator("activation-cases", "publish and inspect typed behavioral case denominators",
		"an operator curating activation behaviour cases"),
	"inspect-safetensors": operator("inspect-safetensors", "list a safetensors model directory's tensors and configuration",
		"an operator examining a checkpoint before conversion"),
	"dataset-catalog": operator("dataset-catalog", "publish the legacy dataset inventory into OvergoDB",
		"an operator migrating dataset records into the store"),
	"remote-provider": operator("remote-provider", "declare hosted OpenAI-compatible models in the store and list them",
		"an operator connecting hosted models"),
	"sealed-authority": operator("sealed-authority", "serve a signed-request authority over its own OvergoDB",
		"an operator running the sealed authority service"),
	"tool-workflow": operator("tool-workflow", "compile and execute one closed-world tool workflow whose manuals resolve from the store",
		"an operator running an agent tool workflow from the command line"),
	"model-config": operator("model-config", "record a model's provided generation configuration and recommended sampling in the store",
		"an operator onboarding a checkpoint"),
	"router-observation": operator("router-observation", "publish one exact per-layer MoE router observation",
		"an operator measuring mixture-of-experts routing"),
	"model-build": operator("model-build", "train a scratch model from a JSON corpus through the model builder",
		"an operator building a small model from scratch"),
	"profile-catalog": operator("profile-catalog", "publish the architecture profile registry",
		"an operator refreshing the architecture registry in the store"),
	"rerank": operator("rerank", "score a document against a query with a Qwen3 or Qwen3-VL reranker GGUF",
		"an operator ranking documents with a reranker model"),
	"embedding": operator("embedding", "encode text with a GGUF encoder model and print its embedding", "an operator embedding text"),
	"tokenize":  operator("tokenize", "tokenize text with a GGUF model's tokenizer", "an operator inspecting tokenization"),
	"perplexity": operator("perplexity", "measure a GGUF model's perplexity over UTF-8 text",
		"an operator comparing model quality on a text"),
	"model-info":  operator("model-info", "print a GGUF model's metadata and architecture", "an operator inspecting a model file"),
	"block-check": operator("block-check", "check a GGUF model's blocks against their declared shapes and types", "an operator validating a model file"),
	"gguf": operator("gguf", "hash, inspect, merge, split or quantize a GGUF model file, one subcommand each",
		"an operator handling GGUF model files"),
	"gemma4-gguf-convert": operator("gemma4-gguf-convert", "convert a Gemma 4 checkpoint into language-model and projector GGUF files",
		"an operator onboarding a Gemma 4 checkpoint"),
	"json-schema-grammar": operator("json-schema-grammar", "compile a JSON schema into the sampling grammar that constrains generation",
		"an operator checking structured-output constraints"),
	"agent-loop": operator("agent-loop", "propose one agent tool step through the coordinator and print its recorded result",
		"an operator driving an agent step by hand"),
	"audio-oracle": operator("audio-oracle", "verify and publish one immutable external audio-oracle election",
		"an operator electing an audio reference"),
	"controller-action": operator("controller-action", "compile one allowlisted controller action into its artifacts and commit them",
		"an operator executing a controller emission"),
	"hf-gguf-convert": operator("hf-gguf-convert", "convert a Hugging Face checkpoint to GGUF and register the result",
		"an operator onboarding a model"),
	"overgodb-backup": operator("overgodb-backup", "publish a verified copy of an OvergoDB store",
		"an operator backing up the store"),
	"overgodb-import": operator("overgodb-import", "ingest a wire-format import stream into the store",
		"an operator importing records into the store"),
	"sensenovaparity": operator("sensenovaparity", "check SenseNova stages against their fixtures and the real checkpoint",
		"an operator checking the SenseNova port"),
	"api-manifest": operator("api-manifest", "regenerate or check docs/api_manifest.json, the declared binaries, routes and contracts",
		"a worker whose change moves the declared API surface"),
	"swap": operator("swap", "serve every servable model in the store from one endpoint, swapping the running server",
		"an operator serving several models"),
	"training-objective": operator("training-objective", "bind a training objective's contract identities and evidence note",
		"an operator declaring a training objective"),
}
