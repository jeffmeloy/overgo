"""Golden reference for needle's served path, from the model's own code.

Runs the upstream needle package (cactus-compute/needle, as shipped beside the
weights) over the cases of overgo's function-call evaluation and rewrites
fixtures/needle_upstream_calls.json with, for each case:

  encoder_ids        the encoder input upstream builds for the request and its
                     tools as given: the request's ids, the tools marker, the
                     tools' ids, each text tokenized by itself
  named_tools        the tools after upstream's normalize_tools
  named_encoder_ids  the encoder input for those, which is what a served
                     request feeds the model
  plain_f32          the plain greedy answer with tool names as given, the
                     parameters as upstream loads them, rounded to bfloat16,
                     and the arithmetic in float32
  named_f32          the answer upstream's generate returns with names
                     normalized and its constrained decoding off, in that
                     arithmetic
  named              the same in upstream's own arithmetic, bfloat16 throughout
  served             what upstream's generate returns by default: names
                     normalized, decoding constrained to the tools, bfloat16

Upstream loads its checkpoint as bfloat16 and computes in bfloat16. A port
that holds the same bfloat16 parameters and computes in float32, as overgo
does from the shipped safetensors file, is held to plain_f32 and named_f32 to
the character. The bfloat16 answers differ from those wherever two next
tokens are close enough for the rounding of an activation to choose between
them, which is the measure of how many answers turn on arithmetic alone.
Nothing is downloaded: the checkpoint and the tokenizer are the files in the
model directory.

    <python with jax, flax and sentencepiece> -B needle_upstream_calls.py <model directory> <fixture.json> [cases.json]

The model directory holds needle.pkl, tokenizer.model and the upstream code
under needle/. The fixture's cases (name, query, tools) are kept, and so are
the bfloat16 answers it already holds, which take most of an hour and a half
to write: delete the fixture's "named" and "served" to write them again. A
fixture that does not exist yet is started from a JSON list of cases with a
prompt of the form "<query> <tools> <tools json>" given as a third argument.
"""
import json
import pickle
import sys
import time

sys.dont_write_bytecode = True
MODEL, FIXTURE = sys.argv[1], sys.argv[2]
sys.path.insert(0, MODEL + "/needle")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from needle.dataset.tokenizer import NeedleTokenizer  # noqa: E402
from needle.model.architecture import SimpleAttentionNetwork, TransformerConfig  # noqa: E402
from needle.model.run import _build_encoder_input, generate_batch, load_checkpoint, normalize_tools  # noqa: E402

GENERATED = 256  # the tokens the evaluation lets an answer run to
BATCH = 40
KEPT = ("name", "query", "tools", "named", "served")

if len(sys.argv) > 3:
    cases = []
    for case in json.load(open(sys.argv[3], encoding="utf-8")):
        query, separator, tools = case["prompt"].partition(" <tools> ")
        assert separator, case["name"]
        cases.append({"name": case["name"], "query": query, "tools": tools})
else:
    cases = [{key: case[key] for key in KEPT if key in case} for case in json.load(open(FIXTURE, encoding="utf-8"))["cases"]]

tokenizer = NeedleTokenizer(MODEL + "/tokenizer.model")
native_params, native_config = load_checkpoint(MODEL + "/needle.pkl")
native = SimpleAttentionNetwork(native_config)
with open(MODEL + "/needle.pkl", "rb") as checkpoint:
    stored = pickle.load(checkpoint)
# The parameters as upstream loads them, bfloat16, held as float32 values.
float_params = jax.tree.map(lambda value: jnp.array(jnp.array(value, dtype=jnp.bfloat16), dtype=jnp.float32), stored["params"])
float_config = TransformerConfig(**{**stored["config"], "dtype": "float32"})
floating = SimpleAttentionNetwork(float_config)


def write():
    json.dump({
        "schema": "needle_upstream_calls/v1",
        "source": "needle's upstream generate_batch over overgo's bfcl-v3/simple function-call cases; internal/seq2seq/testdata/needle_upstream_calls.py",
        "jax_version": jax.__version__,
        "generated_tokens": GENERATED,
        "cases": cases,
    }, open(FIXTURE, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)


started = time.time()
for at in range(0, len(cases), BATCH):
    batch = cases[at:at + BATCH]
    queries = [case["query"] for case in batch]
    tools = [case["tools"] for case in batch]
    limit = GENERATED + 1  # the decoder's buffer opens with its start token
    plain_f32 = generate_batch(floating, float_params, tokenizer, queries, tools, max_gen_len=limit, normalize=False, constrained=False)
    named_f32 = generate_batch(floating, float_params, tokenizer, queries, tools, max_gen_len=limit, normalize=True, constrained=False)
    if not all("named" in case and "served" in case for case in batch):
        named = generate_batch(native, native_params, tokenizer, queries, tools, max_gen_len=limit, normalize=True, constrained=False)
        served = generate_batch(native, native_params, tokenizer, queries, tools, max_gen_len=limit)
        for index, case in enumerate(batch):
            case["named"], case["served"] = named[index], served[index]
    for index, case in enumerate(batch):
        normalized, _ = normalize_tools(case["tools"])
        case["encoder_ids"] = [int(token) for token in _build_encoder_input(tokenizer, case["query"], case["tools"])]
        case["named_tools"] = normalized
        case["named_encoder_ids"] = [int(token) for token in _build_encoder_input(tokenizer, case["query"], normalized)]
        case["plain_f32"], case["named_f32"] = plain_f32[index], named_f32[index]
    print(f"{at + len(batch)}/{len(cases)} in {time.time() - started:.0f}s", flush=True)
    write()
