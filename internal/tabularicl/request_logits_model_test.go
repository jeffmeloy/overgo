//go:build modeltest

package tabularicl

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"overgo/internal/dataroot"
	"overgo/internal/testutil"
)

// requestLogitsScript writes the fixture this test reads, from the model's
// own code. It is kept where Python is kept, on the machine that ran it and
// not in git: the fixture is the data.
const requestLogitsScript = "scripts/python_golden_references/tabfm_request_logits.py"

const (
	// requestDirectory holds the held-out classification requests, beneath the
	// datasets root, that the fixture's outputs are of.
	requestDirectory = "credit_g_requests_context"
	// answeredEvery is how many requests apart the requests answered through
	// the model are: every request's file is held to the fixture's digest of
	// it, and one in six is answered, at half a minute a request on the host.
	answeredEvery = 6
	// outputTolerance is how far an output for a class may be from the model's
	// own code's: both compute in float32 through 24 blocks over 550 rows, and
	// the largest difference measured is under a tenth of this.
	outputTolerance = 1e-3
)

// TestRequestLogitsMatchUpstream holds the path a held-out table request
// takes, from the request the objective reads to the class the verdict
// counts, to the model's own code on the requests the model is scored on. A
// request is asked as the verdict asks it: its query rows' classes cleared,
// the columns it calls categorical marked, one forward pass. For one request
// in six, every query row's output for every class is that code's to within a
// tolerance, and the class answered, the largest output, is the same wherever
// that code's two largest outputs are further apart than twice the tolerance.
// The earlier comparison of this forward pass was with an earlier
// implementation on tables with no categorical column; the held-out table is
// mostly categorical.
func TestRequestLogitsMatchUpstream(t *testing.T) {
	raw, err := os.ReadFile(testutil.FixturePath(t, "tabfm_request_logits.json"))
	if err != nil {
		t.Fatalf("the upstream outputs are absent (%v); write them with %s", err, requestLogitsScript)
	}
	var fixture struct {
		Classes int `json:"classes"`
		Cases   []struct {
			Name        string      `json:"name"`
			SHA256      string      `json:"sha256"`
			ContextRows int         `json:"context_rows"`
			Answers     []int       `json:"answers"`
			Outputs     [][]float64 `json:"outputs"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	directory := filepath.Join(roots.Datasets, requestDirectory)
	if _, err := os.Stat(directory); err != nil {
		t.Skipf("UNAVAILABLE: held-out requests absent at %s; parity NOT verified", directory)
	}
	model, err := LoadTask(testutil.ModelArtifactDir(t, "tabfm-1.0.0-pytorch"), TaskClassification)
	if err != nil {
		t.Skipf("UNAVAILABLE: table model unavailable: %v; parity NOT verified", err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("the fixture holds no request")
	}
	var answered, rows, agreed, close int
	var worst float64
	for index, reference := range fixture.Cases {
		document, err := os.ReadFile(filepath.Join(directory, reference.Name))
		if err != nil {
			t.Fatal(err)
		}
		if sum := sha256.Sum256(document); hex.EncodeToString(sum[:]) != reference.SHA256 {
			t.Fatalf("%s is not the request the fixture's outputs are of; write the fixture again with %s", reference.Name, requestLogitsScript)
		}
		if index%answeredEvery != 0 {
			continue
		}
		var request Request
		if err := json.Unmarshal(document, &request); err != nil {
			t.Fatal(err)
		}
		request.Y = slices.Clone(request.Y)
		clear(request.Y[request.TrainRows:])
		outputs, width, err := model.Predict(request)
		if err != nil {
			t.Fatal(err)
		}
		if request.TrainRows != reference.ContextRows || width != fixture.Classes || len(reference.Outputs) != request.Rows-request.TrainRows {
			t.Fatalf("%s: %d context rows and %d outputs a row, and the fixture holds %d and %d over %d query rows", reference.Name, request.TrainRows, width, reference.ContextRows, fixture.Classes, len(reference.Outputs))
		}
		for row, wanted := range reference.Outputs {
			scores := outputs[(request.TrainRows+row)*width : (request.TrainRows+row+1)*width]
			for class, value := range wanted {
				apart := math.Abs(float64(scores[class]) - value)
				if worst = max(worst, apart); apart > outputTolerance {
					t.Fatalf("%s query row %d class %d: the output is %.5f and the model's own code gives %.5f", reference.Name, row, class, scores[class], value)
				}
			}
			ordered := slices.Sorted(slices.Values(wanted))
			decided := ordered[len(ordered)-1]-ordered[len(ordered)-2] > outputTolerance+outputTolerance
			same := slices.Index(scores, slices.Max(scores)) == reference.Answers[row]
			if decided && !same {
				t.Fatalf("%s query row %d: class %d is answered and the model's own code answers %d", reference.Name, row, slices.Index(scores, slices.Max(scores)), reference.Answers[row])
			}
			rows++
			if same {
				agreed++
			}
			if !decided {
				close++
			}
		}
		answered++
	}
	t.Logf("%d requests are the fixture's; of %d answered, every output is within %.2g of the model's own code's, and the class answered is the same on %d of %d query rows, %d of them too close for the tolerance to decide",
		len(fixture.Cases), answered, worst, agreed, rows, close)
}
