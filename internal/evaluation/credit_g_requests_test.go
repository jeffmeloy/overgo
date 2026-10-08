package evaluation

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"testing"
)

// creditGFixture is a table of the credit-g shape: an identifying text
// column whose sorted index is the row itself, a text column of three values,
// an integer column, and the class of each row, three rows in ten bad, with
// rows enough of each class for every request.
func creditGFixture() []TableColumn {
	identity := TableColumn{Name: "identity"}
	purpose := TableColumn{Name: "purpose"}
	amount := TableColumn{Name: "amount", Integer: true}
	class := TableColumn{Name: creditGLabel}
	for row := range 1000 {
		identity.Cells = append(identity.Cells, fmt.Sprintf("row-%04d", row))
		purpose.Cells = append(purpose.Cells, []string{"car", "education", "repairs"}[row%3])
		amount.Cells = append(amount.Cells, fmt.Sprint(250+(row*7919)%9001))
		label := creditGGood
		if row%10 < 3 {
			label = creditGBad
		}
		class.Cells = append(class.Cells, label)
	}
	return []TableColumn{identity, purpose, amount, class}
}

// TestCreditGRequestsAreDisjointAndStandardized holds the requests to their
// construction: sixteen requests, each asking about thirty rows, nine bad and
// twenty-one good customers; no row asked about twice; the context of every
// request the same rows, all those no request asks about, at the table's own
// share of each class, so that no query row's class is shown anywhere; a text
// column its value's index in the column's sorted values; an integer column
// standardized by the context rows; the same table building the same bytes;
// and a table without the classes, with a third class, with too few rows of
// one, or with no row left for the context refused.
func TestCreditGRequestsAreDisjointAndStandardized(t *testing.T) {
	t.Parallel()
	table := creditGFixture()
	built, err := BuildCreditGRequests(table)
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := BuildCreditGRequests(table); len(built) != creditGRequests || !slices.EqualFunc(built, again, slices.Equal[[]byte]) {
		t.Fatalf("the same table built %d requests, and not the same bytes twice", len(built))
	}
	asked, query := map[float64]int{}, creditGBadRows+creditGGoodRows
	context := len(table[0].Cells) - creditGRequests*query
	var shown []float64
	for index, document := range built {
		var request struct {
			Task        string    `json:"task"`
			X           []float64 `json:"x"`
			Y           []float64 `json:"y"`
			Rows        int       `json:"rows"`
			Cols        int       `json:"cols"`
			TrainRows   int       `json:"train_rows"`
			Categorical []int     `json:"categorical_columns"`
		}
		if err := json.Unmarshal(document, &request); err != nil {
			t.Fatal(err)
		}
		rows := context + query
		if request.Task != "classification" || request.Rows != rows || request.Cols != 3 || request.TrainRows != context ||
			len(request.X) != rows*request.Cols || len(request.Y) != rows || !slices.Equal(request.Categorical, []int{0, 1}) {
			t.Fatalf("request %d = %s rows %d cols %d train %d categorical %v", index, request.Task, request.Rows, request.Cols, request.TrainRows, request.Categorical)
		}
		var sum, squares, contextGood, queryGood float64
		var identities []float64
		for row := range rows {
			identity, purpose, amount := request.X[row*3], request.X[row*3+1], request.X[row*3+2]
			source := int(identity)
			if purpose != float64(source%3) || request.Y[row] != creditGClassOf(table, source) {
				t.Fatalf("request %d row %d: purpose %v class %v do not belong to source row %d", index, row, purpose, request.Y[row], source)
			}
			if row >= context {
				if asked[identity]++; asked[identity] > 1 {
					t.Fatalf("row %v is asked about twice", identity)
				}
				queryGood += request.Y[row]
				continue
			}
			identities = append(identities, identity)
			sum, squares, contextGood = sum+amount, squares+amount*amount, contextGood+request.Y[row]
		}
		if shown == nil {
			shown = identities
		}
		// The table's seven good rows in ten, in the query and in the context.
		if !slices.Equal(identities, shown) || queryGood != creditGGoodRows || contextGood*float64(query) != creditGGoodRows*float64(context) {
			t.Fatalf("request %d shows %d context rows with %v good and asks about %v good, want the same context as the first request and the table's share of each class", index, len(identities), contextGood, queryGood)
		}
		// Standardized by the context rows: their mean is zero and their
		// spread one, to the six decimals a value keeps.
		mean := sum / float64(context)
		if spread := math.Sqrt(squares/float64(context) - mean*mean); math.Abs(mean) > 1e-5 || math.Abs(spread-1) > 1e-5 {
			t.Fatalf("request %d context amounts have mean %v and spread %v, want 0 and 1", index, mean, spread)
		}
	}
	if len(asked) != creditGRequests*query {
		t.Fatalf("the requests ask about %d distinct rows", len(asked))
	}
	for _, identity := range shown {
		if asked[identity] != 0 {
			t.Fatalf("row %v is shown as context and asked about", identity)
		}
	}
	for name, change := range map[string]func([]TableColumn) []TableColumn{
		"no class column": func(columns []TableColumn) []TableColumn { return columns[:3] },
		"a third class": func(columns []TableColumn) []TableColumn {
			columns[3].Cells = slices.Clone(columns[3].Cells)
			columns[3].Cells[5] = "unknown"
			return columns
		},
		"too few rows": func(columns []TableColumn) []TableColumn {
			for index := range columns {
				columns[index].Cells = columns[index].Cells[:400]
			}
			return columns
		},
		"no row left for the context": func(columns []TableColumn) []TableColumn {
			for index := range columns {
				columns[index].Cells = columns[index].Cells[:creditGRequests*(creditGBadRows+creditGGoodRows)]
			}
			return columns
		},
		"an integer that is not one": func(columns []TableColumn) []TableColumn {
			columns[2].Cells = slices.Clone(columns[2].Cells)
			columns[2].Cells[0] = "many"
			return columns
		},
	} {
		if _, err := BuildCreditGRequests(change(creditGFixture())); err == nil {
			t.Errorf("a table with %s built requests", name)
		}
	}
}

// creditGClassOf is the class number the fixture gives a source row.
func creditGClassOf(table []TableColumn, row int) float64 {
	if table[3].Cells[row] == creditGGood {
		return 1
	}
	return 0
}

// TestSeededPythonRandomDrawsWhatCPythonDraws holds the seeded generator to
// CPython 3.13's random.Random(17): its first three 32-bit outputs and its
// shuffle of the ten integers from zero, recorded from that interpreter, and
// a float written as Python's repr writes it.
func TestSeededPythonRandomDrawsWhatCPythonDraws(t *testing.T) {
	t.Parallel()
	random := seededPythonRandom(creditGSeed)
	if got := []uint32{random.next(), random.next(), random.next()}; !slices.Equal(got, []uint32{2241903809, 1778837931, 3464710496}) {
		t.Fatalf("first outputs = %v", got)
	}
	values := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	seededPythonRandom(creditGSeed).shuffle(values)
	if !slices.Equal(values, []int{7, 3, 0, 5, 1, 9, 2, 4, 6, 8}) {
		t.Fatalf("shuffle = %v", values)
	}
	for value, want := range map[float64]string{
		3: "3.0", 2.562648: "2.562648", 0: "0.0", math.Copysign(0, -1): "-0.0", 0.0001: "0.0001",
		0.00001234: "1.234e-05", 1e-06: "1e-06", 123456.789012: "123456.789012", 1e16: "1e+16",
	} {
		if got := pythonFloat(value); got != want {
			t.Errorf("pythonFloat(%v) = %q, want %q", value, got, want)
		}
	}
}
