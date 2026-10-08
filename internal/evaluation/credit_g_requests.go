package evaluation

import (
	"fmt"
	"math/big"
	"slices"
	"strconv"
	"strings"
)

const (
	// creditGLabel is the column the requests' classes are read from, and
	// creditGBad and creditGGood its two classes, in the sorted order that
	// numbers them 0 and 1.
	creditGLabel = "good_or_bad_customer"
	creditGBad   = "bad"
	creditGGood  = "good"
	// creditGSeed fixes which rows each request takes and their order.
	creditGSeed = 17
	// creditGRequests requests each ask about creditGBadRows bad and
	// creditGGoodRows good customers, the table's own share of each class, and
	// no row is asked about twice. Every row that is asked about by no request
	// is context for all of them: 520 of the table's thousand, at the same
	// share. The model learns a table from its context rows, and a request
	// that showed it 30 read below the rate of always answering good, in the
	// model's own code as in overgo's; from a hundred rows up both read above
	// it.
	creditGRequests = 16
	creditGBadRows  = 9
	creditGGoodRows = 21
	// creditGSpreadFloor keeps a column every context row agrees on from
	// dividing by zero, and creditGDecimals is how many decimals a
	// standardized value keeps.
	creditGSpreadFloor = 1e-6
	creditGDecimals    = 6
	// creditGRootBits is the precision a standard deviation's square root is
	// taken at before it is rounded to a float once.
	creditGRootBits = 256
	// decimalBase is the base a table's integer cells are written in, and
	// cellBits the width they and the standardized values are read at.
	decimalBase = 10
	cellBits    = 64
	// pythonFixedFrom and pythonFixedBelow bound the decimal exponents Python
	// writes without an exponent: from a ten-thousandth up to, and not
	// including, ten to the sixteenth.
	pythonFixedFrom  = -4
	pythonFixedBelow = 16
)

// TableColumn is one column of a table a tabular evaluation set is built
// from: its name, whether it holds integers, and its cells as text, one a row.
type TableColumn struct {
	Name    string
	Integer bool
	Cells   []string
}

// BuildCreditGRequests builds the held-out classification requests of the
// credit-g table, each as the JSON document the tabular objective reads: a
// request holds context rows with their classes and query rows whose classes
// the model answers. No row is a query row of two requests, and no query row
// is a context row of any, so a split of the requests keeps every held-out
// row's class off the training ones; the context rows, every row no request
// asks about, are the same in all of them, as the rows of a table a model has
// been shown are the same for every new row it is asked about. A text column
// is its value's index in the column's sorted values; an integer column is
// standardized by the context rows. Rows and order come from a seeded
// generator, so the same table builds the same bytes.
func BuildCreditGRequests(columns []TableColumn) ([][]byte, error) {
	label := slices.IndexFunc(columns, func(column TableColumn) bool { return column.Name == creditGLabel })
	if label < 0 || columns[label].Integer {
		return nil, fmt.Errorf("evaluation: the table has no text column %q to read classes from", creditGLabel)
	}
	classes := columns[label].Cells
	features := slices.Delete(slices.Clone(columns), label, label+1)
	codes := make([][]float64, len(features))
	integers := make([][]int64, len(features))
	var categorical []string
	for index, column := range features {
		if len(column.Cells) != len(classes) {
			return nil, fmt.Errorf("evaluation: column %q holds %d rows, the classes %d", column.Name, len(column.Cells), len(classes))
		}
		if !column.Integer {
			categorical = append(categorical, strconv.Itoa(index))
			vocabulary := slices.Compact(slices.Sorted(slices.Values(column.Cells)))
			for _, cell := range column.Cells {
				code, _ := slices.BinarySearch(vocabulary, cell)
				codes[index] = append(codes[index], float64(code))
			}
			continue
		}
		for _, cell := range column.Cells {
			value, err := strconv.ParseInt(cell, decimalBase, cellBits)
			if err != nil {
				return nil, fmt.Errorf("evaluation: column %q: %w", column.Name, err)
			}
			integers[index] = append(integers[index], value)
		}
	}
	var bad, good []int
	for row, class := range classes {
		switch class {
		case creditGBad:
			bad = append(bad, row)
		case creditGGood:
			good = append(good, row)
		default:
			return nil, fmt.Errorf("evaluation: row %d has class %q, neither %s nor %s", row, class, creditGBad, creditGGood)
		}
	}
	if len(bad) < creditGRequests*creditGBadRows || len(good) < creditGRequests*creditGGoodRows {
		return nil, fmt.Errorf("evaluation: %d %s and %d %s rows are too few for %d requests of %d and %d", len(bad), creditGBad, len(good), creditGGood, creditGRequests, creditGBadRows, creditGGoodRows)
	}
	random := seededPythonRandom(creditGSeed)
	random.shuffle(bad)
	random.shuffle(good)
	// The rows after every request's query rows are asked about by none.
	context := slices.Concat(bad[creditGRequests*creditGBadRows:], good[creditGRequests*creditGGoodRows:])
	if len(context) == 0 {
		return nil, fmt.Errorf("evaluation: the table's %d rows leave none for the requests' context", len(classes))
	}
	random.shuffle(context)
	means, spreads := make([]float64, len(features)), make([]float64, len(features))
	for index, column := range integers {
		if column != nil {
			means[index], spreads[index] = meanAndSpread(column, context)
		}
	}
	var requests [][]byte
	for request := range creditGRequests {
		query := slices.Concat(bad[request*creditGBadRows:(request+1)*creditGBadRows], good[request*creditGGoodRows:(request+1)*creditGGoodRows])
		random.shuffle(query)
		order := slices.Concat(context, query)
		var x, y []string
		for _, row := range order {
			for index := range features {
				var value float64
				if integers[index] != nil {
					standardized := (float64(integers[index][row]) - means[index]) / max(spreads[index], creditGSpreadFloor)
					value, _ = strconv.ParseFloat(strconv.FormatFloat(standardized, 'f', creditGDecimals, cellBits), cellBits)
				} else {
					value = codes[index][row]
				}
				x = append(x, pythonFloat(value))
			}
			var class float64
			if classes[row] == creditGGood {
				class = 1
			}
			y = append(y, pythonFloat(class))
		}
		requests = append(requests, fmt.Appendf(nil, `{"task":"classification","x":[%s],"y":[%s],"rows":%d,"cols":%d,"train_rows":%d,"categorical_columns":[%s]}`,
			strings.Join(x, ","), strings.Join(y, ","), len(order), len(features), len(context), strings.Join(categorical, ",")))
	}
	return requests, nil
}

// meanAndSpread are the mean and the population standard deviation of a
// column's values at the given rows, each rounded to a float once: the sums
// are exact, as the integers allow, and the root is taken at a precision far
// above a float's. A mean summed in floats can differ in its last bit, and a
// standardized value rounded to six decimals can then differ in its last
// digit.
func meanAndSpread(column []int64, rows []int) (float64, float64) {
	if len(rows) == 0 {
		return 0, 0
	}
	sum, squares := new(big.Int), new(big.Int)
	for _, row := range rows {
		value := big.NewInt(column[row])
		sum.Add(sum, value)
		squares.Add(squares, new(big.Int).Mul(value, value))
	}
	count := big.NewInt(int64(len(rows)))
	mean, _ := new(big.Rat).SetFrac(sum, count).Float64()
	// The variance is (n*sum of squares - square of the sum) / n squared.
	numerator := new(big.Int).Sub(new(big.Int).Mul(count, squares), new(big.Int).Mul(sum, sum))
	precise := func(value *big.Int) *big.Float { return new(big.Float).SetPrec(creditGRootBits).SetInt(value) }
	variance := new(big.Float).SetPrec(creditGRootBits).Quo(precise(numerator), precise(new(big.Int).Mul(count, count)))
	spread, _ := new(big.Float).SetPrec(creditGRootBits).Sqrt(variance).Float64()
	return mean, spread
}

// pythonFloat writes a float as Python's repr does, which is how the
// published requests hold their numbers: the shortest digits that read back
// as the same float, with a decimal point always, and an exponent only below
// a ten-thousandth or from ten to the sixteenth up.
func pythonFloat(value float64) string {
	mantissa, exponent, _ := strings.Cut(strconv.FormatFloat(value, 'e', -1, cellBits), "e")
	if power, _ := strconv.Atoi(exponent); value != 0 && (power < pythonFixedFrom || power >= pythonFixedBelow) {
		return fmt.Sprintf("%se%+03d", mantissa, power)
	}
	text := strconv.FormatFloat(value, 'f', -1, cellBits)
	if !strings.Contains(text, ".") {
		text += ".0"
	}
	return text
}
