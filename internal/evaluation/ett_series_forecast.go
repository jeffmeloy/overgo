package evaluation

import (
	"encoding/csv"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
)

const (
	ettSource = "etth1/oil-temperature"
	ettSchema = "overgo/etth1-oil-temperature-windows/v1"
	// ettColumn is the column of the ETT files the suite forecasts: the oil
	// temperature of the transformer.
	ettColumn = "OT"
	// minimumETTRows is a header and a reading.
	minimumETTRows = 2
	// float32Bits is the precision a reading is read at, a model reading float32.
	float32Bits = 32
)

// SeriesForecastSpec names what a series-forecast suite is built from: the
// ETT file of one transformer's hourly readings, and how many readings a model
// is shown and how many it forecasts.
type SeriesForecastSpec struct {
	CSV     string
	Context int
	Horizon int
}

// BuildETTSeriesForecast builds the series-forecast suite of an ETT file: its
// oil-temperature column cut into consecutive windows that do not overlap,
// each showing a model Context readings and asking for the Horizon after
// them. The windows span the whole file, and the last, which would run past
// its end, is left out.
func BuildETTSeriesForecast(spec SeriesForecastSpec) (SeriesForecastSuite, error) {
	if spec.Context <= 0 || spec.Horizon <= 0 {
		return SeriesForecastSuite{}, errors.New("evaluation: a series-forecast window shows and forecasts at least one reading")
	}
	file, err := os.Open(spec.CSV)
	if err != nil {
		return SeriesForecastSuite{}, err
	}
	defer file.Close()
	rows, err := csv.NewReader(file).ReadAll()
	if err != nil || len(rows) < minimumETTRows {
		return SeriesForecastSuite{}, fmt.Errorf("evaluation: read %s: %d rows, %v", spec.CSV, len(rows), err)
	}
	column := slices.Index(rows[0], ettColumn)
	if column < 0 {
		return SeriesForecastSuite{}, fmt.Errorf("evaluation: %s has no %s column", spec.CSV, ettColumn)
	}
	values := make([]float32, 0, len(rows)-1)
	for index, row := range rows[1:] {
		value, err := strconv.ParseFloat(row[column], float32Bits)
		if err != nil {
			return SeriesForecastSuite{}, fmt.Errorf("evaluation: %s row %d: %w", spec.CSV, index+1, err)
		}
		values = append(values, float32(value))
	}
	suite := SeriesForecastSuite{Kind: seriesForecastKind, Schema: ettSchema, Source: ettSource}
	window := spec.Context + spec.Horizon
	for start := 0; start+window <= len(values); start += window {
		suite.Cases = append(suite.Cases, SeriesForecastCase{
			Name:    fmt.Sprintf("%s/%03d", ettSource, len(suite.Cases)),
			Context: slices.Clone(values[start : start+spec.Context]),
			Target:  slices.Clone(values[start+spec.Context : start+window]),
		})
	}
	if len(suite.Cases) == 0 {
		return SeriesForecastSuite{}, fmt.Errorf("evaluation: %s holds %d readings, fewer than one window of %d", spec.CSV, len(values), window)
	}
	return suite, nil
}
