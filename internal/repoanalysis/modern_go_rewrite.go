package repoanalysis

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/gosource"
)

type modernGoSourceEdit struct {
	start, end  int
	replacement string
}

// ModernGoManualRewrite reports deterministic exact-idiom replacements.
type ModernGoManualRewrite struct {
	Fallbacks   int
	Tickers     int
	SliceClones int
	TypedSorts  int
	Files       []string
}

// RewriteModernGoManualIdioms replaces only eagerly safe fallback values,
// channel-only tickers, exact append clones, and typed sort wrappers.
func RewriteModernGoManualIdioms(root string) (ModernGoManualRewrite, error) {
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return ModernGoManualRewrite{}, err
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return ModernGoManualRewrite{}, err
	}
	parsed, err := parseModernGoFiles(snapshot, selection)
	if err != nil {
		return ModernGoManualRewrite{}, err
	}
	var result ModernGoManualRewrite
	for _, file := range parsed {
		if file.generated || !file.typed || file.info == nil {
			continue
		}
		imports := importBases(file.syntax)
		cmpQualifier, cmpImported := modernGoImportQualifier(file.syntax, "cmp")
		slicesQualifier, slicesImported := modernGoImportQualifier(file.syntax, "slices")
		var edits []modernGoSourceEdit
		fallbacks, tickers, clones, sorts := 0, 0, 0, 0
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			switch current := node.(type) {
			case *ast.IfStmt:
				left, right, ok := modernGoManualFallback(current, file.info)
				if !ok || cmpQualifier == "" || modernGoCommentsWithin(file.syntax, current.Pos(), current.End()) {
					return true
				}
				leftText := modernGoNodeText(file, left)
				rightText := modernGoNodeText(file, right)
				edits = append(edits, modernGoSourceEdit{
					start:       file.fileSet.Position(current.Pos()).Offset,
					end:         file.fileSet.Position(current.End()).Offset,
					replacement: leftText + " = " + cmpQualifier + ".Or(" + leftText + ", " + rightText + ")",
				})
				fallbacks++
				return false
			case *ast.CallExpr:
				key := modernGoCallKey(current, file.info, imports)
				switch {
				case key == "append" && len(current.Args) == 2 && current.Ellipsis.IsValid() && isNilSlice(current.Args[0]) && slicesQualifier != "":
					edits = append(edits, modernGoSourceEdit{
						start:       file.fileSet.Position(current.Pos()).Offset,
						end:         file.fileSet.Position(current.End()).Offset,
						replacement: slicesQualifier + ".Clone(" + modernGoNodeText(file, current.Args[1]) + ")",
					})
					clones++
					return false
				case (key == "sort.Strings" || key == "sort.Ints" || key == "sort.Float64s") && len(current.Args) == 1 && slicesQualifier != "":
					edits = append(edits, modernGoSourceEdit{
						start:       file.fileSet.Position(current.Fun.Pos()).Offset,
						end:         file.fileSet.Position(current.Fun.End()).Offset,
						replacement: slicesQualifier + ".Sort",
					})
					sorts++
				}
			}
			return true
		})
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			block, ok := node.(*ast.BlockStmt)
			if !ok {
				return true
			}
			for index := 0; index+1 < len(block.List); index++ {
				tickerEdits, matched := modernGoManualTicker(block, index, file, imports)
				if !matched {
					continue
				}
				edits = append(edits, tickerEdits...)
				tickers++
			}
			return true
		})
		if len(edits) == 0 {
			continue
		}
		slices.SortFunc(edits, func(left, right modernGoSourceEdit) int { return left.start - right.start })
		name := filepath.Join(root, filepath.FromSlash(file.source.Path))
		current, err := os.ReadFile(name)
		if err != nil {
			return result, err
		}
		if !bytes.Equal(current, file.source.blob.data) {
			return result, fmt.Errorf("modern-Go manual idiom source changed during rewrite: %s", file.source.Path)
		}
		var rewritten bytes.Buffer
		prior := 0
		for _, edit := range edits {
			if edit.start < prior || edit.end < edit.start || edit.end > len(current) {
				return result, fmt.Errorf("modern-Go manual idiom edit overlaps in %s", file.source.Path)
			}
			rewritten.Write(current[prior:edit.start])
			rewritten.WriteString(edit.replacement)
			prior = edit.end
		}
		rewritten.Write(current[prior:])
		updated := rewritten.Bytes()
		if fallbacks != 0 && !cmpImported {
			updated, err = ensureNamedImport(name, updated, "cmp")
			if err != nil {
				return result, err
			}
		}
		if clones+sorts != 0 && !slicesImported {
			updated, err = ensureNamedImport(name, updated, "slices")
			if err != nil {
				return result, err
			}
		}
		if sorts != 0 {
			updated, err = removeUnusedNamedImport(name, updated, "sort")
			if err != nil {
				return result, err
			}
		}
		formatted, err := format.Source(updated)
		if err != nil {
			return result, fmt.Errorf("format modern-Go manual idiom rewrite "+file.source.Path+": %w", err)
		}
		info, err := os.Stat(name)
		if err != nil {
			return result, err
		}
		if err := os.WriteFile(name, formatted, info.Mode()); err != nil {
			return result, err
		}
		result.Fallbacks += fallbacks
		result.Tickers += tickers
		result.SliceClones += clones
		result.TypedSorts += sorts
		result.Files = append(result.Files, file.source.Path)
	}
	slices.Sort(result.Files)
	return result, nil
}

func modernGoNodeText(file modernGoParsedFile, node ast.Node) string {
	start := file.fileSet.Position(node.Pos()).Offset
	end := file.fileSet.Position(node.End()).Offset
	return string(file.source.blob.data[start:end])
}

func modernGoManualFallback(statement *ast.IfStmt, info *types.Info) (*ast.Ident, ast.Expr, bool) {
	if !modernGoFallbackIf(statement) {
		return nil, nil, false
	}
	assignment := statement.Body.List[0].(*ast.AssignStmt)
	left, ok := assignment.Lhs[0].(*ast.Ident)
	if !ok || !types.Comparable(info.TypeOf(left)) || !modernGoEagerSafeFallback(assignment.Rhs[0]) {
		return nil, nil, false
	}
	return left, assignment.Rhs[0], true
}

func modernGoEagerSafeFallback(expression ast.Expr) bool {
	switch value := expression.(type) {
	case *ast.BasicLit, *ast.Ident:
		return true
	case *ast.ParenExpr:
		return modernGoEagerSafeFallback(value.X)
	case *ast.UnaryExpr:
		return value.Op != token.MUL && modernGoEagerSafeFallback(value.X)
	default:
		return false
	}
}

func modernGoManualTicker(block *ast.BlockStmt, index int, file modernGoParsedFile, imports map[string]string) ([]modernGoSourceEdit, bool) {
	assignment, ok := block.List[index].(*ast.AssignStmt)
	if !ok || assignment.Tok != token.DEFINE || len(assignment.Lhs) != 1 || len(assignment.Rhs) != 1 {
		return nil, false
	}
	name, nameOK := assignment.Lhs[0].(*ast.Ident)
	call, callOK := assignment.Rhs[0].(*ast.CallExpr)
	if !nameOK || !callOK || len(call.Args) != 1 || modernGoCallKey(call, file.info, imports) != "time.NewTicker" {
		return nil, false
	}
	deferred, ok := block.List[index+1].(*ast.DeferStmt)
	if !ok || deferred.Call == nil || len(deferred.Call.Args) != 0 {
		return nil, false
	}
	stop, ok := deferred.Call.Fun.(*ast.SelectorExpr)
	stopName, stopNameOK := stop.X.(*ast.Ident)
	if !ok || !stopNameOK || stopName.Name != name.Name || stop.Sel.Name != "Stop" ||
		modernGoCommentsWithin(file.syntax, assignment.Pos(), deferred.End()) {
		return nil, false
	}
	object := file.info.ObjectOf(name)
	if object == nil {
		return nil, false
	}
	var channelSelectors []*ast.SelectorExpr
	valid := true
	for _, statement := range block.List[index+2:] {
		ast.Inspect(statement, func(node ast.Node) bool {
			identifier, ok := node.(*ast.Ident)
			if !ok || file.info.ObjectOf(identifier) != object {
				return valid
			}
			selector, selectorOK := modernGoParentSelector(statement, identifier)
			if !selectorOK || selector.Sel.Name != "C" {
				valid = false
				return false
			}
			channelSelectors = append(channelSelectors, selector)
			return true
		})
		if !valid {
			return nil, false
		}
	}
	if len(channelSelectors) == 0 {
		return nil, false
	}
	edits := []modernGoSourceEdit{{
		start:       file.fileSet.Position(assignment.Pos()).Offset,
		end:         file.fileSet.Position(deferred.End()).Offset,
		replacement: name.Name + " := time.Tick(" + modernGoNodeText(file, call.Args[0]) + ")",
	}}
	for _, selector := range channelSelectors {
		edits = append(edits, modernGoSourceEdit{
			start:       file.fileSet.Position(selector.Pos()).Offset,
			end:         file.fileSet.Position(selector.End()).Offset,
			replacement: name.Name,
		})
	}
	return edits, true
}

func modernGoParentSelector(root ast.Node, identifier *ast.Ident) (*ast.SelectorExpr, bool) {
	parents := modernGoParents(root)
	selector, ok := parents[identifier].(*ast.SelectorExpr)
	return selector, ok && selector.X == identifier
}

// RewriteModernGoExtrema replaces type-proven ordered extrema with the Go 1.21
// built-ins. Floating-point extrema deliberately adopt the built-in policy:
// NaNs propagate and ties select -0 for min and +0 for max. Canonical slice
// reductions retain their existing empty-input panic because they already read
// element zero before looping.
func RewriteModernGoExtrema(root string) (int, []string, error) {
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return 0, nil, err
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return 0, nil, err
	}
	parsed, err := parseModernGoFiles(snapshot, selection)
	if err != nil {
		return 0, nil, err
	}
	total := 0
	var changed []string
	for _, file := range parsed {
		if file.generated {
			continue
		}
		imports := importBases(file.syntax)
		parents := modernGoParents(file.syntax)
		slicesQualifier, slicesImportOK := modernGoImportQualifier(file.syntax, "slices")
		var edits []modernGoSourceEdit
		var occupied []modernGoSourceEdit
		needsSlices := false
		if file.typed && file.info != nil {
			ast.Inspect(file.syntax, func(node ast.Node) bool {
				block, ok := node.(*ast.BlockStmt)
				if !ok {
					return true
				}
				for index := 0; index+1 < len(block.List); index++ {
					start, end, replacement, matched := modernGoSliceExtrema(block.List[index], block.List[index+1], file.info, slicesQualifier)
					if !matched || modernGoCommentsWithin(file.syntax, start, end) {
						continue
					}
					edit := modernGoSourceEdit{
						start: file.fileSet.Position(start).Offset, end: file.fileSet.Position(end).Offset,
						replacement: replacement,
					}
					edits = append(edits, edit)
					occupied = append(occupied, edit)
					needsSlices = needsSlices || !slicesImportOK
				}
				return true
			})
		}
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			switch current := node.(type) {
			case *ast.CallExpr:
				key := modernGoCallKey(current, file.info, imports)
				if key != "math.Min" && key != "math.Max" {
					return true
				}
				selector := current.Fun.(*ast.SelectorExpr)
				edits = append(edits, modernGoSourceEdit{
					start:       file.fileSet.Position(selector.Pos()).Offset,
					end:         file.fileSet.Position(selector.End()).Offset,
					replacement: strings.ToLower(selector.Sel.Name),
				})
			case *ast.IfStmt:
				matchInfo := file.info
				if !file.typed {
					matchInfo = nil
				}
				destination, other, function, ok := modernGoExtremaAssignment(current, matchInfo)
				if !ok {
					return true
				}
				comment, commentsOK := modernGoExtremaTrailingComment(file.syntax, current)
				if !commentsOK {
					return true
				}
				start, end := file.fileSet.Position(current.Pos()).Offset, file.fileSet.Position(current.End()).Offset
				if modernGoEditOverlaps(occupied, start, end) {
					return false
				}
				replacement := destination.Name + " = " + function + "(" + destination.Name + ", " + other.Name + ")" + comment
				if parent, ok := parents[current].(*ast.IfStmt); ok && parent.Else == current {
					replacement = "{\n" + replacement + "\n}"
				}
				edits = append(edits, modernGoSourceEdit{
					start: start, end: end,
					replacement: replacement,
				})
				return false
			}
			return true
		})
		if len(edits) == 0 {
			continue
		}
		slices.SortFunc(edits, func(left, right modernGoSourceEdit) int { return left.start - right.start })
		name := filepath.Join(root, filepath.FromSlash(file.source.Path))
		current, err := os.ReadFile(name)
		if err != nil {
			return total, changed, err
		}
		if !bytes.Equal(current, file.source.blob.data) {
			return total, changed, fmt.Errorf("modern-Go extrema source changed during rewrite: %s", file.source.Path)
		}
		var rewritten bytes.Buffer
		prior := 0
		for _, edit := range edits {
			if edit.start < prior || edit.end < edit.start || edit.end > len(current) {
				return total, changed, fmt.Errorf("modern-Go extrema edit overlaps in %s", file.source.Path)
			}
			rewritten.Write(current[prior:edit.start])
			rewritten.WriteString(edit.replacement)
			prior = edit.end
		}
		rewritten.Write(current[prior:])
		updated := rewritten.Bytes()
		if needsSlices {
			updated, err = ensureNamedImport(name, updated, "slices")
			if err != nil {
				return total, changed, err
			}
		}
		updated, err = removeUnusedNamedImport(name, updated, "math")
		if err != nil {
			return total, changed, err
		}
		formatted, err := format.Source(updated)
		if err != nil {
			return total, changed, fmt.Errorf("format modern-Go extrema rewrite "+file.source.Path+": %w", err)
		}
		info, err := os.Stat(name)
		if err != nil {
			return total, changed, err
		}
		if err := os.WriteFile(name, formatted, info.Mode()); err != nil {
			return total, changed, err
		}
		total += len(edits)
		changed = append(changed, file.source.Path)
	}
	slices.Sort(changed)
	return total, changed, nil
}

// RewriteModernGoNumericIntegerRanges replaces type-proven zero-based,
// unit-step loops in numerical runtime packages with range-over-integer. The
// matcher rejects changing bounds and induction variables before this writer
// changes only the loop header.
func RewriteModernGoNumericIntegerRanges(root string) (int, []string, error) {
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return 0, nil, err
	}
	bootstrapCount, bootstrapFiles, err := rewriteUnusedNumericRangeVariables(root, snapshot)
	if err != nil {
		return 0, nil, err
	}
	if bootstrapCount != 0 {
		snapshot, err = DiscoverGo(root, "cmd", "internal")
		if err != nil {
			return 0, nil, err
		}
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return 0, nil, err
	}
	parsed, err := parseModernGoFiles(snapshot, selection)
	if err != nil {
		return 0, nil, err
	}
	total := bootstrapCount
	changed := slices.Clone(bootstrapFiles)
	for _, file := range parsed {
		if file.generated || !modernGoNumericRuntimePath(file.source.Path) {
			continue
		}
		var edits []modernGoSourceEdit
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			switch loop := node.(type) {
			case *ast.RangeStmt:
				counter, ok := loop.Key.(*ast.Ident)
				if !ok || loop.Value != nil || loop.Tok != token.DEFINE || astContainsIdent(loop.Body, counter.Name) {
					return true
				}
				boundStart := file.fileSet.Position(loop.X.Pos()).Offset
				boundEnd := file.fileSet.Position(loop.X.End()).Offset
				edits = append(edits, modernGoSourceEdit{
					start:       file.fileSet.Position(counter.Pos()).Offset,
					end:         file.fileSet.Position(loop.X.End()).Offset,
					replacement: "range " + string(file.source.blob.data[boundStart:boundEnd]),
				})
				return true
			case *ast.ForStmt:
				if !file.typed || file.info == nil || !modernGoCountingLoop(loop, file.info) {
					return true
				}
				initial := loop.Init.(*ast.AssignStmt)
				condition := loop.Cond.(*ast.BinaryExpr)
				counter := initial.Lhs[0].(*ast.Ident)
				if modernGoCommentsWithin(file.syntax, initial.Pos(), loop.Post.End()) {
					return true
				}
				boundStart := file.fileSet.Position(condition.Y.Pos()).Offset
				boundEnd := file.fileSet.Position(condition.Y.End()).Offset
				replacement := "range " + string(file.source.blob.data[boundStart:boundEnd])
				if modernGoIdentifierObjectUsed(loop.Body, counter, file.info) {
					replacement = counter.Name + " := " + replacement
				}
				edits = append(edits, modernGoSourceEdit{
					start:       file.fileSet.Position(initial.Pos()).Offset,
					end:         file.fileSet.Position(loop.Post.End()).Offset,
					replacement: replacement,
				})
				return true
			}
			return true
		})
		if len(edits) == 0 {
			continue
		}
		slices.SortFunc(edits, func(left, right modernGoSourceEdit) int { return left.start - right.start })
		name := filepath.Join(root, filepath.FromSlash(file.source.Path))
		current, err := os.ReadFile(name)
		if err != nil {
			return total, changed, err
		}
		if !bytes.Equal(current, file.source.blob.data) {
			return total, changed, fmt.Errorf("modern-Go numeric range source changed during rewrite: %s", file.source.Path)
		}
		var rewritten bytes.Buffer
		prior := 0
		for _, edit := range edits {
			if edit.start < prior || edit.end < edit.start || edit.end > len(current) {
				return total, changed, fmt.Errorf("modern-Go numeric range edit overlaps in %s", file.source.Path)
			}
			rewritten.Write(current[prior:edit.start])
			rewritten.WriteString(edit.replacement)
			prior = edit.end
		}
		rewritten.Write(current[prior:])
		formatted, err := format.Source(rewritten.Bytes())
		if err != nil {
			return total, changed, fmt.Errorf("format modern-Go numeric range rewrite "+file.source.Path+": %w", err)
		}
		info, err := os.Stat(name)
		if err != nil {
			return total, changed, err
		}
		if err := os.WriteFile(name, formatted, info.Mode()); err != nil {
			return total, changed, err
		}
		total += len(edits)
		changed = append(changed, file.source.Path)
	}
	slices.Sort(changed)
	return total, changed, nil
}

func modernGoIdentifierObjectUsed(root ast.Node, declaration *ast.Ident, info *types.Info) bool {
	object := info.ObjectOf(declaration)
	if object == nil {
		return astContainsIdent(root, declaration.Name)
	}
	used := false
	ast.Inspect(root, func(node ast.Node) bool {
		identifier, ok := node.(*ast.Ident)
		used = used || ok && info.ObjectOf(identifier) == object
		return !used
	})
	return used
}

func rewriteUnusedNumericRangeVariables(root string, snapshot SourceSnapshot) (int, []string, error) {
	total := 0
	var changed []string
	for _, source := range snapshot.Files {
		if !modernGoNumericRuntimePath(source.Path) {
			continue
		}
		generated, err := source.Generated()
		if err != nil {
			return total, changed, err
		}
		if generated {
			continue
		}
		fileSet := token.NewFileSet()
		file, err := parser.ParseFile(fileSet, source.Path, source.blob.data, parser.ParseComments)
		if err != nil {
			return total, changed, err
		}
		var edits []modernGoSourceEdit
		ast.Inspect(file, func(node ast.Node) bool {
			loop, ok := node.(*ast.RangeStmt)
			if !ok || loop.Value != nil || loop.Tok != token.DEFINE {
				return true
			}
			counter, ok := loop.Key.(*ast.Ident)
			if !ok || counter.Name == "_" || astContainsIdent(loop.Body, counter.Name) {
				return true
			}
			boundStart := fileSet.Position(loop.X.Pos()).Offset
			boundEnd := fileSet.Position(loop.X.End()).Offset
			edits = append(edits, modernGoSourceEdit{
				start:       fileSet.Position(counter.Pos()).Offset,
				end:         fileSet.Position(loop.X.End()).Offset,
				replacement: "range " + string(source.blob.data[boundStart:boundEnd]),
			})
			return true
		})
		if len(edits) == 0 {
			continue
		}
		slices.SortFunc(edits, func(left, right modernGoSourceEdit) int { return left.start - right.start })
		name := filepath.Join(root, filepath.FromSlash(source.Path))
		current, err := os.ReadFile(name)
		if err != nil {
			return total, changed, err
		}
		if !bytes.Equal(current, source.blob.data) {
			return total, changed, fmt.Errorf("modern-Go numeric range source changed during bootstrap rewrite: %s", source.Path)
		}
		var rewritten bytes.Buffer
		prior := 0
		for _, edit := range edits {
			if edit.start < prior || edit.end < edit.start || edit.end > len(current) {
				return total, changed, fmt.Errorf("modern-Go numeric range bootstrap edit overlaps in %s", source.Path)
			}
			rewritten.Write(current[prior:edit.start])
			rewritten.WriteString(edit.replacement)
			prior = edit.end
		}
		rewritten.Write(current[prior:])
		formatted, err := format.Source(rewritten.Bytes())
		if err != nil {
			return total, changed, fmt.Errorf("format modern-Go numeric range bootstrap rewrite "+source.Path+": %w", err)
		}
		info, err := os.Stat(name)
		if err != nil {
			return total, changed, err
		}
		if err := os.WriteFile(name, formatted, info.Mode()); err != nil {
			return total, changed, err
		}
		total += len(edits)
		changed = append(changed, source.Path)
	}
	slices.Sort(changed)
	return total, changed, nil
}

func modernGoSliceExtrema(initialStatement, loopStatement ast.Stmt, info *types.Info, slicesQualifier string) (token.Pos, token.Pos, string, bool) {
	initial, initialOK := initialStatement.(*ast.AssignStmt)
	loop, loopOK := loopStatement.(*ast.RangeStmt)
	if slicesQualifier == "" || !initialOK || !loopOK || len(initial.Lhs) != 1 || len(initial.Rhs) != 1 || len(loop.Body.List) != 1 ||
		(initial.Tok != token.DEFINE && initial.Tok != token.ASSIGN) {
		return token.NoPos, token.NoPos, "", false
	}
	destination, destinationOK := initial.Lhs[0].(*ast.Ident)
	indexed, indexedOK := initial.Rhs[0].(*ast.IndexExpr)
	if !destinationOK || !indexedOK {
		return token.NoPos, token.NoPos, "", false
	}
	source, sourceOK := indexed.X.(*ast.Ident)
	loopValue, valueOK := loop.Value.(*ast.Ident)
	condition, conditionOK := loop.Body.List[0].(*ast.IfStmt)
	if !sourceOK || !valueOK || loopValue.Name == "_" || !conditionOK ||
		!basicLiteral(indexed.Index, "0") || !modernGoExtremaRange(loop.X, source.Name) {
		return token.NoPos, token.NoPos, "", false
	}
	if key, ok := loop.Key.(*ast.Ident); loop.Key != nil && (!ok || key.Name != "_") {
		return token.NoPos, token.NoPos, "", false
	}
	assigned, other, function, ok := modernGoExtremaAssignment(condition, info)
	if !ok || assigned.Name != destination.Name || other.Name != loopValue.Name {
		return token.NoPos, token.NoPos, "", false
	}
	sliceType := info.TypeOf(source)
	if sliceType == nil {
		return token.NoPos, token.NoPos, "", false
	}
	switch underlying := sliceType.Underlying().(type) {
	case *types.Slice:
		if !modernGoOrderedType(underlying.Elem()) {
			return token.NoPos, token.NoPos, "", false
		}
	case *types.Array:
		if !modernGoOrderedType(underlying.Elem()) {
			return token.NoPos, token.NoPos, "", false
		}
	default:
		return token.NoPos, token.NoPos, "", false
	}
	operator := "="
	if initial.Tok == token.DEFINE {
		operator = ":="
	}
	helper := strings.ToUpper(function[:1]) + function[1:]
	return initial.Pos(), loop.End(), destination.Name + " " + operator + " " + slicesQualifier + "." + helper + "(" + source.Name + ")", true
}

func modernGoExtremaRange(expression ast.Expr, source string) bool {
	if directIdent(expression, source) {
		return true
	}
	slice, ok := expression.(*ast.SliceExpr)
	return ok && !slice.Slice3 && directIdent(slice.X, source) && basicLiteral(slice.Low, "1") && slice.High == nil
}

func modernGoImportQualifier(file *ast.File, importPath string) (string, bool) {
	for _, imported := range file.Imports {
		path, err := strconv.Unquote(imported.Path.Value)
		if err != nil || path != importPath {
			continue
		}
		if imported.Name == nil {
			return importPath, true
		}
		if imported.Name.Name == "." || imported.Name.Name == "_" {
			return "", false
		}
		return imported.Name.Name, true
	}
	return importPath, false
}

func modernGoCommentsWithin(file *ast.File, start, end token.Pos) bool {
	for _, group := range file.Comments {
		if group.Pos() >= start && group.End() <= end {
			return true
		}
	}
	return false
}

func modernGoExtremaTrailingComment(file *ast.File, statement *ast.IfStmt) (string, bool) {
	if statement == nil || statement.Body == nil || len(statement.Body.List) != 1 {
		return "", false
	}
	var matched *ast.CommentGroup
	for _, group := range file.Comments {
		if group.Pos() < statement.Pos() || group.End() > statement.End() {
			continue
		}
		if matched != nil || len(group.List) != 1 {
			return "", false
		}
		matched = group
	}
	if matched == nil {
		return "", true
	}
	assignment, ok := statement.Body.List[0].(*ast.AssignStmt)
	comment := matched.List[0]
	if !ok || len(assignment.Rhs) != 1 || comment.Pos() < assignment.Rhs[0].End() || !strings.HasPrefix(comment.Text, "//") {
		return "", false
	}
	return " " + comment.Text, true
}

func modernGoEditOverlaps(edits []modernGoSourceEdit, start, end int) bool {
	for _, edit := range edits {
		if start < edit.end && edit.start < end {
			return true
		}
	}
	return false
}

// RewriteModernGoOmitZeroEquivalent replaces scalar omitempty tags only when
// the Go zero value and encoding/json's empty value are type-proven identical.
// Structs and types with IsZero methods remain explicit contract decisions.
func RewriteModernGoOmitZeroEquivalent(root string) (int, []string, error) {
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return 0, nil, err
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return 0, nil, err
	}
	parsed, err := parseModernGoFiles(snapshot, selection)
	if err != nil {
		return 0, nil, err
	}
	total := 0
	var changed []string
	for _, file := range parsed {
		if file.generated || !file.typed || file.info == nil {
			continue
		}
		var edits []modernGoSourceEdit
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			field, ok := node.(*ast.Field)
			if !ok || field.Tag == nil || !modernGoOmitZeroWireEquivalent(field, file.info) {
				return true
			}
			tag, err := strconv.Unquote(field.Tag.Value)
			if err != nil {
				return true
			}
			replaced, ok := modernGoReplaceJSONTagOption(tag, "omitempty", "omitzero")
			if !ok {
				return true
			}
			literal := strconv.Quote(replaced)
			if !strings.ContainsRune(replaced, '`') {
				literal = "`" + replaced + "`"
			}
			edits = append(edits, modernGoSourceEdit{
				start:       file.fileSet.Position(field.Tag.Pos()).Offset,
				end:         file.fileSet.Position(field.Tag.End()).Offset,
				replacement: literal,
			})
			return true
		})
		if len(edits) == 0 {
			continue
		}
		slices.SortFunc(edits, func(left, right modernGoSourceEdit) int { return left.start - right.start })
		name := filepath.Join(root, filepath.FromSlash(file.source.Path))
		current, err := os.ReadFile(name)
		if err != nil {
			return total, changed, err
		}
		if !bytes.Equal(current, file.source.blob.data) {
			return total, changed, fmt.Errorf("modern-Go omitzero source changed during rewrite: %s", file.source.Path)
		}
		var rewritten bytes.Buffer
		prior := 0
		for _, edit := range edits {
			if edit.start < prior || edit.end < edit.start || edit.end > len(current) {
				return total, changed, fmt.Errorf("modern-Go omitzero edit overlaps in %s", file.source.Path)
			}
			rewritten.Write(current[prior:edit.start])
			rewritten.WriteString(edit.replacement)
			prior = edit.end
		}
		rewritten.Write(current[prior:])
		formatted, err := format.Source(rewritten.Bytes())
		if err != nil {
			return total, changed, fmt.Errorf("format modern-Go omitzero rewrite "+file.source.Path+": %w", err)
		}
		info, err := os.Stat(name)
		if err != nil {
			return total, changed, err
		}
		if err := os.WriteFile(name, formatted, info.Mode()); err != nil {
			return total, changed, err
		}
		total += len(edits)
		changed = append(changed, file.source.Path)
	}
	slices.Sort(changed)
	return total, changed, nil
}

func modernGoReplaceJSONTagOption(tag, oldOption, newOption string) (string, bool) {
	const key = "json:"
	for offset := 0; offset < len(tag); {
		index := strings.Index(tag[offset:], key)
		if index < 0 {
			return tag, false
		}
		valueStart := offset + index + len(key)
		quoted, err := strconv.QuotedPrefix(tag[valueStart:])
		if err != nil {
			offset = valueStart
			continue
		}
		value, err := strconv.Unquote(quoted)
		if err != nil {
			return tag, false
		}
		parts := strings.Split(value, ",")
		for index, part := range parts {
			if index == 0 || part != oldOption {
				continue
			}
			parts[index] = newOption
			replacement := strconv.Quote(strings.Join(parts, ","))
			return tag[:valueStart] + replacement + tag[valueStart+len(quoted):], true
		}
		offset = valueStart + len(quoted)
	}
	return tag, false
}

// RewriteModernGoErrorIdentityComparisons replaces direct comparisons between
// two type-proven errors with errors.Is so wrapped sentinel identity is kept.
func RewriteModernGoErrorIdentityComparisons(root string) (int, []string, error) {
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return 0, nil, err
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return 0, nil, err
	}
	parsed, err := parseModernGoFiles(snapshot, selection)
	if err != nil {
		return 0, nil, err
	}
	total := 0
	var changed []string
	for _, file := range parsed {
		if file.generated || !file.typed || file.info == nil {
			continue
		}
		var edits []modernGoSourceEdit
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			comparison, ok := node.(*ast.BinaryExpr)
			if !ok || comparison.Op != token.EQL && comparison.Op != token.NEQ ||
				!expressionIsError(comparison.X, file.info) || !expressionIsError(comparison.Y, file.info) {
				return true
			}
			start := file.fileSet.Position(comparison.Pos()).Offset
			end := file.fileSet.Position(comparison.End()).Offset
			leftStart := file.fileSet.Position(comparison.X.Pos()).Offset
			leftEnd := file.fileSet.Position(comparison.X.End()).Offset
			rightStart := file.fileSet.Position(comparison.Y.Pos()).Offset
			rightEnd := file.fileSet.Position(comparison.Y.End()).Offset
			prefix := ""
			if comparison.Op == token.NEQ {
				prefix = "!"
			}
			edits = append(edits, modernGoSourceEdit{
				start: start, end: end,
				replacement: prefix + "errors.Is(" + string(file.source.blob.data[leftStart:leftEnd]) + ", " +
					string(file.source.blob.data[rightStart:rightEnd]) + ")",
			})
			return false
		})
		if len(edits) == 0 {
			continue
		}
		slices.SortFunc(edits, func(left, right modernGoSourceEdit) int { return left.start - right.start })
		name := filepath.Join(root, filepath.FromSlash(file.source.Path))
		current, err := os.ReadFile(name)
		if err != nil {
			return total, changed, err
		}
		if !bytes.Equal(current, file.source.blob.data) {
			return total, changed, fmt.Errorf("modern-Go error identity source changed during rewrite: %s", file.source.Path)
		}
		var rewritten bytes.Buffer
		prior := 0
		for _, edit := range edits {
			if edit.start < prior || edit.end < edit.start || edit.end > len(current) {
				return total, changed, fmt.Errorf("modern-Go error identity edit overlaps in %s", file.source.Path)
			}
			rewritten.Write(current[prior:edit.start])
			rewritten.WriteString(edit.replacement)
			prior = edit.end
		}
		rewritten.Write(current[prior:])
		updated, err := ensureNamedImport(name, rewritten.Bytes(), "errors")
		if err != nil {
			return total, changed, err
		}
		info, err := os.Stat(name)
		if err != nil {
			return total, changed, err
		}
		if err := os.WriteFile(name, updated, info.Mode()); err != nil {
			return total, changed, err
		}
		total += len(edits)
		changed = append(changed, file.source.Path)
	}
	slices.Sort(changed)
	return total, changed, nil
}

// RewriteModernGoBenchmarkLoops replaces index-free ranges over testing.B.N
// with testing.B.Loop. Indexed loops are left for human review because their
// index may be semantically significant.
func RewriteModernGoBenchmarkLoops(root string) (int, []string, error) {
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return 0, nil, err
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return 0, nil, err
	}
	parsed, err := parseModernGoFiles(snapshot, selection)
	if err != nil {
		return 0, nil, err
	}
	total := 0
	var changed []string
	for _, file := range parsed {
		if !file.source.Test || file.generated {
			continue
		}
		var edits []modernGoSourceEdit
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			loop, ok := node.(*ast.RangeStmt)
			if !ok || loop.Key != nil || loop.Value != nil {
				return true
			}
			selector, ok := loop.X.(*ast.SelectorExpr)
			if !ok || !modernGoBenchmarkCounter(selector, file.info) {
				return true
			}
			receiver, ok := selector.X.(*ast.Ident)
			if !ok {
				return true
			}
			edits = append(edits, modernGoSourceEdit{
				start:       file.fileSet.Position(loop.Range).Offset,
				end:         file.fileSet.Position(loop.X.End()).Offset,
				replacement: receiver.Name + ".Loop()",
			})
			return true
		})
		if len(edits) == 0 {
			continue
		}
		slices.SortFunc(edits, func(left, right modernGoSourceEdit) int { return left.start - right.start })
		name := filepath.Join(root, filepath.FromSlash(file.source.Path))
		current, err := os.ReadFile(name)
		if err != nil {
			return total, changed, err
		}
		if !bytes.Equal(current, file.source.blob.data) {
			return total, changed, fmt.Errorf("modern-Go benchmark source changed during rewrite: %s", file.source.Path)
		}
		var rewritten bytes.Buffer
		prior := 0
		for _, edit := range edits {
			if edit.start < prior || edit.end < edit.start || edit.end > len(current) {
				return total, changed, fmt.Errorf("modern-Go benchmark edit overlaps in %s", file.source.Path)
			}
			rewritten.Write(current[prior:edit.start])
			rewritten.WriteString(edit.replacement)
			prior = edit.end
		}
		rewritten.Write(current[prior:])
		formatted, err := format.Source(rewritten.Bytes())
		if err != nil {
			return total, changed, fmt.Errorf("format modern-Go benchmark rewrite "+file.source.Path+": %w", err)
		}
		info, err := os.Stat(name)
		if err != nil {
			return total, changed, err
		}
		if err := os.WriteFile(name, formatted, info.Mode()); err != nil {
			return total, changed, err
		}
		total += len(edits)
		changed = append(changed, file.source.Path)
	}
	slices.Sort(changed)
	return total, changed, nil
}

// RewriteModernGoTestingContexts binds type-proven test and benchmark work to
// its testing owner. Helpers without a *testing.T or *testing.B in lexical
// scope are deliberately left unchanged.
func RewriteModernGoTestingContexts(root string) (int, []string, error) {
	snapshot, err := DiscoverGo(root, "cmd", "internal")
	if err != nil {
		return 0, nil, err
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/...", "./internal/...")
	if err != nil {
		return 0, nil, err
	}
	parsed, err := parseModernGoFiles(snapshot, selection)
	if err != nil {
		return 0, nil, err
	}
	total := 0
	var changed []string
	for _, file := range parsed {
		if !file.source.Test || file.generated || !file.typed || file.info == nil {
			continue
		}
		imports := importBases(file.syntax)
		parents := modernGoParents(file.syntax)
		var edits []modernGoSourceEdit
		needsContextImport := modernGoUsesContextWithoutCancel(file.syntax)
		ast.Inspect(file.syntax, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			key := modernGoCallKey(call, file.info, imports)
			owner := modernGoTestingOwnerAt(call.Pos(), file.info)
			if owner == "" {
				return true
			}
			parentCall, _ := parents[call].(*ast.CallExpr)
			if parentCall != nil && modernGoCallKey(parentCall, file.info, imports) == "context.WithoutCancel" {
				return true
			}
			independent := modernGoCleanupSensitiveContext(call, parents[call]) || modernGoContextInCleanup(call, parents)
			replacement := ""
			switch {
			case key == "context.Background" || key == "context.TODO":
				replacement = owner + ".Context()"
			case (key == "T.Context" || key == "B.Context") && independent:
				replacement = "context.WithoutCancel(" + owner + ".Context())"
				needsContextImport = true
			default:
				return true
			}
			if independent && !strings.HasPrefix(replacement, "context.WithoutCancel(") {
				replacement = "context.WithoutCancel(" + replacement + ")"
				needsContextImport = true
			}
			edits = append(edits, modernGoSourceEdit{
				start:       file.fileSet.Position(call.Pos()).Offset,
				end:         file.fileSet.Position(call.End()).Offset,
				replacement: replacement,
			})
			return true
		})
		slices.SortFunc(edits, func(left, right modernGoSourceEdit) int { return left.start - right.start })
		name := filepath.Join(root, filepath.FromSlash(file.source.Path))
		current, err := os.ReadFile(name)
		if err != nil {
			return total, changed, err
		}
		if !bytes.Equal(current, file.source.blob.data) {
			return total, changed, fmt.Errorf("modern-Go testing context source changed during rewrite: %s", file.source.Path)
		}
		var rewritten bytes.Buffer
		prior := 0
		for _, edit := range edits {
			if edit.start < prior || edit.end < edit.start || edit.end > len(current) {
				return total, changed, fmt.Errorf("modern-Go testing context edit overlaps in %s", file.source.Path)
			}
			rewritten.Write(current[prior:edit.start])
			rewritten.WriteString(edit.replacement)
			prior = edit.end
		}
		rewritten.Write(current[prior:])
		updated := rewritten.Bytes()
		if needsContextImport {
			updated, err = ensureContextImport(name, updated)
			if err != nil {
				return total, changed, err
			}
		}
		cleaned, err := removeUnusedContextImport(name, updated)
		if err != nil {
			return total, changed, err
		}
		if bytes.Equal(current, cleaned) {
			continue
		}
		info, err := os.Stat(name)
		if err != nil {
			return total, changed, err
		}
		if err := os.WriteFile(name, cleaned, info.Mode()); err != nil {
			return total, changed, err
		}
		total += len(edits)
		changed = append(changed, file.source.Path)
	}
	slices.Sort(changed)
	return total, changed, nil
}

func modernGoParents(root ast.Node) map[ast.Node]ast.Node {
	parents := map[ast.Node]ast.Node{}
	ast.PreorderStack(root, nil, func(node ast.Node, stack []ast.Node) bool {
		if len(stack) != 0 {
			parents[node] = stack[len(stack)-1]
		}
		return true
	})
	return parents
}

func modernGoCleanupSensitiveContext(contextCall *ast.CallExpr, parent ast.Node) bool {
	call, ok := parent.(*ast.CallExpr)
	if !ok || !slices.Contains(call.Args, ast.Expr(contextCall)) {
		return false
	}
	selector, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	return slices.Contains([]string{
		"AllocateDeviceBuffer", "Execute", "ExecuteProgram", "ExecuteCompiled", "ExecuteRetained",
		"ExecuteRetainedCompiled", "executeWithDeviceFeeds", "executeRetainedWithDeviceFeeds",
	}, selector.Sel.Name)
}

func modernGoContextInCleanup(node ast.Node, parents map[ast.Node]ast.Node) bool {
	for current := parents[node]; current != nil; current = parents[current] {
		call, ok := current.(*ast.CallExpr)
		if !ok {
			continue
		}
		selector, selectorOK := call.Fun.(*ast.SelectorExpr)
		identifier, identifierOK := call.Fun.(*ast.Ident)
		if selectorOK && selector.Sel.Name == "Cleanup" ||
			identifierOK && strings.HasSuffix(strings.ToLower(identifier.Name), "cleanup") {
			return true
		}
	}
	return false
}

func ensureContextImport(name string, source []byte) ([]byte, error) {
	return ensureNamedImport(name, source, "context")
}

func ensureNamedImport(name string, source []byte, importPath string) ([]byte, error) {
	fileSet := token.NewFileSet()
	file, err := parser.ParseFile(fileSet, name, source, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	for _, imported := range file.Imports {
		if path, pathErr := strconv.Unquote(imported.Path.Value); pathErr == nil && path == importPath {
			return source, nil
		}
	}
	for _, declaration := range file.Decls {
		imports, ok := declaration.(*ast.GenDecl)
		if !ok || imports.Tok != token.IMPORT {
			continue
		}
		if imports.Lparen.IsValid() {
			offset := fileSet.Position(imports.Lparen).Offset + 1
			withImport := append(slices.Clone(source[:offset]), append([]byte("\n\t"+strconv.Quote(importPath)), source[offset:]...)...)
			return format.Source(withImport)
		}
		start, end := fileSet.Position(imports.Pos()).Offset, fileSet.Position(imports.End()).Offset
		existing := source[fileSet.Position(imports.Specs[0].Pos()).Offset:fileSet.Position(imports.Specs[0].End()).Offset]
		replacement := append([]byte("import (\n\t"+strconv.Quote(importPath)+"\n\t"), existing...)
		replacement = append(replacement, []byte("\n)")...)
		withImport := append(slices.Clone(source[:start]), append(replacement, source[end:]...)...)
		return format.Source(withImport)
	}
	offset := fileSet.Position(file.Name.End()).Offset
	withImport := append(slices.Clone(source[:offset]), append([]byte("\n\nimport "+strconv.Quote(importPath)), source[offset:]...)...)
	return format.Source(withImport)
}

func removeUnusedContextImport(name string, source []byte) ([]byte, error) {
	return removeUnusedNamedImport(name, source, "context")
}

func removeUnusedNamedImport(name string, source []byte, wantedPath string) ([]byte, error) {
	fileSet := token.NewFileSet()
	file, err := parser.ParseFile(fileSet, name, source, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	for _, declaration := range file.Decls {
		imports, ok := declaration.(*ast.GenDecl)
		if !ok || imports.Tok != token.IMPORT {
			continue
		}
		for _, specification := range imports.Specs {
			candidate := specification.(*ast.ImportSpec)
			importPath, err := strconv.Unquote(candidate.Path.Value)
			if err != nil || importPath != wantedPath {
				continue
			}
			importName := wantedPath
			if candidate.Name != nil {
				importName = candidate.Name.Name
			}
			if importName == "." || importName == "_" || astUsesQualifier(file, importName) {
				return source, nil
			}
			start, end := candidate.Pos(), candidate.End()
			if len(imports.Specs) == 1 {
				start, end = imports.Pos(), imports.End()
			} else {
				if candidate.Doc != nil {
					start = candidate.Doc.Pos()
				}
				if candidate.Comment != nil {
					end = candidate.Comment.End()
				}
			}
			startOffset := fileSet.Position(start).Offset
			endOffset := fileSet.Position(end).Offset
			for startOffset > 0 && source[startOffset-1] != '\n' {
				startOffset--
			}
			for endOffset < len(source) && source[endOffset] != '\n' {
				endOffset++
			}
			if endOffset < len(source) {
				endOffset++
			}
			without := append(slices.Clone(source[:startOffset]), source[endOffset:]...)
			formatted, err := format.Source(without)
			if err != nil {
				return nil, fmt.Errorf("format source after removing unused %s import: %w", wantedPath, err)
			}
			return formatted, nil
		}
	}
	return source, nil
}

func astUsesQualifier(file *ast.File, name string) bool {
	used := false
	ast.Inspect(file, func(node ast.Node) bool {
		if used {
			return false
		}
		selector, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		identifier, ok := selector.X.(*ast.Ident)
		used = ok && identifier.Name == name
		return !used
	})
	return used
}

func modernGoUsesContextWithoutCancel(file *ast.File) bool {
	found := false
	ast.Inspect(file, func(node ast.Node) bool {
		if found {
			return false
		}
		selector, ok := node.(*ast.SelectorExpr)
		if !ok || selector.Sel.Name != "WithoutCancel" {
			return true
		}
		identifier, ok := selector.X.(*ast.Ident)
		found = ok && identifier.Name == "context"
		return !found
	})
	return found
}
