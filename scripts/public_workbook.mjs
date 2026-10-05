import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [source, planFile, destination, previewDirectory] = process.argv.slice(2);
if (!source || !planFile || !destination || !previewDirectory) {
  throw new Error("Usage: node public_workbook.mjs SOURCE PLAN DESTINATION PREVIEW_DIRECTORY");
}
if (source === destination) throw new Error("Keep the source workbook unchanged.");
const plan = JSON.parse(await fs.readFile(planFile, "utf8"));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
for (const edit of plan.workbook_edits) {
  workbook.worksheets.getItem(edit.sheet).getRange(edit.cell).values = [[edit.value]];
}
workbook.recalculate();
await fs.mkdir(previewDirectory, { recursive: true });
for (const [sheetName, range] of [
  ["Table_S5", "A10:F15"], ["Table_S23", "A7:C12"],
  ["S23_numeric", "A7:C12"], ["Table_S24", "A8:H13"], ["S24_numeric", "A8:H13"],
]) {
  const result = await workbook.inspect({
    kind: "table", range: sheetName + "!" + range, include: "values,formulas",
    tableMaxRows: 6, tableMaxCols: 8, maxChars: 2000,
  });
  await fs.writeFile(previewDirectory + "/" + sheetName + ".ndjson", result.ndjson);
  const preview = await workbook.render({ sheetName, range, scale: 1.3, format: "png" });
  await fs.writeFile(previewDirectory + "/" + sheetName + ".png",
                     new Uint8Array(await preview.arrayBuffer()));
}
const errors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 50 }, maxChars: 2500,
});
await fs.writeFile(previewDirectory + "/formula_errors.ndjson", errors.ndjson);
await (await SpreadsheetFile.exportXlsx(workbook)).save(destination);
console.log(JSON.stringify({ edits: plan.workbook_edits.length, exported: true }));
