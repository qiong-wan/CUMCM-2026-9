import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const out = path.join(root, 'output/problem-3');
const previews = path.join(out, 'previews');
await fs.mkdir(previews, { recursive: true });
const workbook = await SpreadsheetFile.importXlsx(
  await FileBlob.load(path.join(root, 'data/附件3/result3.xlsx')),
);
const sheet = workbook.worksheets.getItem('Sheet1');
if (process.argv.includes('--preview-template')) {
  console.log((await workbook.inspect({ kind: 'region', sheetId: 'Sheet1',
    range: 'A1:F5', maxChars: 2000 })).ndjson);
  const preview = await workbook.render({ sheetName: 'Sheet1', range: 'A1:F5', scale: 2 });
  await fs.writeFile(path.join(previews, 'template_preview.png'), new Uint8Array(await preview.arrayBuffer()));
} else {
  const payload = JSON.parse(await fs.readFile(path.join(out, 'workbook_payload.json'), 'utf8'));
  if (sheet.getRange('A1').values[0][0] !== payload.header[0]) throw new Error('Template A1 mismatch');
  const last = payload.rows.length + 1;
  sheet.getUsedRange().clear({ applyTo: 'contents' });
  sheet.getRange('A1:V1').values = [payload.header];
  sheet.getRange(`A2:V${last}`).values = payload.rows;
  sheet.getRange(`B2:V${last}`).setNumberFormat('0.0000');
  sheet.getRange(`A2:A${last - 1}`).setNumberFormat('0');
  sheet.getRange(`A${last}`).setNumberFormat('0.000000');
  sheet.getRange('B1:V1').setNumberFormat('0.0');
  sheet.getRange(`A1:A${last}`).format.columnWidth = 32;
  sheet.getRange(`B1:V${last}`).format.columnWidth = 10;
  sheet.getRange('A1:V1').format.rowHeight = 34;
  sheet.getRange('A1:V1').format.wrapText = true;
  sheet.getRange(`A2:V${last}`).format.rowHeight = 20;
  sheet.getRange(`A2:V${last}`).format.horizontalAlignment = 'center';
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  const checks = [];
  checks.push((await workbook.inspect({ kind: 'table', range: 'Sheet1!A1:H5',
    include: 'values', tableMaxRows: 5, tableMaxCols: 8, maxChars: 2000 })).ndjson);
  checks.push((await workbook.inspect({ kind: 'match',
    searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
    options: { useRegex: true, maxResults: 20 }, maxChars: 1000 })).ndjson);
  for (const [name, range] of [['workbook_preview', 'A1:V8'],
    ['workbook_end_preview', `A${last - 4}:H${last}`]]) {
    const preview = await workbook.render({ sheetName: 'Sheet1', range, scale: 1.5 });
    await fs.writeFile(path.join(previews, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  const result = await SpreadsheetFile.exportXlsx(workbook);
  await result.save(path.join(out, 'result3.xlsx'));
  await fs.rm(path.join(out, 'result3.xlsx.inspect.ndjson'), { force: true });
  await fs.writeFile(path.join(out, 'artifact_inspection.txt'), checks.join('\n'));
  console.log(`Exported Sheet1: ${last} rows, 22 columns; final time ${payload.rows.at(-1)[0]} s`);
}
