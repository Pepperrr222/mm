import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';
const root=path.resolve(import.meta.dirname,'../..');
const book=await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(root,'01_题目与数据/原始附件/附件5/result3.xlsx')));
console.log((await book.inspect({kind:'sheet',include:'id,name',maxChars:1200})).ndjson);
const preview=await book.render({sheetName:'充放电量',range:'A1:F8',scale:1.5});
await fs.writeFile(path.join(import.meta.dirname,'results/template_preview.png'),new Uint8Array(await preview.arrayBuffer()));
console.log('Template preview saved.');
