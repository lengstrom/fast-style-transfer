import pkg from 'xlsx';
const { readFile, utils } = pkg;

const filePath = "C:\\Users\\Hernán\\Documents\\TRONADOR APP\\TRANSFER\\excelpvlopca1.xls";

try {
    console.log("Attempting to read:", filePath);
    const workbook = readFile(filePath);
    const sheetName = workbook.SheetNames[0];
    console.log("Sheet Name:", sheetName);
    const sheet = workbook.Sheets[sheetName];

    // Get headers (first row)
    const headers = utils.sheet_to_json(sheet, { header: 1 })[0];

    // Get raw JSON to see data structure
    const jsonData = utils.sheet_to_json(sheet);

    console.log("--- EXCEL INSPECTION ---");
    console.log("Headers:", headers);
    console.log("First Row Sample:", jsonData[0]);
    console.log("------------------------");

} catch (error) {
    console.error("Error reading file:", error.message);
    console.error(error);
}
